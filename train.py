import time, os, argparse, shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from glob import glob
import random
from GraphDataset import get_data_loader,get_test_dataloader
from net_utils import train_epoch, val_epoch, test_epoch
from tools import get_logger, visualize_training_history, draw_attentionmap
from model.model_GA_DIFFPOOL import GAN
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime

np.seterr(divide='ignore',invalid='ignore')
import warnings
warnings.filterwarnings("ignore")

def open_log(log_savepath):
    # log_savepath = os.path.join(log_path, name)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    log_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if os.path.isfile(os.path.join(log_savepath, '{}.log'.format(log_name))):
        os.remove(os.path.join(log_savepath, '{}.log'.format(log_name)))
    initLogging(os.path.join(log_savepath, '{}.log'.format(log_name)))
def increment_dir(dir, e=None):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    d = sorted(glob('output/'+dir + '*'))  # directories
    print(d)
    if len(d):
        d = d[-1].split('-')[-1][:3]
        n = int(d) + 1  # increment
        print(n)
    if e is not None:
#        os.makedirs(dir + '-' + str(n).zfill(3) + e, exist_ok=True)
        return dir + '-' + str(n).zfill(3) + e
    else:
#        os.makedirs(dir + '-' + str(n).zfill(3), exist_ok=True)
        return dir + '-' + str(n).zfill(3)
def plot_5fold_roc(score_list, label_list, aucs, save_path):
    tprs = []
    for i in range(5):
        score = score_list[i]
        label = label_list[i]
        fpr, tpr, thresholds = metrics.roc_curve(label, score)
        fpr_mean = np.linspace(0, 1, 100)
        tprs.append(np.interp(fpr_mean, fpr, tpr))
        plt.plot(fpr_mean, np.interp(fpr_mean, fpr, tpr), lw=1, alpha=0.3, label='fold%d (AUC = %0.2f)' % (i, aucs[i]))

    tpr_mean = np.mean(tprs, axis=0)
    tpr_mean[-1] = 1
    tpr_std = np.std(tprs, axis=0)
    tprs_upper = np.minimum(tpr_mean + tpr_std, 1)
    tprs_lower = np.maximum(tpr_mean - tpr_std, 0)
    plt.plot(fpr_mean, tpr_mean, color='b',
             label=r'MeanROC (AUC=%0.4f $\pm$ %0.4f)' % (np.mean(aucs), np.std(aucs)),
             lw=2, alpha=.8)
    plt.fill_between(fpr_mean, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.legend()
    # plt.show()
    plt.savefig(save_path+'/roc.png', dpi=100)
    plt.close()
    np.savez(save_path+'/roc.npz', fpr_mean=fpr_mean, tpr_mean=tpr_mean, aucs=aucs)
# Init for logging
def initLogging(logFilename):
    logging.basicConfig(level    = logging.INFO,
                        format   = '[%(asctime)s-%(levelname)s] %(message)s',
                        datefmt  = '%y-%m-%d %H:%M:%S',
                        filename = logFilename,
                        filemode = 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def train(cosine_loss,i_fold,save_path,DEVICE,weight_decay,lr,lr_decay,train_epochs,node_num,feat_dim,hidden_num,class_num,nheads,dropout,bs):
    os.system(f'cp -r model {save_path}/code')
    os.system(f'cp  train.py {save_path}/code')
    os.system(f'cp  GraphDataset.py {save_path}/code')
    os.system(f'cp  tools.py {save_path}/code')
    os.system(f'cp  net_utils.py {save_path}/code')
    
    save_path = f'{save_path}/fold{i_fold}'
    os.makedirs(os.path.join(save_path), exist_ok=True)
    logging.info(i_fold)
    # writer1 = SummaryWriter('./resultandinew/log1')
    # writer2 = SummaryWriter('./resultandinew/log2')

    best_val_loss, is_best_loss, best_test_loss, is_test_best_loss,best_loss_epoch,best_test_loss_epoch = 2 ** 20, False,2 ** 20,False,0, 0
    best_score_acc, is_best_score, best_score_epoch ,best_test_acc,is_test_best_score,best_test_score_epoch= -2 ** 20, False, 0,-2 ** 20, False, 0
    best_score_auc,best_test_score_auc = -2 ** 20,-2 ** 20
    history = pd.DataFrame()

    def save_model(model, optimizer, lr_scheduler, best_score_acc,is_best_loss=False, is_best_score=False,is_test_best_loss=False,is_test_best_score=False):

        model_state = {'state_dict': model.state_dict(), 'epoch': epoch,
                       'history': history, 'i_fold': i_fold,
                       'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc,
                       'best_val_loss': best_val_loss, 'best_loss_epoch': best_loss_epoch,
                       'best_score_acc': best_score_acc, 'best_score_epoch': best_score_epoch,
                       'preds': preds, 'targets': targets, 'preds_score': preds_score,
                       'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict()
                       }

        model_path = os.path.join(save_path, f"{network_name}.pth")
        torch.save(model_state, model_path)
        if is_best_loss:
            best_model_path = os.path.join(save_path, f"{network_name}_best_loss.pth")
            shutil.copy(model_path, best_model_path)

            # attentionlist = []
            # for attention in attentions:
            #     attentionlist.append(attention.detach().cpu().numpy())
            # attentionarray = np.array(attentionlist, dtype=np.float32)
            # np.save(os.path.join(save_path, 'attention_best_loss.npy'), attentionarray)
            # draw_attentionmap(attentionarray, os.path.join(save_path, 'attention_best_loss.png'))
        if is_best_score:
            best_model_path = os.path.join(save_path, f"{network_name}_best_score.pth")
            shutil.copy(model_path, best_model_path)

            # attentionlist = []
            # for attention in attentions:
            #     attentionlist.append(attention.detach().cpu().numpy())
            # attentionarray = np.array(attentionlist, dtype=np.float32)
            # np.save(os.path.join(save_path, 'attention_best_score.npy'), attentionarray)
            # draw_attentionmap(attentionarray, os.path.join(save_path, 'attention_best_score.png'))
        if is_test_best_score:
            test_best_model_path = os.path.join(save_path, f"{network_name}_{best_score_acc:.2f}.pth")
            shutil.copy(model_path, test_best_model_path)
    TrainLoader, ValLoader, TestLoader = get_data_loader(i_fold,node_num,bs)
    # TestLoader = get_test_dataloader(node_num,1)
    # model = GCN(feat_dim=2, node_num=116, assign_ratio=0.5, class_num=3).to(DEVICE)
    model = GAN(feat_dim=feat_dim, node_num=node_num, hidden_num=hidden_num, class_num=class_num,nheads=nheads,dropout=dropout).to(DEVICE)
    # model = GraphConvModel(feat_dim=3, node_num=116, assign_ratio=0.5, class_num=2).to(DEVICE)
    # model = GAT(nfeat=3, nhid=3, nclass=2).to(DEVICE)

    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_epochs, eta_min=lr_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=83, gamma=0.5)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20)

    

    for epoch in range(train_epochs):
        cur_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc,CE_loss,Pos_loss,Neg_loss = train_epoch(cosine_loss,model, TrainLoader, DEVICE, optimizer,i_fold,save_path,epoch)
        val_loss, val_acc, score_auc, preds, preds_score, targets, risks,val_CE_loss,val_Pos_loss,val_Neg_loss = val_epoch(cosine_loss,model, ValLoader, DEVICE,epoch,i_fold,save_path)
        # test_loss, test_acc, test_score_auc, test_preds, test_preds_score, test_targets, test_risks,test_attentions = test_epoch(model, TestLoader, DEVICE)

        is_best_loss, is_best_score = val_loss < best_val_loss, val_acc > best_score_acc
        best_val_loss, best_score_acc = min(val_loss, best_val_loss), max(val_acc, best_score_acc)

        # is_test_best_loss, is_test_best_score = test_loss < best_test_loss, test_acc > best_test_acc
        # best_test_loss, best_test_acc = min(test_loss, best_test_loss), max(test_acc, best_test_acc)
        
        lr_scheduler.step()

        _h = pd.DataFrame(
            {'lr': [cur_lr], 'train_loss': [train_loss], 'train_acc': [train_acc], 'val_loss': [val_loss], 'val_acc': [val_acc],
            })
        """ _h = pd.DataFrame(
            {'lr': [cur_lr], 'train_loss': [train_loss], 'train_acc': [train_acc], 'val_loss': [val_loss], 'val_acc': [val_acc],
            'test_loss':[test_loss],'test_acc':[test_acc]}) """
        history = history.append(_h, ignore_index=True)
        visualize_training_history(history, save_path=os.path.join(save_path, f"history_{network_name}.png"))
        history.to_csv(os.path.join(save_path, f"history_{network_name}.csv"))

        # msg = f"Epoch{epoch}, lr:{cur_lr:.4f}, train_loss:{train_loss:.4f}, train_acc:{train_acc:.4f}, val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}, test_loss:{test_loss:.4f}, test_acc:{test_acc:.4f}"
        msg = f"Epoch{epoch}, lr:{cur_lr:.4f}, train_loss:{train_loss:.4f}, CE_loss:{CE_loss:.4f},Pos_loss:{Pos_loss:.4f},Neg_loss:{Neg_loss:.4f},train_acc:{train_acc:.4f}, val_loss:{val_loss:.4f}, val_CE_loss:{val_CE_loss:.4f},val_Pos_loss:{val_Pos_loss:.4f},val_Neg_loss:{val_Neg_loss:.4f},val_acc:{val_acc:.4f}, val_auc:{score_auc:.4f}, high_risk:{risks}"
        if is_best_loss:
            best_loss_epoch, msg = epoch, msg + "  => best loss"
        if is_best_score:
            best_score_epoch, msg = epoch, msg + "  => best score"
            best_score_auc = score_auc
        # if is_test_best_score:
        #     best_test_score_epoch,msg = epoch, msg+"  => best test score "
        logging.info(msg)
        save_model(model, optimizer, lr_scheduler,best_score_acc, is_best_loss, is_best_score,is_test_best_loss,is_test_best_score)
        
    test_model_path = os.path.join(save_path, f"{network_name}_best_score.pth")
    test_model = GAN(feat_dim=feat_dim, node_num=node_num, hidden_num=hidden_num, class_num=class_num,nheads=nheads,dropout=dropout).to(DEVICE)
    test_model.load_state_dict(torch.load(test_model_path)["state_dict"])
    test_loss, test_acc, test_auc, test_preds, test_preds_score, test_targets, test_risks,_,_,_,test_score_pre,test_score_rec,test_score_f1=test_epoch(cosine_loss,test_model,TestLoader,DEVICE)
    logging.info(f'best_test_epoch{best_score_epoch},test_acc{test_acc:.4f},test_auc{test_auc:.4f}')

    return test_preds_score,best_score_acc, best_score_auc,test_acc,test_auc,test_preds,test_targets,test_score_pre,test_score_rec,test_score_f1


if __name__ == '__main__':
    # Training settings
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # set seed
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_decay = 0.0001
    lr = 4e-5
    lr_decay = 4e-5
    train_epochs = 50
    risk=111
    node_num = 20
    feat_dim = 2048
    hidden_num = 512
    class_num = 2
    nheads = 8
    dropout = 0.2
    bs = 128
    cosine_loss = True
    network_name = increment_dir(f'seed{args.seed}_lr{lr:.1e}_lr_decay{lr_decay:.1e}_wd{weight_decay}' \
                f'_node{node_num}_hidden_num{hidden_num}_bs{bs}_nheads{nheads}_dropout{dropout}')
    save_path = f'output/{network_name}'
    os.makedirs(os.path.join(save_path), exist_ok=True)
    open_log(save_path)
    accs = []
    aucs = []
    test_accs = []
    test_aucs = []
    test_score_pres,test_score_recs,test_score_f1s=[],[],[]
    test_preds_scores,test_targets =[],[]
    for i_fold in range(1,6):
        test_preds_score,best_score_acc, best_score_auc,test_acc,test_auc,test_preds,test_target,test_score_pre,test_score_rec,test_score_f1 = train(cosine_loss,i_fold,save_path,DEVICE,weight_decay,lr,lr_decay,train_epochs,node_num,feat_dim,hidden_num,class_num,nheads,dropout,bs)
        test_targets.append(test_target)
        test_preds_scores.append(test_preds_score)

        accs.append(best_score_acc)
        aucs.append(best_score_auc)
        test_accs.append(test_acc)
        test_aucs.append(test_auc)
        test_score_pres.append(test_score_pre)
        test_score_f1s.append(test_score_f1)
        test_score_recs.append(test_score_rec)
    plot_5fold_roc(test_preds_scores,test_targets,test_aucs,save_path)
    logging.info(accs)
    logging.info(aucs)
    logging.info(f'val_ACC {np.array(accs).mean():.4f}±{np.array(accs).std():.4f}')
    logging.info(f'val_AUC {np.array(aucs).mean():.4f}±{np.array(aucs).std():.4f}')
    logging.info(test_accs)
    logging.info(test_aucs)
    logging.info(test_score_recs)
    logging.info(test_score_pres)
    logging.info(test_score_f1s)
    logging.info(f'test_ACC {np.array(test_accs).mean():.4f}±{np.array(test_accs).std():.4f}')
    logging.info(f'test_AUC {np.array(test_aucs).mean():.4f}±{np.array(test_aucs).std():.4f}')
    logging.info(f'test_recall {np.array(test_score_recs).mean():.4f}±{np.array(test_score_recs).std():.4f}')
    logging.info(f'test_precision {np.array(test_score_pres).mean():.4f}±{np.array(test_score_pres).std():.4f}')
    logging.info(f'test_f1 {np.array(test_score_f1s).mean():.4f}±{np.array(test_score_f1s).std():.4f}')

    torch.cuda.empty_cache()
