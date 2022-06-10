import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm as tqdmauto
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score,precision_recall_fscore_support
from sklearn import manifold
import os
def train_epoch(cosine_loss,model, loader, device, optimizer,i_fold,save_path,epoch, verbose=False):
    model.train()
    train_losses = []
    CE_losses,Link_losses,Pos_Losses,Neg_Losses = [],[],[],[]
    pred_list, target_list = [], []
    out_pos_features,out_neg_features=[],[]
    optimizer.zero_grad()
    progress_bar = tqdmauto(loader) if verbose else None
    flag = 0
    for batch_idx, (fc_matrixs, fc_matrixs_neg,feature, feature_neg,targets,risks) in enumerate(loader):
        # print(fc_matrixs.shape, feature.shape, targets.shape)
        # print(batch_idx)
        # batchsize, node_num, _ = fc_matrixs.shape
        fc_matrixs_batch = fc_matrixs.to(device)
        fc_matrixs_neg_batch = fc_matrixs_neg.to(device)
        targets_batch = targets.to(device)
        feature_batch = feature.to(device)
        feature_neg_batch = feature_neg.to(device)
        risks_batch = risks.to(device)
        targets_batch = torch.as_tensor(targets_batch, dtype=torch.long).to(device)
        risks_batch = torch.as_tensor(risks_batch,dtype=torch.long).to(device)
        preds, cosine_similarity,out_pos_feature, out_neg_feature = model(feature_batch,feature_neg_batch, fc_matrixs_batch,fc_matrixs_neg_batch)
        # print(cosine_similarity)
        distance = 1-cosine_similarity
        # print(targets_batch)
        # targets_batch_neg = 
        out_pos_feature = out_pos_feature.cpu().detach().numpy()
        out_neg_feature = out_neg_feature.cpu().detach().numpy()
        out_pos_features.append(out_pos_feature)
        out_neg_features.append(out_neg_feature)

        pos_batch = (targets_batch == 1).nonzero(as_tuple=True)
        neg_batch = (targets_batch == 0).nonzero(as_tuple=True)
        pos_loss,neg_loss = 0,0
        for i in pos_batch:
            # pos_loss += torch.max(torch.tensor(0).float().to('cuda'),-distance[i]+0.5)
            pos_loss += -distance[i]+2
        for j in neg_batch:
            neg_loss += distance[j]

        # pos_loss = 20*1/5*torch.log(1+5*torch.sum(torch.exp(targets_batch*(cosine_similarity-0.2))))/len(pos_batch[0])
        # neg_loss = 20*1/2*torch.log(1+2*torch.sum(torch.exp((targets_batch-1)*(cosine_similarity-0.2))))/len(neg_batch[0])
        loss = nn.CrossEntropyLoss(reduction='none')(preds, targets_batch) 
        
        # loss = loss * risks_batch
        # print(loss)
        loss = loss.mean()
        # print('CELoss:',loss)
        CE_losses.append(loss.detach().cpu().item())
        
        if cosine_loss:
            loss += pos_loss.mean() + neg_loss.mean()
            # loss += neg_loss.mean()
        Pos_Losses.append(pos_loss.mean().detach().cpu().item())
        Neg_Losses.append(neg_loss.mean().detach().cpu().item())
        # print('PosLoss:',pos_loss)
        # print('NegLoss:',neg_loss)
        loss_np = loss.detach().cpu().item()
        train_losses.append(loss_np)

        preds_score = F.softmax(preds, dim=1).detach().cpu().numpy()
        preds = np.argmax(preds_score, axis=1)
        pred_list.append(preds)
        targets = targets.detach().cpu().numpy()
        target_list.append(targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose:
            progress_bar.set_postfix_str(f"loss: {loss_np:.4f}, smooth_loss: {np.mean(train_losses[-20:]):.4f}")
            progress_bar.update(1)
    if verbose:
        progress_bar.close()

    preds = np.concatenate(pred_list)
    targets = np.concatenate(target_list)
    score_acc = accuracy_score(targets, preds)
    out_pos_features = np.concatenate(out_pos_features)
    out_neg_features = np.concatenate(out_neg_features)
    feature_for_tsne = np.concatenate((out_pos_features,out_neg_features))
    if epoch % 5 == 0 and i_fold==1:
        labels_pos_draw = targets
        labels_neg_draw = targets+2
        labels_draw = np.concatenate((labels_pos_draw,labels_neg_draw))
        # print(labels_draw.shape)
        # print(feature_for_tsne.shape)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        # print('start')
        X_tsne = tsne.fit_transform(feature_for_tsne)
        # print('start painting')
        #print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
        '''嵌入空间可视化'''
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        # print(X_norm.shape)
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            if str(labels_draw[i])==str(0):
                color_ = 'b'
                label_ = 'sample_neg_pos'
            elif str(int(labels_draw[i]))==str(1):
                color_ = 'g'
                label_ = 'sample_pos_pos'
            elif str(int(labels_draw[i]))==str(2):
                color_ = 'r'
                label_ = 'sample_neg_neg'
            elif str(int(labels_draw[i]))==str(3):
                color_ = 'y'
                label_ = 'sample_pos_neg'
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=2,c=color_)
            # plt.text(X_norm[i, 0], X_norm[i, 1], str(labels_draw[i]), color=color_,
            #         fontdict={'weight': 'bold', 'size': 9})
        font={'size':23}
        plt.title('Feature Space',fontdict = font)
        plt.xticks([])
        plt.yticks([])
        os.makedirs(os.path.join(save_path,'tsne'), exist_ok=True)
        savepath = save_path+'/tsne/'+'train_'+str(epoch)+'.png'
        #plt.show()
        plt.savefig(savepath)
        plt.close('all')
        # print('painting ok')
    # print('total loss:',np.asarray(train_losses).mean())
    # print('CE loss:',np.asarray(CE_losses).mean())
    # # print('Link loss:',np.asarray(Link_losses).mean())
    # print('Pos loss:',np.asarray(Pos_Losses).mean())
    # print('Neg loss:',np.asarray(Neg_Losses).mean())
    return np.asarray(train_losses).mean(), score_acc,np.asarray(CE_losses).mean(),np.asarray(Pos_Losses).mean(),np.asarray(Neg_Losses).mean()


def val_epoch(cosine_loss,model, loader, device,epoch,i_fold,save_path):
    model.eval()
    val_losses = []
    CE_losses,Pos_Losses,Neg_Losses = [],[],[]
    pred_list, target_list,risk_list = [], [],[]
    out_pos_features,out_neg_features=[],[]
    preds_score_list = []
    with torch.no_grad():
        for fc_matrixs, fc_matrixs_neg,feature, feature_neg,targets,risks in loader:
            # batchsize, node_num = fc_matrixs.shape
            fc_matrixs_batch = fc_matrixs.to(device)
            fc_matrixs_neg_batch = fc_matrixs_neg.to(device)
            feature_neg_batch = feature_neg.to(device)
            targets_batch = targets.to(device)
            feature_batch = feature.to(device)
            targets_batch = torch.as_tensor(targets_batch, dtype=torch.long).to(device)
            risks_batch = risks.to(device)
            risks_batch = torch.as_tensor(risks_batch,dtype=torch.long).to(device)
            preds, cosine_similarity,out_pos_feature, out_neg_feature = model(feature_batch,feature_neg_batch, fc_matrixs_batch,fc_matrixs_neg_batch)
            pos_batch = (targets_batch == 1).nonzero(as_tuple=True)
            out_pos_feature = out_pos_feature.cpu().detach().numpy()
            out_neg_feature = out_neg_feature.cpu().detach().numpy()
            out_pos_features.append(out_pos_feature)
            out_neg_features.append(out_neg_feature)
            neg_batch = (targets_batch == 0).nonzero(as_tuple=True)

            distance = 1-cosine_similarity
            # print(distance)
            pos_loss,neg_loss = 0,0
            if len(pos_batch[0]) != 0:
                # print(distance)
                # pos_loss += torch.max(torch.tensor(0).float().to('cuda'),-distance[i]+0.5)
                pos_loss += -distance[0]+2
                Pos_Losses.append(pos_loss.detach().cpu().item())
                # print(pos_loss)
            if len(neg_batch[0]) != 0:
                # print(distance)
                neg_loss += distance[0]
                Neg_Losses.append(neg_loss.detach().cpu().item())
                # print(neg_loss)
            # if len(pos_batch[0]) != 0:
            #     pos_loss = 20*1/5*torch.log(1+5*torch.sum(torch.exp(targets_batch*(cosine_similarity-0.5))))/len(pos_batch[0])
            # else:
            #     pos_loss = 0
            # if len(neg_batch[0]) != 0:
            #     neg_loss = 20*1/2*torch.log(1+2*torch.sum(torch.exp((targets_batch-1)*(cosine_similarity-0.5))))/len(neg_batch[0])
            # else:
            #     neg_loss = 0
            # loss = nn.CrossEntropyLoss()(preds, targets_batch) + 0.5 * link_loss
            # print(preds)
            loss = nn.CrossEntropyLoss(reduction='none')(preds, targets_batch) 
            # print(loss)
            # loss = loss * risks_batch
            # print(loss)
            loss = loss.mean()
            # print(loss)
            # loss +=  0.25 * link_loss
            
            CE_losses.append(loss.detach().cpu().item())
            if cosine_loss:
                loss += pos_loss + neg_loss
                # loss += neg_loss
            # print(loss)
            
            
            # print('PosLoss:',pos_loss)
            # print('NegLoss:',neg_loss)
            loss_np = loss.detach().cpu().item()
            val_losses.append(loss_np)

            preds_score = F.softmax(preds, dim=1).cpu().numpy()
            preds = np.argmax(preds_score, axis=1)
            risks = risks.cpu().numpy()
            targets = targets.cpu().numpy()
            
            pred_list.append(preds)
            target_list.append(targets)
            risk_list.append(risks)
            preds_score_list.append(preds_score[:, 1])#
    risks= np.concatenate(risk_list).squeeze()
    preds = np.concatenate(pred_list)
    preds_score = np.concatenate(preds_score_list)
    targets = np.concatenate(target_list)
    preds = preds.squeeze()
    targets = targets.squeeze()
    feature_for_tsne = np.concatenate((out_pos_features,out_neg_features),axis=0).squeeze()
    # print(feature_for_tsne.shape)
    if epoch % 5 == 0:
        labels_pos_draw = targets
        labels_neg_draw = targets+2
        labels_draw = np.concatenate((labels_pos_draw,labels_neg_draw))
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(feature_for_tsne)
        #print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
        '''嵌入空间可视化'''
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        # print(X_norm.shape)
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            if str(labels_draw[i])==str(0):
                color_ = 'b'
                label_ = 'sample_neg_pos'
            elif str(int(labels_draw[i]))==str(1):
                color_ = 'g'
                label_ = 'sample_pos_pos'
            elif str(int(labels_draw[i]))==str(2):
                color_ = 'r'
                label_ = 'sample_neg_neg'
            elif str(int(labels_draw[i]))==str(3):
                color_ = 'y'
                label_ = 'sample_pos_neg'
            plt.scatter(X_norm[i, 0], X_norm[i, 1], s=2,c=color_)
            # plt.text(X_norm[i, 0], X_norm[i, 1], str(labels_draw[i]), color=color_,
            #         fontdict={'weight': 'bold', 'size': 9})
        font={'size':23}
        plt.title('Feature Space',fontdict = font)
        plt.xticks([])
        plt.yticks([])
        os.makedirs(os.path.join(save_path,'tsne'), exist_ok=True)
        savepath = save_path+'/tsne/'+'valid_'+str(epoch)+'.png'
        #plt.show()
        plt.savefig(savepath)
        plt.close('all')
    risk = 0
    for i,num in enumerate(targets):
        if targets[i] == 1:
            if risks[i]==5 and preds[i]==0:
                risk+=1
    # print(preds_score)
    # print(targets)
    score_acc = accuracy_score(targets, preds)
    score_auc = roc_auc_score(targets, preds_score)
    return np.asarray(val_losses).mean(), score_acc, score_auc, preds, preds_score, targets, risk,np.asarray(CE_losses).mean(),np.asarray(Pos_Losses).mean(),np.asarray(Neg_Losses).mean()

def test_epoch(cosine_loss,model, loader, device):
    model.eval()
    val_losses = []
    CE_losses,Pos_Losses,Neg_Losses = [],[],[]
    pred_list, target_list,risk_list = [], [],[]
    
    preds_score_list = []
    with torch.no_grad():
        for fc_matrixs, fc_matrixs_neg,feature, feature_neg,targets,risks in loader:
            # batchsize, node_num = fc_matrixs.shape
            fc_matrixs_batch = fc_matrixs.to(device)
            fc_matrixs_neg_batch = fc_matrixs_neg.to(device)
            feature_neg_batch = feature_neg.to(device)
            targets_batch = targets.to(device)
            feature_batch = feature.to(device)
            targets_batch = torch.as_tensor(targets_batch, dtype=torch.long).to(device)
            risks_batch = risks.to(device)
            risks_batch = torch.as_tensor(risks_batch,dtype=torch.long).to(device)
            preds, cosine_similarity,out_pos_feature, out_neg_feature = model(feature_batch,feature_neg_batch, fc_matrixs_batch,fc_matrixs_neg_batch)
            pos_batch = (targets_batch == 1).nonzero(as_tuple=True)
            neg_batch = (targets_batch == 0).nonzero(as_tuple=True)
            
            distance = 1-cosine_similarity
            # print(distance)
            pos_loss,neg_loss = 0,0
            if len(pos_batch[0]) != 0:
                # print(distance)
                # pos_loss += torch.max(torch.tensor(0).float().to('cuda'),-distance[i]+0.5)
                pos_loss += -distance[0]+2
                Pos_Losses.append(pos_loss.detach().cpu().item())
                # print(pos_loss)
            if len(neg_batch[0]) != 0:
                # print(distance)
                neg_loss += distance[0]
                Neg_Losses.append(neg_loss.detach().cpu().item())
                # print(neg_loss)
            
            loss = nn.CrossEntropyLoss(reduction='none')(preds, targets_batch) 
            # print(loss)
            # loss = loss * risks_batch
            # print(loss)
            loss = loss.mean()
            # print(loss)
            # loss +=  0.25 * link_loss
            
            CE_losses.append(loss.detach().cpu().item())
            if cosine_loss:
                loss += pos_loss + neg_loss
                # loss += neg_loss
            # print(loss)
            
            
            # print('PosLoss:',pos_loss)
            # print('NegLoss:',neg_loss)
            loss_np = loss.detach().cpu().item()
            val_losses.append(loss_np)

            preds_score = F.softmax(preds, dim=1).cpu().numpy()
            preds = np.argmax(preds_score, axis=1)
            risks = risks.cpu().numpy()
            targets = targets.cpu().numpy()
            
            pred_list.append(preds)
            target_list.append(targets)
            risk_list.append(risks)
            preds_score_list.append(preds_score[:, 1])#
            
    risks= np.concatenate(risk_list).squeeze()
    preds = np.concatenate(pred_list)
    preds_score = np.concatenate(preds_score_list)
    targets = np.concatenate(target_list)
    preds = preds.squeeze()
    targets = targets.squeeze()

    
    # print(labels_draw.shape)
    risk = 0
    for i,num in enumerate(targets):
        if targets[i] == 1:
            if risks[i]==2 and preds[i]==0:
                risk+=1
    # print(preds_score)
    # print(targets)
    score_acc = accuracy_score(targets, preds)
    score_auc = roc_auc_score(targets, preds_score)
    score_pre, score_rec, score_f1,_ = precision_recall_fscore_support(targets,preds,average="binary")
    # print('score_pre:',score_pre)
        
    return np.asarray(val_losses).mean(), score_acc, score_auc, preds, preds_score, targets, risk,np.asarray(CE_losses).mean(),np.asarray(Pos_Losses).mean(),np.asarray(Neg_Losses).mean(),score_pre,score_rec,score_f1
