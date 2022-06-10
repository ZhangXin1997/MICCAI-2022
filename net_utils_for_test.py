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

def test_epoch(cosine_loss,model, loader, device):
    model.eval()
    val_losses = []
    CE_losses,Pos_Losses,Neg_Losses = [],[],[]
    pred_list, target_list,risk_list,dir_name_list = [], [],[],[]
    
    preds_score_list = []
    with torch.no_grad():
        for fc_matrixs, fc_matrixs_neg,feature, feature_neg,targets,risks,dir_name in loader:
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
            dir_name_list.append(dir_name[0])
    risks= np.concatenate(risk_list).squeeze()
    preds = np.concatenate(pred_list)
    preds_score = np.concatenate(preds_score_list)
    targets = np.concatenate(target_list)
    preds = preds.squeeze()
    targets = targets.squeeze()

    
    # print(labels_draw.shape)
    # risk = 0
    # for i,num in enumerate(targets):
    #     if targets[i] == 1:
    #         if risks[i]==2 and preds[i]==0:
    #             risk+=1
    # print(preds_score)
    # print(targets)
    score_acc = accuracy_score(targets, preds)
    score_auc = roc_auc_score(targets, preds_score)
    score_pre, score_rec, score_f1,_ = precision_recall_fscore_support(targets,preds,average="binary")
    # print('score_pre:',score_pre)
        
    return np.asarray(val_losses).mean(), score_acc, score_auc, preds, preds_score, targets, risks,np.asarray(CE_losses).mean(),np.asarray(Pos_Losses).mean(),np.asarray(Neg_Losses).mean(),score_pre,score_rec,score_f1,dir_name_list

# def test_epoch(model, loader, device):
#     model.eval()
    
#     pred_list, target_list, risk_list,dir_name_list = [], [], [],[]
#     preds_score_list = []
#     with torch.no_grad():
#         for fc_matrixs, feature, targets, risks,dir_name in loader:
#             # print(dir_name)
#             # batchsize, node_num = fc_matrixs.shape
#             # print(dir_name[0],end=' ')
#             fc_matrixs_batch = fc_matrixs.to(device)
#             targets_batch = targets.to(device)
#             feature_batch = feature.to(device)
#             #risk_batch=risks.to(device)
#             targets_batch = torch.as_tensor(targets_batch, dtype=torch.long).to(device)

#             preds, link_loss, attentions = model(feature_batch, fc_matrixs_batch)
#             #print(preds)
#             #loss = nn.CrossEntropyLoss()(preds, targets_batch) + 0.5 * link_loss

#             #loss_np = loss.detach().cpu().item()
#             preds_score = F.softmax(preds, dim=1).cpu().numpy()
#             preds = np.argmax(preds_score, axis=1)
#             # if preds == 1:
#             #     print('阳性')
#             # else:
#             #     print('阴性')
#             risks = risks.cpu().numpy()
#             targets = targets.cpu().numpy()
#             #val_losses.append(loss_np)
#             pred_list.append(preds)
#             target_list.append(targets)
#             preds_score_list.append(preds_score[:, 1])
#             risk_list.append(risks)
#             dir_name_list.append(dir_name[0])
#     preds = np.concatenate(pred_list)
#     preds_score = np.concatenate(preds_score_list)
#     targets = np.concatenate(target_list)
#     risks=np.concatenate(risk_list)
#     #print(preds.shape)
#     #print(targets.shape)
#     score_acc = accuracy_score(targets, preds)
#     score_auc = roc_auc_score(targets, preds_score)
#     return score_acc, score_auc, preds, preds_score, targets, risks,attentions,dir_name_list
