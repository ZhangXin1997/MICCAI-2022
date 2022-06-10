import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from sklearn.model_selection import StratifiedKFold

class GDataset(Dataset):
    def __init__(self, idxs, csv,  node_num):
        super(GDataset, self).__init__()
        
        self.fc_matrix_dot = './data/top-K'
        
        self.fc_matrix_dot_neg = './data/bottom-K'
        self.csv = csv
        self.idxs = idxs
        # self.class_num = class_num
        # self.feature_mean = feature_mean
        # self.feature_std = feature_std
        self.node_num = node_num

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, item):
        index = self.idxs[item]
        dir_name = self.csv['name'].iloc[index]
        label = self.csv['label'].iloc[index].astype(np.int)
        risk = self.csv['risk'].iloc[index].astype(np.int)
        fc_matrix_path = os.path.join(self.fc_matrix_dot, dir_name, f'fcmatrix_{self.node_num}.npy')
        fc_matrix_neg_path = os.path.join(self.fc_matrix_dot_neg, dir_name, f'fcmatrix_{self.node_num}.npy')
        fc_matrix = np.load(fc_matrix_path).astype(np.float32)
        #fc_matrix = 1 - np.sqrt((1 - fc_matrix) / 2)
        fc_matrix_neg = np.load(fc_matrix_neg_path).astype(np.float32)
        feature_path = os.path.join(self.fc_matrix_dot, dir_name, f'feature_{self.node_num}.npy')
        feature_path_neg = os.path.join(self.fc_matrix_dot_neg, dir_name, f'feature_{self.node_num}.npy')
        feature = np.load(feature_path).astype(np.float32)
        feature_neg = np.load(feature_path_neg).astype(np.float32)
        #feature = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)[:, :3]
        #feature = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)
        # feature_path = os.path.join(self.fc_matrix_dot, dir_name, 'ROISignals_feature.npy')
        # feature = np.load(feature_path).astype(np.float32)
        # feature_mean = np.array([[0.00304313, 2.71610968, 0.01176067]])
        # feature_std = np.array([[0.02716032, 3.4016537, 0.00158216405]])
        # feature = ((feature - feature_mean) / feature_std).astype(np.float32)

        # features = np.ones([116, 1], dtype=np.float32)
        return torch.tensor(fc_matrix), torch.tensor(fc_matrix_neg),torch.tensor(feature), torch.tensor(feature_neg),torch.tensor(label), torch.tensor(risk)

class GTestDataset(Dataset):
    def __init__(self, idxs, csv, node_num):
        super(GTestDataset, self).__init__()
        # self.fc_matrix_dot = os.path.join('adni','input_0974')
        self.fc_matrix_dot = './data/top-K'
        
        self.fc_matrix_dot_neg = './data/bottom-K'
        # self.fc_matrix_dot = os.path.join('test_input_200_20_095')
        self.csv = csv
        self.idxs = idxs
        self.node_num = node_num

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, item):
        index = self.idxs[item]
        dir_name = self.csv['name'].iloc[index]
        label = self.csv['label'].iloc[index].astype(np.int)
        risk = self.csv['risk'].iloc[index].astype(np.int)
        fc_matrix_path = os.path.join(self.fc_matrix_dot, dir_name, f'fcmatrix_{self.node_num}.npy')
        fc_matrix_neg_path = os.path.join(self.fc_matrix_dot_neg, dir_name, f'fcmatrix_{self.node_num}.npy')
        fc_matrix = np.load(fc_matrix_path).astype(np.float32)
        # fc_matrix = 1 - np.sqrt((1 - fc_matrix) / 2)
        fc_matrix_neg = np.load(fc_matrix_neg_path).astype(np.float32)
        feature_path = os.path.join(self.fc_matrix_dot, dir_name, f'feature_{self.node_num}.npy')
        feature_path_neg = os.path.join(self.fc_matrix_dot_neg, dir_name, f'feature_{self.node_num}.npy')
        feature = np.load(feature_path).astype(np.float32)
        #feature = ((feature - self.feature_mean) / self.feature_std).astype(np.float32)
        feature_neg = np.load(feature_path_neg).astype(np.float32)
        # feature_path = os.path.join(self.fc_matrix_dot, dir_name, 'ROISignals_feature.npy')
        # feature = np.load(feature_path).astype(np.float32)
        # feature_mean = np.array([[0.00304313, 2.71610968, 0.01176067]])
        # feature_std = np.array([[0.02716032, 3.4016537, 0.00158216405]])
        # feature = ((feature - feature_mean) / feature_std).astype(np.float32)

        # features = np.ones([116, 1], dtype=np.float32)
        return torch.tensor(fc_matrix), torch.tensor(fc_matrix_neg),torch.tensor(feature), torch.tensor(feature_neg),torch.tensor(label), torch.tensor(risk),dir_name
def get_data_loader(i_fold,node_num,bs):
    df = pd.read_csv('data_csv/fold_train.csv')
    df = df.reset_index()
    #print(df.shape)
    if i_fold != 5:
        train_idxs = np.where((df['fold'] != i_fold)&(df['fold'] != i_fold+1))[0]
    # train_idxs = np.where(df['fold'] != 6)[0]
        val_idxs = np.where(df['fold']  == i_fold+1)[0]
    else:
        train_idxs = np.where((df['fold'] != i_fold)&(df['fold'] != 1))[0]
        val_idxs = np.where(df['fold']  == 1)[0]
    test_idxs = np.where(df['fold'] == i_fold)[0]
   
    TrainDataset = GDataset(train_idxs, df, node_num)
    ValDataset = GDataset(val_idxs, df, node_num)
    TestDataset = GDataset(test_idxs,df,node_num)
    TrainLoader = DataLoader(TrainDataset, batch_size=bs, shuffle=True)
    ValLoader = DataLoader(ValDataset, batch_size=1)
    TestLoader = DataLoader(TestDataset,batch_size =1)
    return TrainLoader, ValLoader,TestLoader

def get_test_dataloader(node_num,batch_size,data_path):
    df = pd.read_csv(data_path)
    df = df.reset_index()
    #print(df.shape)

    test_idxs=np.where(df['name'] != None)[0]
    TestDataset = GTestDataset(test_idxs,df,node_num)
    TestLoader = DataLoader(TestDataset,batch_size=1)
    return TestLoader
if __name__ == '__main__':
    TrainLoader, ValLoader = get_data_loader(0, 'zero_vs_one_', 90)
    for matrix, feature, crs in TrainLoader:
        print(matrix.shape, feature.shape, crs.shape)
        break
    print(' ')
    for matrix, feature, crs in ValLoader:
        print(matrix.shape, feature.shape, crs)
        break
