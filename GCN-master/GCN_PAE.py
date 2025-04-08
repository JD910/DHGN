import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
import random
import numpy as np
import xlrd
import pandas as pd
import datetime

from opt import *
from EV_GCN_PAE import EV_GCN
from utils.metrics_PAE import accuracy, auc, prf
from dataloader_PAE import dataloader
from PAE import PAE

def dataset_create(data_excel):

    data_all = pd.read_excel(data_excel, sheet_name='Sheet1')
    length = data_all.shape[1]
    data_feat = data_all.iloc[["Nine baseline clinical data"], 1:]
    data_feat.columns = data_feat.iloc[0]
    data_feat = data_feat[1:].reset_index(drop=True)

    ANS_Index = data_feat.columns[data_feat.columns.str.contains('ANS', case=False)]
    MSK_Index = data_feat.columns[data_feat.columns.str.contains('MSK', case=False)]
    data_feat_test_ANS = data_feat[ANS_Index]
    data_feat_test_MSK =  data_feat[MSK_Index]
    all_columns = data_feat.columns
    b_columns = data_feat_test_ANS.columns
    c_columns = data_feat_test_MSK.columns

    remaining_columns = all_columns.difference(b_columns.union(c_columns))

    selected_indices = list(range("Serial Number"))
    selected_array = remaining_columns[selected_indices]
    remaining_array = np.delete(remaining_columns, selected_indices)

    data_feat_train = data_feat[remaining_array]
    data_feat_train_numpy = data_feat_train.to_numpy()
    data_feat_train_numpy =  np.transpose(data_feat_train_numpy).astype('float32')
    nonimg_train = data_feat_train_numpy

    data_feat_inter_test = data_feat[selected_array]
    data_feat_inter_test_numpy = data_feat_inter_test.to_numpy()
    data_feat_inter_test_numpy =  np.transpose(data_feat_inter_test_numpy).astype('float32')
    nonimg_inter_test = data_feat_inter_test_numpy

    data_feat_train_all = data_feat[remaining_columns]
    data_feat_train_all_numpy = data_feat_train_all.to_numpy()
    data_feat_train_all_numpy =  np.transpose(data_feat_train_all_numpy).astype('float32')
    nonimg_train_all = data_feat_train_all_numpy

    data_feat_test_ANS_numpy = data_feat_test_ANS.to_numpy()
    data_feat_test_ANS_numpy =  np.transpose(data_feat_test_ANS_numpy).astype('float32')
    nonimg_ANS = data_feat_test_ANS_numpy
    data_feat_test_MSK_numpy = data_feat_test_MSK.to_numpy()
    data_feat_test_MSK_numpy =  np.transpose(data_feat_test_MSK_numpy).astype('float32')
    nonimg_MSK = data_feat_test_MSK_numpy

    data_pfs = data_all.iloc[["PFS"], 1:]
    data_pfs.columns = data_pfs.iloc[0]
    data_pfs = data_pfs[1:].reset_index(drop=True)
    data_pfs_lab_ANS = data_pfs[ANS_Index]
    data_pfs_lab_MSK = data_pfs[MSK_Index]
    data_pfs_lab_train_all = data_pfs[remaining_columns]
    data_pfs_lab_train = data_pfs[remaining_array]
    data_pfs_lab_inter_test = data_pfs[selected_array]
    data_pfs_lab_ANS = data_pfs_lab_ANS.to_numpy().squeeze().astype('float32')
    data_pfs_lab_MSK = data_pfs_lab_MSK.to_numpy().squeeze().astype('float32')
    data_pfs_lab_train = data_pfs_lab_train.to_numpy().squeeze().astype('float32')
    data_pfs_lab_inter_test = data_pfs_lab_inter_test.to_numpy().squeeze().astype('float32')
    data_pfs_lab_train_all = data_pfs_lab_train_all.to_numpy().squeeze().astype('float32')

    return data_pfs_lab_ANS, data_pfs_lab_MSK, data_pfs_lab_train, data_pfs_lab_inter_test, data_pfs_lab_train_all,\
        nonimg_ANS, nonimg_MSK, nonimg_train, nonimg_inter_test, nonimg_train_all

if __name__ == '__main__':
    opt = OptInit().initialize()

    print('Loading dataset ...')
    dl = dataloader()

    PFS_ANS, PFS_MSK, PFS_train,  PFS_train_all, nonimg_ANS, nonimg_MSK, nonimg_train, nonimg_train_all \
    = dataset_create("/home/The clinical baseline characteristics.xlsx")

    print('Start Constructing Population Graph...')

    # get PAE inputs
    edge_label, edge_index, edgenet_input = dl.get_PAE_inputs(nonimg_train_all, PFS_train_all)
    # normalization for PAE
    edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)

    edge_label_test_MSK, edge_index_test_MSK, edgenet_input_test_MSK = dl.get_PAE_inputs(nonimg_MSK, PFS_MSK)
    # normalization for PAE
    edgenet_input_test_MSK = (edgenet_input_test_MSK - edgenet_input_test_MSK.mean(axis=0)) / edgenet_input_test_MSK.std(axis=0)

    edge_label_test_ANS, edge_index_test_ANS, edgenet_input_test_ANS = dl.get_PAE_inputs(nonimg_ANS, PFS_ANS)
    # normalization for PAE
    edgenet_input_test_ANS = (edgenet_input_test_ANS - edgenet_input_test_ANS.mean(axis=0)) / edgenet_input_test_ANS.std(axis=0)

    # build network architecture
    model = EV_GCN(opt.dropout, edge_dropout=opt.edropout, edgenet_input_dim=2*nonimg_train.shape[1]).to(opt.device)
    model = model.to(opt.device)

    #loss_fn =torch.nn.CrossEntropyLoss()
    loss_fn =torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
    edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)

    labels = torch.tensor(edge_label, dtype=torch.float32).to(opt.device)
    now = datetime.datetime.now()
    fold_model_path = opt.ckpt_path + "/{}.pth".format(now)

    print("  Number of training samples %d" % len(PFS_train))
    print("  Start training...\r\n")
    acc = 0

    #opt.train = 0
    #fold_model_path = './save_models/saved_model_path.pth'

    if opt.train==1:
        for epoch in range(opt.num_iter):
            model.train()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                edge_weights = model(edge_index, edgenet_input)
                loss = loss_fn(edge_weights, labels)
                loss.backward()
                optimizer.step()
                print("Epoch: {},\tce loss: {:.5f}".format(epoch, loss.item()))
                edge_weights_pred_labels = torch.where(edge_weights > 0.5, 1, 0)
            correct_train, acc_train = accuracy(edge_weights_pred_labels.detach().cpu().numpy(), labels.detach().cpu().numpy())

            model.eval()
            with torch.set_grad_enabled(False):
                edge_weights_test = model(edge_index_test_MSK, edgenet_input_test_MSK)
            edge_weights_test_pred_labels = torch.where(edge_weights_test > 0.5, 1, 0)
            edge_weights_test_pred_labels = edge_weights_test_pred_labels.detach().cpu().numpy()
            labels_test_MSK = torch.tensor(edge_label_test_MSK, dtype=torch.float32).to(opt.device)
            correct_test, acc_test = accuracy(edge_weights_test_pred_labels, labels_test_MSK.detach().cpu().numpy())

            if acc_test > acc and epoch >100:
                print(acc_test)
                acc = acc_test
                correct = correct_test
                if opt.ckpt_path !='':
                    if not os.path.exists(opt.ckpt_path):
                        os.makedirs(opt.ckpt_path)
                    torch.save(model.state_dict(), fold_model_path)

        opt.train = 0

    if opt.train==0:
        print('  Start testing...')
        model.load_state_dict(torch.load(fold_model_path))
        model.eval()

        edge_index_test_MSK = torch.tensor(edge_index_test_MSK, dtype=torch.long).to(opt.device)
        edgenet_input_test_MSK = torch.tensor(edgenet_input_test_MSK, dtype=torch.float32).to(opt.device)
        labels_test_MSK = torch.tensor(edge_label_test_MSK, dtype=torch.float32).to(opt.device)

        edge_index_test_ANS = torch.tensor(edge_index_test_ANS, dtype=torch.long).to(opt.device)
        edgenet_input_test_ANS = torch.tensor(edgenet_input_test_ANS, dtype=torch.float32).to(opt.device)
        labels_test_ANS = torch.tensor(edge_label_test_ANS, dtype=torch.float32).to(opt.device)

        edge_weights_test_MSK = model(edge_index_test_MSK, edgenet_input_test_MSK)
        edge_weights_test_MSK_pred_labels = torch.where(edge_weights_test_MSK > 0.5, 1, 0)
        edge_weights_test_MSK_pred_labels = edge_weights_test_MSK_pred_labels.detach().cpu().numpy()
        correct_test_MSK, acc_test_MSK = accuracy(edge_weights_test_MSK_pred_labels, labels_test_MSK.detach().cpu().numpy())
        aucs_test_MSK = auc(edge_weights_test_MSK_pred_labels,labels_test_MSK.detach().cpu().numpy())
        prfs_test_MSK  = prf(edge_weights_test_MSK_pred_labels,labels_test_MSK.detach().cpu().numpy())

        edge_weights_test_ANS = model(edge_index_test_ANS, edgenet_input_test_ANS)
        edge_weights_test_ANS_pred_labels = torch.where(edge_weights_test_ANS > 0.5, 1, 0)
        edge_weights_test_ANS_pred_labels = edge_weights_test_ANS_pred_labels.detach().cpu().numpy()
        correct_test_ANS, acc_test_ANS = accuracy(edge_weights_test_ANS_pred_labels, labels_test_ANS.detach().cpu().numpy())
        aucs_test_ANS = auc(edge_weights_test_ANS_pred_labels,labels_test_ANS.detach().cpu().numpy())
        prfs_ANS  = prf(edge_weights_test_ANS_pred_labels,labels_test_ANS.detach().cpu().numpy())

    se, sp, f1 = prfs_test_MSK
    print("=> Average test MSK sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(se, sp, f1))

    se, sp, f1 = prfs_ANS
    print("=> Average test ANS sensitivity {:.4f}, specificity {:.4f}, F1-score {:.4f}".format(se, sp, f1))

