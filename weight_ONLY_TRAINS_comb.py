import argparse
import awkward
import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import uproot
import uproot3
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import yaml
import scipy.sparse as ss
from datetime import datetime, timedelta
from torch_geometric.utils import degree
from torch_geometric.data import DataListLoader, DataLoader

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils_newdata import *

import gc
print("Libraries loaded!")


def main():

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)

    path_to_file = config['data']['path_to_trainfiles']

    print("dataset used:", path_to_file)

    save_trained_model = True

    
    #output_path_graphs = "data/graphs_NewDataset_"

    dataset = torch.load( path_to_file)


    ## define architecture
    batch_size = config['architecture']['batch_size']
    test_size = config['architecture']['test_size']

    dataset= shuffle(dataset,random_state=42)
    train_ds, validation_ds = train_test_split(dataset, test_size = test_size, random_state = 144)
    #train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=3)
    #val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, num_workers=3)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False)


    print ("train dataset size:", len(train_ds))
    print ("validation dataset size:", len(validation_ds))

    deg = torch.zeros(100, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())


    n_epochs = config['architecture']['n_epochs']
    learning_rate = config['architecture']['learning_rate']
    choose_model = config['architecture']['choose_model']
    save_every_epoch = config['architecture']['save_every_epoch']

    if choose_model == "LundNet":
        model = LundNet()
    if choose_model == "GATNet":
        model = GATNet()
    if choose_model == "GINNet":
        model = GINNet()
    if choose_model == "EdgeGinNet":
        model = EdgeGinNet()
    if choose_model == "PNANet":
        model = PNANet()

    flag = config['retrain']['flag']
    path_to_ckpt = config['retrain']['path_to_ckpt']

    adv_model_name = "adversarial_MODEL_"
    
    #lambda_parameter = config["combined"]["lambda_parameter"]
    #num_gaussians = config["architecture"]["num_gaussians"]
    #loss_parameter = config["architecture"]["loss_parameter"]
    
    lambda_parameter = 25
    num_gaussians = 20
    loss_parameter = 1
    loss_weights = [1,1]
    adv = Adversary_new(lambda_parameter, num_gaussians)

    if flag==True:
        path = path_to_ckpt
        model.load_state_dict(torch.load(path))

    #device = torch.device('cpu')
    device = torch.device('cuda') # Usually gpu 4 worked best, it had the most memory available
    
    #model = torch.nn.DataParallel(model)
    model.to(device)
    adv.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_small = torch.optim.Adam(model.parameters(), lr=0.4*learning_rate)
    optimizer_adv = torch.optim.Adam(adv.parameters(), lr=5*learning_rate)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=4*learning_rate)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=10*learning_rate)

    train_jds = []
    val_jds = []

    train_bgrej = []
    val_bgrej = []

    model_name = config['data']['model_name']
    path_to_save = config['data']['path_to_save']
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    metrics_filename = path_to_save+"losses_"+model_name+datetime.now().strftime("%d%m-%H%M")+".txt"

    for epoch in range(n_epochs):
        train_loss.append(train_clas(train_loader, model, device, optimizer, optimizer2, optimizer3, epoch))
        val_loss.append(my_test(val_loader, model, device))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, train_loss[epoch], val_loss[epoch]))
        if (save_every_epoch):
            torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")
        elif epoch == n_epochs-1:
            torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss[epoch])+".pt")

    metrics = pd.DataFrame({"Train_Loss":train_loss,"Val_Loss":val_loss})
    metrics.to_csv(metrics_filename, index = False)
    #return

    do_combined_training = True
    n_epochs_adv = 20
    n_epochs_common = 200
    train_loss_clsf = []
    train_loss_adv = []
    train_loss_total = []
    val_loss_clsf = []
    val_loss_adv = []
    val_loss_total = []
    train_acc = []
    val_acc = []

    if do_combined_training:
        MASSBINS = np.linspace(40, 300, (300 - 40) // 5 + 1, endpoint=True)
        ############ -------------- adversarial pre-trained ------------- ###############
        for epoch in range(n_epochs_adv):
            ad_lt, clsf_lt, total_lt =  train_adversary_2(train_loader, model, adv, optimizer_adv, device, loss_parameter ,loss_weights) 
            train_loss_adv.append(ad_lt)
            train_loss_clsf.append(clsf_lt)
            train_loss_total.append(total_lt)
            train_acc.append(get_accuracy(train_loader, model, device))
    
            ad_lv, clsf_lv, total_lv =  test_combined(val_loader, model, adv, device, loss_parameter , loss_weights) 
            val_loss_adv.append(ad_lv)
            val_loss_clsf.append(clsf_lv)
            val_loss_total.append(total_lv)
            val_acc.append(get_accuracy(val_loader, model, device))
    
            print('Epoch: {:03d}, Train Loss total: {:.5f}, Train Loss adv: {:.5f}, Train Loss clsf: {:.5f}, val_loss_adv: {:.5f}, val_loss_clsf: {:.5f}, val_loss_total: {:.5f},train_acc: {:.5f},val_acc: {:.5f}'.format(epoch, train_loss_total[epoch],train_loss_adv[epoch],train_loss_clsf[epoch], val_loss_adv[epoch], val_loss_clsf[epoch], val_loss_total[epoch],train_acc[epoch],val_acc[epoch]))
            metrics = pd.DataFrame({"Train_Loss_adv":train_loss_adv,"Train_Loss_clsf":train_loss_clsf,"Train_Loss_total":train_loss_total,"Val_Loss_Adv":val_loss_adv,"Val_loss_Class":val_loss_clsf,"val_loss_total":val_loss_total, "Train_Acc":train_acc,"Val_Acc":val_acc})
            metrics.to_csv(metrics_filename, index = False)
            if (save_every_epoch):
                torch.save(adv.state_dict(), path_to_save+model_name+adv_model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_adv[epoch])+".pt")
            elif epoch == n_epochs-1:
                torch.save(adv.state_dict(), path_to_save+model_name+adv_model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_adv[epoch])+".pt")

        train_loss_clsf = []
        train_loss_adv = []
        train_loss_total = []
        val_loss_clsf = []
        val_loss_adv = []
        val_loss_total = []
        train_acc = []
        val_acc = []
        train_jsdbg = []
        val_jsdbg = []
        for epoch in range(n_epochs_common):
            print("Epoch:{}".format(epoch))
            if epoch < 12:
                ad_lt, clsf_lt, total_lt =  train_combined_2(train_loader, model, adv, optimizer_small, optimizer_adv, device, loss_parameter,loss_weights)
            else:
                ad_lt, clsf_lt, total_lt =  train_combined_2(train_loader, model, adv, optimizer, optimizer_adv, device, loss_parameter,loss_weights)
            
            train_loss_clsf.append(clsf_lt)
            train_loss_adv.append(ad_lt)
            train_loss_total.append(total_lt)
            epsilon_bg, jds = aux_metrics(train_loader, model, adv, device, MASSBINS)
            #epsilon_bg, jds = 0,0
            train_jds.append(jds)
            train_bgrej.append(epsilon_bg)
            if jds:
                train_jsdbg.append(epsilon_bg - 1/jds)
            else:
                train_jsdbg.append(0)

            ad_lv, clsf_lv, total_lv =  test_combined(val_loader, model, adv, device, loss_parameter, loss_weights)
            val_loss_adv.append(ad_lv)
            val_loss_clsf.append(clsf_lv)
            val_loss_total.append(total_lv)
    
            epsilon_bg_test, jds_test = aux_metrics(val_loader, model, adv, device, MASSBINS)
            #epsilon_bg_test, jds_test = 0,0
            val_jds.append(jds_test)
            val_bgrej.append(epsilon_bg_test)
            if jds_test:
                val_jsdbg.append(epsilon_bg_test - 1/jds_test)
            else:
                val_jsdbg.append(0)
    
            print('Epoch: {:03d}, Train Loss total: {:.5f}, Train Loss adv: {:.5f}, Train Loss clsf: {:.5f}, val_loss_adv: {:.5f}, val_loss_clsf: {:.5f}, val_loss_total: {:.5f},train_jds: {:.5f},val_jds: {:.5f},train_jdsbg: {:.5f},val_jdsbg: {:.5f}'.format(epoch,
                train_loss_total[epoch],train_loss_adv[epoch],train_loss_clsf[epoch], val_loss_adv[epoch], val_loss_clsf[epoch], val_loss_total[epoch], train_jds[epoch], val_jds[epoch],train_jsdbg[epoch],val_jsdbg[epoch]))
            metrics = pd.DataFrame({"Train_Loss_adv":train_loss_adv,"Train_Loss_clsf":train_loss_clsf,"Train_Loss_total":train_loss_total,"Val_Loss_Adv":val_loss_adv,"Val_loss_Class":val_loss_clsf,
                "val_loss_total":val_loss_total, "Train_jds":train_jds,"Val_jds":val_jds,"Train_bgrej":train_bgrej,"Val_bgrej":val_bgrej, "Train_jsdbg":train_jsdbg,"Val_jsdbg":val_jsdbg})
            metrics.to_csv(metrics_filename, index = False)
            if (save_every_epoch):
                torch.save(model.state_dict(), path_to_save+model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_clsf[epoch])+"_comb_"+".pt")
                torch.save(adv.state_dict(), path_to_save+model_name+adv_model_name+"e{:03d}".format(epoch+1)+"_{:.5f}".format(val_loss_adv[epoch])+"_comb_"+".pt")
    
    return



if __name__ == "__main__":
    main()

