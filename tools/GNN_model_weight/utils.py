import awkward
import os.path as osp
import os
import glob
import torch
import awkward as ak
import time
import yaml
import uproot
import uproot3
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
#from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataListLoader, DataLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel, EdgeConv, GATConv, GINConv, PNAConv
from torch_geometric.data import Data
import scipy.sparse as ss
from datetime import datetime, timedelta
from torch_geometric.utils import degree
from scipy.stats import entropy
import math
import networkx as nx
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
from ..GNN_model_weight.models import mdn_loss, mdn_loss_new

## for top tagging:
# weights_file = uproot.open("/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/flat_weights.root")
weights_file = uproot.open("/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/5_Signal_and_BKG_cutTruth350/flat_weights.root")

flatweights_bg = weights_file["bg_inv"].to_numpy()
flatweights_sig = weights_file["h_sig_inv"].to_numpy()

def GetPtWeight( dsid , pt, SF):

    lenght_sig = len(flatweights_bg[0])
    lenght_bkg = len(flatweights_bg[0])
    scale_factor = 1
    weight_out = []

    for i in range ( 0,len(dsid) ):
        pt_bin = int( ((pt[i]-200)/3000)*lenght_sig )
        if pt_bin==lenght_sig :
            pt_bin = lenght_sig-1
        if dsid[i] < 370000 :
            #weight_out.append( (flatweights_bg[0][pt_bin]*scale_factor)*10**4 ) ## used for W tagging
            weight_out.append( (flatweights_bg[0][pt_bin])*1 )
        if dsid[i] > 370000 :
            #weight_out.append( (flatweights_sig[0][pt_bin])*10**4 ) ## used for W tagging
            weight_out.append( (flatweights_sig[0][pt_bin])*1 )
    return np.array(weight_out)

def GetPtWeight_2( dsid , pt, SF):

    ## PT histograms of all qcd and top jets in dataset
    filename1 = "/data/bcifuentes/histograms/qcd.root"
    filename2 = "/data/bcifuentes/histograms/top.root"
    weights_file1 = uproot.open(filename1)
    flatweights_bg = weights_file1["pt"].to_numpy()
    weights_file2 = uproot.open(filename2)
    flatweights_sig = weights_file2["pt"].to_numpy()
    
    lenght_sig = len(flatweights_sig[0])
    lenght_bkg = len(flatweights_bg[0])
    #print("lenght_sig:", lenght_sig, "  lenght_bkg:", lenght_bkg)
    
    sig_bkg_proportion = 10  ## if is taked 5% of signal and 1% of qcd for training then sig_bkg_proportion=5
    scale_factor = (lenght_bkg/lenght_sig) / 5 #1
    weight_out = []
    Inv_hist_bg = []#flatweights_bg[0]
    Inv_hist_sig = []#flatweights_sig[0]
    
    ## it's time to calcuate the 1/hist
    for i in range (0,lenght_bkg):
    	if flatweights_bg[0][i]==0:
    		Inv_hist_bg.append(0)
    		continue
    	else:
    		Inv_hist_bg.append(np.sum(flatweights_bg[0]) / (lenght_bkg * flatweights_bg[0][i]))
    		
    for i in range (0,lenght_sig):
    	if flatweights_sig[0][i]==0:
    		Inv_hist_sig.append(0)
    		continue
    	else:
    		Inv_hist_sig.append(np.sum(flatweights_sig[0]) / (lenght_sig * flatweights_sig[0][i]))
        
    for i in range ( 0,len(dsid) ):
        pt_bin = int( ((pt[i]-100)/3000)*lenght_sig )
        if pt_bin>=lenght_sig : # ==
            pt_bin = lenght_sig-1
        if dsid[i] ==10:#< 370000 :
            #print("pt[i] ->", pt[i])
            #print("bin_pt->", pt_bin)
            weight_out.append( (Inv_hist_bg[pt_bin])*1  )
        if dsid[i] !=10: ##events with other values than 1 and 10 must be removed in data creation
            weight_out.append( (Inv_hist_sig[pt_bin]*scale_factor)*1 ) #*10**2 )
    return np.array(weight_out)



def load_yaml(file_name):
    assert(os.path.exists(file_name))
    with open(file_name) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def create_train_dataset_fulld_new_Ntrk_pt_weight_file(graphs, z, k, d, edge1, edge2, weight, label, Ntracks, jet_pts, jet_ms):

    for i in range(len(z)):
        if len(z[i])<5: 
            continue
        #print(label[i])
        if (label[i]!=1) and (label[i]!=10) :
            continue
        label_out = label[i]
        if label[i]== 10:
            label_out = 0

        if jet_pts[i] > 3200: continue
        if jet_pts[i] < 350: continue 
        
        z_out = ak.to_numpy(z[i])
        k_out = ak.to_numpy(k[i])
        d_out = ak.to_numpy(d[i])

        z_out += 1e-4 
        k_out += 1e-4 
        d_out += 1e-4 
        z_out = np.log(1/z_out)
        k_out = np.log(k_out)
        d_out = np.log(1/d_out)
        # 1e-4 # Rafael 15% data
        
        mean_z, std_z = 2.0778886185997316, 1.4714192378460587
        mean_dr, std_dr = 3.8924797952050296, 2.279933125432541
        mean_kt, std_kt = -2.3919176643005926, 2.97112274201009
        mean_ntrks, std_ntrks = 39.81133623360089, 10.99193693271175
        
        z_out = (z_out - mean_z) / std_z
        k_out = (k_out - mean_kt) / std_kt
        d_out = (d_out - mean_dr) / std_dr
        #Ntracks = (Ntracks - mean_ntrks) / std_ntrks
        Ntrk = (Ntracks[i] - mean_ntrks) / std_ntrks
        
        if (len(edge1[i])== 0) or (len(edge2[i])== 0) or (len(k[i])== 0) or (len(z[i])== 0) or (len(d[i])== 0):
            continue
        else:
            edge = torch.tensor(np.array([edge1[i], edge2[i]]) , dtype=torch.long)

        
        vec = []
        vec.append(np.array([d_out, z_out, k_out]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float).detach(),
                           edge_index = torch.tensor(edge, dtype=torch.int64).detach(),
                           #Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
                           Ntrk=torch.tensor(Ntrk, dtype=torch.float).detach(),
                           weights =torch.tensor(weight[i], dtype=torch.float).detach(),
                           pt=torch.tensor(jet_pts[i], dtype=torch.float).detach(),
                           mass=torch.tensor(jet_ms[i], dtype=torch.float).detach(),
                           y=torch.tensor(label_out, dtype=torch.float).detach() ))
    return graphs


'''
def create_train_dataset_fulld_new_Ntrk_pt_weight_file_test(graphs, z, k, d, edge1, edge2, weight, label, Ntracks, jet_pts, jet_ms):
    print(weight)
    print("mlp Kelly")
    data_under3 = 0
    for i in range(len(z)):
        if len(z[i])<5: 
            continue

        if (label[i]!=1) and (label[i]!=10) :
            continue
        label_out = label[i]
        if label[i]== 10:
            label_out = 0

        if jet_pts[i] > 3200: continue
        if jet_pts[i] < 350: continue 

        z_out = ak.to_numpy(z[i])
        k_out = ak.to_numpy(k[i])
        d_out = ak.to_numpy(d[i])

        z_out += 1e-4 
        k_out += 1e-4 
        d_out += 1e-4 
        z_out = np.log(1/z_out)
        k_out = np.log(k_out)
        d_out = np.log(1/d_out)
        
        mean_z, std_z = 2.0778886185997316, 1.4714192378460587
        mean_dr, std_dr = 3.8924797952050296, 2.279933125432541
        mean_kt, std_kt = -2.3919176643005926, 2.97112274201009
        mean_ntrks, std_ntrks = 39.81133623360089, 10.99193693271175
        
        z_out = (z_out - mean_z) / std_z
        k_out = (k_out - mean_kt) / std_kt
        d_out = (d_out - mean_dr) / std_dr
        Ntrk = (Ntracks[i] - mean_ntrks) / std_ntrks
        
        if (len(edge1[i])== 0) or (len(edge2[i])== 0) or (len(k[i])== 0) or (len(z[i])== 0) or (len(d[i])== 0):
            continue
        else:
            edge = torch.tensor(np.array([edge1[i], edge2[i]]) , dtype=torch.long)

        
        vec = []
        vec.append(np.array([d_out, z_out, k_out]).T)
        vec = np.array(vec)
        vec = np.squeeze(vec)

        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float).detach(),
                           #edge_index = torch.tensor(edge).detach(),
                           edge_index = torch.tensor(edge, dtype=torch.int64).detach(),
                           #Ntrk=torch.tensor(Ntracks[i], dtype=torch.int).detach(),
                           Ntrk=torch.tensor(Ntrk, dtype=torch.float).detach(),
                           pt=torch.tensor(jet_pts[i], dtype=torch.float).detach(),
                           mass=torch.tensor(jet_ms[i], dtype=torch.float).detach(),
                           y=torch.tensor(label[i], dtype=torch.float).detach() ))
        
    return graphs

'''

def train(loader, model, device, optimizer):
    print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        batch_counter+=1

        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1)) ## add weights

        # loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        l2_lambda = 0.01 # regularization strength
        for param in model.parameters():
            if param.dim() > 1:
                # apply L2 regularization to all parameters except biases
                loss = loss + l2_lambda * nn.MSELoss()(param, torch.zeros_like(param))

        loss.backward()

        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(loader.dataset)


def train_clas(loader, model, device, optimizer1, optimizer2, optimizer3, epoch):
    print ("dataset size:",len(loader.dataset))
    model.train()
    loss_all = 0
    batch_counter = 0
    for data in loader:
        if len(data)<650:
            continue
        batch_counter+=1
        data = data.to(device)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1)) ## add weights

        loss = F.binary_cross_entropy(output, new_y, weight = new_w)
        loss.backward()
        loss_all += data.num_graphs * loss.item()

        if epoch < 10:
            optimizer3.step()
        elif epoch < 20:
            optimizer2.step()
        else:
            optimizer1.step()

    return loss_all / len(loader.dataset)




@torch.no_grad()
def get_accuracy(loader, model, device):
    #remember to change this when evaluating combined model
    model.eval()
    correct = 0
    for data in loader:
        cl_data = data.to(device)
        new_y = torch.reshape(cl_data.y, (int(list(cl_data.y.shape)[0]),1))
        pred = model(cl_data).max(dim=1)[1]
        correct += pred.eq(new_y[0,:]).sum().item()
    return correct / len(loader.dataset)

@torch.no_grad()
def my_test (loader, model, device):
    model.eval()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        new_y = torch.reshape(data.y, (int(list(data.y.shape)[0]),1))
        new_w = torch.reshape(data.weights, (int(list(data.weights.shape)[0]),1))
        loss = F.binary_cross_entropy(output, new_y, weight=new_w)
        loss_all += data.num_graphs * loss.item()
    return loss_all/len(loader.dataset)

@torch.no_grad()
def get_scores(loader, model, device):
    model.eval()
    total_output = np.array([[1]])
    batch_counter = 0
    for data in loader:
        batch_counter+=1
        # print ("Processing batch", batch_counter, "of",len(loader))
        data = data.to(device)
        pred = model(data)
        total_output = np.append(total_output, pred.cpu().detach().numpy(), axis=0)

    return total_output[1:]
