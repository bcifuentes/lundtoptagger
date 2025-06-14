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


def flatten_small_branch(ref, vec): ## ref and vec same len()
    vec_out = []
    for i in range (0, len(ref)):
        for j in range (0, len(ref[i])):
            vec_out.append(vec[i])
    return np.array(vec_out)

def main():
    
    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)
    config_signal = load_yaml("configs/config_signal.yaml") # TODO: make this an optional argument, but then the same file needs to be used in utils_newdata.py
    signal = config_signal["signal"]
    


    do_all_MC = config['data_test']['do_all_MC']
    path_MCs = []
    if do_all_MC == 0:
        path_MCs.append(config['data_test']['path_to_test_file_pythia'])
    else:
        path_MCs.append(config['data_test']['path_to_test_file_pythia'])
        path_MCs.append(config['data_test']['path_to_test_file_HerwigDipole'])
        path_MCs.append(config['data_test']['path_to_test_file_HerwigAngular'])
        path_MCs.append(config['data_test']['path_to_test_file_SherpaCluster'])
        path_MCs.append(config['data_test']['path_to_test_file_SherpaLund'])

    MC_count = 0
    for path_to_test_file in path_MCs:
        files = glob.glob(path_to_test_file)
        print ("path_to_test_file:",path_to_test_file)
        print ("files:",files)
        path_to_outdir = config['data_test']['path_to_outdir']
    
        path_to_combined_ckpt = config['test']['path_to_combined_ckpt']
        #path_to_combined_ckpt = path_minimun_val_loss
        print("ckpt used:", path_to_combined_ckpt )
    
        output_name = config['test']['output_name']
        if MC_count == 0:
            output_name += "Pythia"
        if MC_count == 1:
            output_name += "HerwigDipole"
        if MC_count == 2:
            output_name += "HerwigAngular"
        if MC_count == 3:
            output_name += "SherpaCluster"
        if MC_count == 4:
            output_name += "SherpaLund"
        
        MC_count += 1
        kT_selection = config['architecture']['kT_cut']
        
        files = glob.glob(path_to_test_file)
    
        #print ("files:",files)
        intreename = "AnalysisTree"
    
        nentries_total = 0
        nentries_done = 0
    
        learning_rate = 0.0005
        batch_size = 1600
        scale_factor = 1
        
        for file in files:
            nentries_total += uproot3.numentries(file, intreename)
        print("Evaluating on {} files with {} entries in total.".format(len(files), nentries_total))
        
        jet_type = "Akt10UFOJet" #UFO jets
        t_filestart = time.time()
        choose_model = config['test']['choose_model']
    
        count_files = -1
        graph_small_example = []
    
        check_file = 1
        file_count = 0
        for file in files:
            
            t_start = time.time()
            dataset = []
            print("Loading file",file)
            
            Good_jets = [] ## jets where at least two nodes fullfill kT condition 
            #1->good, 0->NoGood
            mcweights_out = []
            
            with uproot.open(file) as infile:
                tree = infile[intreename]
    
                count_files += 1
                dsids_test = tree["dsid"].array(library="np")
                if dsids_test[0] in  config_signal[signal]["skip_dsids"]: # don't lose time with jets that don't pass pt cut or wrong signal sample
                   continue
    
                jet_pts_truth = ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) )
                #ptweights = ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) )
                ptweights = np.ones_like( ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) ))
                #cut_pt = (jet_pts_truth > 350000) #& (ptweights < 1000000 )
                
                labels = ak.to_numpy(ak.flatten(tree["LRJ_truthLabel"].array(library="ak")) )
                #labels = labels == 1
                #labels = 1*labels
                
                dsids = dsids_test[0]*np.ones_like(ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) ))
                LRJ_pt_ref = tree["LRJ_pt"].array(library="np") 
                mcweights = tree["mcEventWeight"].array(library="np")  #mcEventWeight
                mcweights = flatten_small_branch(LRJ_pt_ref, mcweights)
                #print("len(mcweights)", len(mcweights))
                NBHadrons = ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) )
                
                parent1 =  ak.flatten(tree["jetLundIDParent1"].array(library="ak")) 
                parent2 = ak.flatten(tree["jetLundIDParent2"].array(library="ak")) 
                jet_pts = ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) )
                jet_etas = ak.to_numpy(ak.flatten(tree["LRJ_eta"].array(library="ak")) )
                jet_phis = ak.to_numpy(ak.flatten(tree["LRJ_phi"].array(library="ak")) )
                jet_ms =  ak.to_numpy(ak.flatten(tree["LRJ_mass"].array(library="ak")))
                all_lund_zs = ak.flatten(tree["jetLundZ"].array(library="ak")) 
                all_lund_kts = ak.flatten(tree["jetLundKt"].array(library="ak")) 
                all_lund_drs = ak.flatten(tree["jetLundDeltaR"].array(library="ak"))
                N_tracks = ak.to_numpy(ak.flatten(tree["LRJ_Nconst_Charged"].array(library="ak")) )
                #N_tracks = ak.to_numpy(ak.flatten(tree["LRJ_Nconst"].array(library="ak")) )
                
                flat_weights = GetPtWeight_all_MC( dsids, dsids_test,  jet_pts, 5, Pythia_or_All=True)
                
                #dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file( dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels ,N_tracks,jet_pts, jet_ms  )
                
    
                if count_files==0:
                    index_count_out = [2,1]
                    selected_nodes = [1,2]
                    edge_ID1_ex = np.concatenate((index_count_out, selected_nodes))
                    edge_ID2_ex = np.concatenate((selected_nodes, index_count_out))
                    edge = torch.tensor(np.array([edge_ID1_ex, edge_ID2_ex]) , dtype=torch.int64)
                    d_test = np.array([-0.7824,1.4473,1.2722])
                    z_test = np.array([-0.6482,-0.0148,1.8903])
                    kt_test = np.array([0.6482,-0.0848,1.1903])
                    vec = []
                    vec.append(np.array([d_test, z_test, kt_test]).T)
                    vec = np.array(vec)
                    vec = np.squeeze(vec)
                    vec=torch.tensor(vec, dtype=torch.float).detach()
                    
                    #'''
                    graph_small_example = Data(x= vec.detach() ,
                                   edge_index = edge.detach() ,
                                    Ntrk=torch.tensor(7, dtype=torch.float).detach(),
                                   weights= torch.tensor(1, dtype=torch.float).detach(),
                                   graph_size = torch.tensor(2, dtype=torch.float).detach(),
                                   mass= float(80) ,
                                   y= float(0) )
    
                dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file_test(
                    dataset, graph_small_example, all_lund_zs, all_lund_kts, all_lund_drs,
                    parent1, parent2, flat_weights, labels,
                    N_tracks, jet_pts, jet_ms, kT_selection,
                    mcweights, mcweights_out, Good_jets,
                    config_signal[signal]["signal_jet_truth_label"]
                )#, count_files)
    
                #Good_jets = labels
                labels = labels==config_signal[signal]["signal_jet_truth_label"] #1
                labels = 1*labels
                
                if count_files==0:
                    graph_small_example = dataset[2]
            
            #Good_jets = np.array(Good_jets)
            mcweights_out = np.array(mcweights_out)
    
            s_evt = 0
            events = 100
            delta_t_fileax = time.time() - t_start
            #print("Created dataset in {:.4f} seconds.".format(delta_t_fileax))
    
            #test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2 )
    
            print ("dataset dataset size:", len(dataset))
    
    
            #EVALUATING
            #torch.save(model.state_dict(), path)
    
            if choose_model == "LundNet":
                model = LundNet()
                # model = LundNet_old()
            if choose_model == "GATNet":
                model = GATNet()
            if choose_model == "GINNet":
                model = GINNet()
            if choose_model == "EdgeGinNet":
                model = EdgeGinNet()
            if choose_model == "PNANet":
                model = PNANet()
    
            model.load_state_dict(torch.load(path_to_combined_ckpt))
            device = torch.device('cuda') # Usually gpu 4 worked best, it had more memory available
            
            #model.load_state_dict(torch.load(path_to_combined_ckpt, map_location=torch.device('cpu')))
            #device = torch.device('cpu')
            
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
            #Predict scores
            y_pred = get_scores(test_loader, model, device)
            #print(y_pred)
            tagger_scores = y_pred[:,0]
            delta_t_pred = time.time() - t_filestart - delta_t_fileax
            #print("Calculated predicitions in {:.4f} seconds,".format(delta_t_pred))
            
            #Save root files containing model scores
            filename = file.split("/")[-1]
            outfile_path = os.path.join(path_to_outdir, filename)
    
            # tagger_scores += [-1] * (len(dsids) - len(tagger_scores))
            tagger_scores = np.array(tagger_scores)
            
            
            with uproot3.recreate("{}_score_{}.root".format(outfile_path, output_name)) as f:
                treename = "FlatSubstructureJetTree"
                #Declare branch data types
                f[treename] = uproot3.newtree({"EventInfo_mcChannelNumber": "int32",
                                              "EventInfo_mcEventWeight": "float32",
                                              #"EventInfo_NBHadrons": "int32",   # I doubt saving the parents is necessary here
                                              "fjet_nnscore": "float32",        # which is why I didn't include them
                                              "fjet_pt": "float32",
                                              "fjet_eta": "float32",
                                              "fjet_phi": "float32",
                                              "fjet_m": "float32",
                                              "fjet_weight_pt": "float32", 
                                              "labels" : "float32",
                                              "Good_jets" : "float32",
                                               # "ungroomedtruthjet_m" : "float32",
                                               # "ungroomedtruthjet_split12" : "float32",
                                              })
                
                #Save branches
                f[treename].extend({"EventInfo_mcChannelNumber": dsids,
                                    "EventInfo_mcEventWeight": mcweights_out,
                                    #"EventInfo_NBHadrons": NBHadrons,
                                    "fjet_nnscore": tagger_scores,
                                    "fjet_pt": jet_pts,
                                    "fjet_eta": jet_etas,
                                    "fjet_phi": jet_phis,
                                    "fjet_m": jet_ms,
                                    "fjet_weight_pt": ptweights,
                                    "labels" : labels,
                                    "Good_jets" : Good_jets,
                                    # "ungroomedtruthjet_m" : truth_ungroomedjet_m,
                                    # "ungroomedtruthjet_split12" : truth_ungroomedjet_split12,
                                    })
    
            delta_t_save = time.time() - t_start - delta_t_fileax - delta_t_pred
            print("Saved data in {:.4f} seconds.".format(delta_t_save))
    
            #nentries = 0
            #Time statistics
            nentries_done += uproot3.numentries(file, intreename)
            time_per_entry = (time.time() - t_start)/(nentries_done+1)
            eta = time_per_entry * (nentries_total - nentries_done)
    
            print("Evaluated on {} out of {} events".format(nentries_done, nentries_total))
            print("Estimated time until completion: {}".format(str(timedelta(seconds=eta))))
        print("Total evaluation time: {:.4f} seconds.".format(time.time()-t_filestart))

    
    return


if __name__ == "__main__":
    main()
