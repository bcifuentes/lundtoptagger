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
from tools.GNN_model_weight.utils_newdata_nnode import *

import gc
print("Libraries loaded!")


def main():
    
    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)
    config_signal = load_yaml("configs/config_signal.yaml") # TODO: make this an optional argument, but then the same file needs to be used in utils_newdata.py
    signal = config_signal["signal"]

    path_to_file = config['data']['path_to_trainfiles']
    files = glob.glob(path_to_file)

    jet_type = "Akt10UFOJet" #UFO jets
    save_trained_model = True
    intreename = "AnalysisTree"

    #print("Training tagger on files", len(files))
    t_start = time.time()

    file_number = 0
    
    dataset = []
    primary_Lund_only_one_arr = []
    
    for file in files:
        
        #print("Loading file",file)
        with uproot.open(file) as infile:
            tree = infile[intreename]
            file_number += 1
            
            dsids_test = tree["dsid"].array(library="np")
            if dsids_test[0] in config_signal[signal]["skip_dsids"]: # don't lose time with jets that don't pass pt cut or wrong signal sample
                continue

            dsids = ak.to_numpy(ak.flatten(tree["LRJ_truthLabel"].array(library="ak")) )

            print("length dataset:", len(dataset), " file number:", file_number)
            parent1 = ak.flatten(tree["jetLundIDParent1"].array(library="ak")) 
            parent2 = ak.flatten(tree["jetLundIDParent2"].array(library="ak")) 
            #print(parent1[0])
            #print(parent2[0])
            jet_ms = ak.to_numpy(ak.flatten(tree["LRJ_mass"].array(library="ak")))
            all_lund_zs = ak.flatten(tree["jetLundZ"].array(library="ak")) 
            all_lund_kts = ak.flatten(tree["jetLundKt"].array(library="ak")) 
            all_lund_drs = ak.flatten(tree["jetLundDeltaR"].array(library="ak")) 
            N_tracks = ak.to_numpy(ak.flatten(tree["LRJ_Nconst_Charged"].array(library="ak")) )
            #N_tracks = ak.to_numpy(ak.flatten(tree["LRJ_Ntrk500"].array(library="ak")) )
            #N_tracks = ak.to_numpy(ak.flatten(tree["LRJ_Nconst"].array(library="ak")) )
            #print(N_tracks)
            jet_pts = ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) )

            #parent1 = ak.to_numpy(parent1)
            #parent2 = ak.to_numpy(parent2)
            #all_lund_zs = ak.to_numpy(all_lund_zs)
            #all_lund_kts = ak.to_numpy(all_lund_kts)
            #all_lund_drs = ak.to_numpy(all_lund_drs)
            
            labels = dsids

            #flat_weights = GetPtWeight_2( dsids, jet_pts, 5)
            flat_weights = GetPtWeight_all_MC( labels, dsids_test,  jet_pts, 5, Pythia_or_All=True)
            kT_selection = config['architecture']['kT_cut']

            #dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file( dataset , all_lund_zs, all_lund_kts, all_lund_drs, parent1, parent2, flat_weights, labels ,N_tracks, jet_pts, jet_ms, kT_selection)
            dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file(
                dataset, all_lund_zs, all_lund_kts, all_lund_drs,
                parent1, parent2, flat_weights, labels,
                N_tracks, jet_pts, jet_ms, kT_selection,
                primary_Lund_only_one_arr,
                config_signal[signal]["signal_jet_truth_label"]
            )

            gc.collect()

    print("Dataset created!", " len():",len(dataset))
    delta_t_fileax = time.time() - t_start
    print("Created dataset in {:.4f} seconds.".format(delta_t_fileax))

    path_to_save = config['data']['path_to_save']
    output_path_graphs = path_to_save + "/graphs_NewDataset_"

    torch.save(dataset, output_path_graphs + config['data']['model_name'])

    return


if __name__ == "__main__":
    main()
