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
import onnxruntime as ort
import scipy.sparse as ss
from datetime import datetime, timedelta
from torch_geometric.utils import degree
from torch_geometric.data import DataListLoader, DataLoader

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    args = parser.parse_args()
    config_file = args.config
    config = load_yaml(config_file)
    config_signal = load_yaml("configs/config_signal.yaml")
    signal = config_signal["signal"]

    path_to_test_file = config['data']['path_to_test_file']
    files = glob.glob(path_to_test_file)

    print ("path_to_test_file:",path_to_test_file)
    print ("files:",files)
    path_to_outdir = config['data']['path_to_outdir']

    path_to_combined_ckpt = config['test']['path_to_combined_ckpt']
    path_to_onnx= config['test']['path_to_onnx_ckpt']
    
    print("ckpt used:", path_to_combined_ckpt )
    print("ckpt used onnx:", path_to_onnx )

    output_name = config['test']['output_name']
    kT_selection = config['data']['kT_cut']
    
    files = glob.glob(path_to_test_file)
    intreename = "AnalysisTree"

    nentries_total = 0
    nentries_done = 0

    learning_rate = 0.0005
    batch_size = 2000
    scale_factor = 1
    
    for file in files:
        with uproot.open(file) as f:
            nentries_total += f[intreename].num_entries

    print("Evaluating on {} files with {} entries in total.".format(len(files), nentries_total))
    
    jet_type = "Akt10UFOJet"
    t_filestart = time.time()
    choose_model = config['test']['choose_model']
    count_files = -1
    graph_small_example = []

    first= True
    for file in files:
        t_start = time.time()
        dataset = []
        dataset_onnx = []
        print("Loading file",file)
        Good_jets = [] 
        Good_jets2 = []
        mcweights_out = []
        mcweights_out2 = []
        
        with uproot.open(file) as infile:
            tree = infile[intreename]
            nentries_file = tree.num_entries

            count_files += 1
            dsids_test = tree["dsid"].array(library="np")
            if dsids_test[0] in  config_signal[signal]["skip_dsids"]: 
               continue

            jet_pts_truth = ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) )
            ptweights = np.ones_like( ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) ))
            labels = ak.to_numpy(ak.flatten(tree["LRJ_truthLabel"].array(library="ak")) )
            dsids = dsids_test[0]*np.ones_like(ak.to_numpy(ak.flatten(tree["LRJ_pt"].array(library="ak")) ))
            LRJ_pt_ref = tree["LRJ_pt"].array(library="np") 
            mcweights = tree["mcEventWeight"].array(library="np")  
            mcweights = flatten_small_branch(LRJ_pt_ref, mcweights)

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
 
            flat_weights = GetPtWeight_all_MC( labels, dsids_test,  jet_pts, 5, Pythia_or_All=True)

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
                
                graph_small_example = Data(x= vec.detach() ,
                               edge_index = edge.detach() ,
                               Ntrk=torch.tensor(7, dtype=torch.float).detach(),
                               weights= torch.tensor(1, dtype=torch.float).detach(),
                               graph_size = torch.tensor(2, dtype=torch.float).detach(),
                               mass= float(80) ,
                               y= float(0) )

            dataset = create_train_dataset_fulld_new_Ntrk_pt_weight_file_test_export(
                dataset, graph_small_example, all_lund_zs, all_lund_kts, all_lund_drs,
                parent1, parent2, flat_weights, labels,
                N_tracks, jet_pts, jet_ms, kT_selection,
                mcweights, mcweights_out, Good_jets,
                config_signal[signal]["signal_jet_truth_label"]
            )
            
            dataset_onnx = create_train_dataset_fulld_new_Ntrk_pt_weight_file_test_onnx(
                dataset_onnx, graph_small_example, all_lund_zs, all_lund_kts, all_lund_drs,
                parent1, parent2, flat_weights, labels,
                N_tracks, jet_pts, jet_ms, kT_selection,
                mcweights, mcweights_out2, Good_jets2,
                config_signal[signal]["signal_jet_truth_label"]
            )

            labels = labels==config_signal[signal]["signal_jet_truth_label"]
            labels = 1*labels
            
            if count_files==0:
                graph_small_example = dataset[2]
        
        mcweights_out = np.array(mcweights_out)

        s_evt = 0
        events = 100
        delta_t_fileax = time.time() - t_start

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        test_loader_onnx = onnx_data_loader(dataset_onnx, batch_size, shuffle_batches=False)
        #test_loader_onnx = onnx_data_loader(dataset_onnx, batch_size=batch_size)
        
        print ("dataset size:", len(dataset))
        print ("dataset_onnx size", len (dataset_onnx))

        # === Load model ===
        model = LundNet4Class()
        model.load_state_dict(torch.load(path_to_combined_ckpt, map_location=torch.device('cpu')))
        device = torch.device('cpu')
        model.to(device)
        model.eval()

        if first == True:
            model_for_export = LundNet4Class()
            model_for_export.load_state_dict(torch.load(path_to_combined_ckpt, map_location="cpu"))
            model_for_export.eval()
            
 
            def print_batchnorm_stats(model):
                print("\n🔎 BatchNorm stats en PyTorch (antes de exportar):")
                for name, m in model.named_modules():
                    if isinstance(m, torch.nn.BatchNorm1d):
                        print(f"{name}: running_mean[0:5] = {m.running_mean[:5].cpu().numpy()}, "
                              f"running_var[0:5] = {m.running_var[:5].cpu().numpy()}")

            example_batch = next(iter(test_loader))
            example_batch = example_batch.to(device)
            x_ex         = example_batch.x
            edge_index_ex= example_batch.edge_index
            batch_ex     = example_batch.batch
            Ntrk_ex      = example_batch.Ntrk
            counts_ex    = torch.bincount(batch_ex)
            print("x_ex:",          type(x_ex),          "dtype:", x_ex.dtype,          "shape:", x_ex.shape)
            print("edge_index_ex:", type(edge_index_ex), "dtype:", edge_index_ex.dtype, "shape:", edge_index_ex.shape)
            print("batch_ex:",      type(batch_ex),      "dtype:", batch_ex.dtype,      "shape:", batch_ex.shape)
            print("Ntrk_ex:",       type(Ntrk_ex),       "dtype:", Ntrk_ex.dtype,       "shape:", Ntrk_ex.shape)
            print("counts_ex:",     type(counts_ex),     "dtype:", counts_ex.dtype,     "shape:", counts_ex.shape)

            with torch.no_grad():
                _ = model_for_export(x_ex, edge_index_ex, batch_ex, Ntrk_ex, counts_ex)
        
            # Imprimir stats de BatchNorm
            print("DEBERIA SER AQUI")
            print_batchnorm_stats(model_for_export) 
            print("DEBERIA SER AQUI")
            #replace_batchnorm_with_frozen(model_for_export)
            # === Export to ONNX ===
            torch.onnx.export(
                model_for_export,
                (x_ex, edge_index_ex, batch_ex, Ntrk_ex,counts_ex),
                path_to_onnx,
                input_names=["x", "edge_index", "batch", "Ntrk","counts"],
                output_names=["output"],
                dynamic_axes={
                    "x": {0: "num_nodes"},
                    "edge_index": {1: "num_edges"},
                    "batch": {0: "num_nodes"},
                    "Ntrk": {0: "batch_size"},
                    "counts": {0: "batch_size"},
                },
                opset_version=17,
                do_constant_folding=False,  
                #do_constant_folding=True,
                keep_initializers_as_inputs=False
            )
            print(f"Model exported to {path_to_onnx}")
            import onnx
            onnx_model = onnx.load(path_to_onnx)
            nodes = [n.op_type for n in onnx_model.graph.node]
            div_nodes = [n for n in onnx_model.graph.node if n.op_type == "Div"]
            
            for i, div in enumerate(div_nodes):
                print(f"\n--- Div #{i} ---")
                print("Inputs:", div.input)
                print("Outputs:", div.output)

        first=False

        # ============================================================
        # === Evaluate PyTorch (multiclass)
        # ============================================================
        y_pred = get_scores_multi(test_loader, model, device)
        
        # ============================================================
        # === Evaluate ONNX (multiclass)
        # ============================================================
        onnx_model = ort.InferenceSession(path_to_onnx)
        y_pred_onnx = evaluate_onnx_multi(test_loader_onnx, onnx_model)
        # shape: (N_jets, n_classes)
        
        # ============================================================
        # === Sanity checks
        # ============================================================
        print("PyTorch scores shape:", y_pred.shape)
        print("ONNX scores shape:", y_pred_onnx.shape)
        print("dsids:", len(dsids))
        print("mcweights_out:", len(mcweights_out))
        
 
        n_jets = y_pred.shape[0]
        y_pred_onnx = y_pred_onnx[:n_jets]
        assert y_pred.shape == y_pred_onnx.shape
        assert y_pred.shape[0] == len(dsids)
        
        # ============================================================
        # === Output file
        # ============================================================
        filename = file.split("/")[-1]
        outfile_path = os.path.join(path_to_outdir, filename)
        outfile = f"{outfile_path}_score_{output_name}.root"
        treename = "FlatSubstructureJetTree"
        
        # ============================================================
        # === Define class names 
        # ============================================================
        class_names = ["cat1", "cat2", "cat3", "cat4"]  # ejemplo
        n_classes = y_pred.shape[1]
        assert n_classes == len(class_names)
        
        # ============================================================
        # === Build branches
        # ============================================================
        branches = {
            "EventInfo_mcChannelNumber": np.array(dsids, dtype="int32"),
            "EventInfo_mcEventWeight":   np.array(mcweights_out, dtype="float32"),
            "fjet_pt":                   np.array(jet_pts, dtype="float32"),
            "fjet_eta":                  np.array(jet_etas, dtype="float32"),
            "fjet_phi":                  np.array(jet_phis, dtype="float32"),
            "fjet_m":                    np.array(jet_ms, dtype="float32"),
            "fjet_weight_pt":            np.array(ptweights, dtype="float32"),
            "labels":                    np.array(labels, dtype="float32"),
            "Good_jets":                 np.array(Good_jets, dtype="float32"),
        }
        
        # === one branch per class (PyTorch + ONNX)
        for i, cname in enumerate(class_names):
            branches[f"fjet_nnscore_{cname}"] = y_pred[:, i].astype("float32")
            branches[f"fjet_nnscore_onnx_{cname}"] = y_pred_onnx[:, i].astype("float32")
        

        with uproot.recreate(outfile) as f:
            f[treename] = branches
        

        delta_t_save = time.time() - t_start - delta_t_fileax
        print("Saved data in {:.4f} seconds.".format(delta_t_save))
        
        nentries_done += nentries_file
        time_per_entry = (time.time() - t_start) / (nentries_done + 1)
        eta = time_per_entry * (nentries_total - nentries_done)
        
        print("Evaluated on {} out of {} events".format(nentries_done, nentries_total))
        print("Estimated time until completion: {}".format(str(timedelta(seconds=eta))))

        print("Total evaluation time: {:.4f} seconds.".format(time.time() - t_filestart))

