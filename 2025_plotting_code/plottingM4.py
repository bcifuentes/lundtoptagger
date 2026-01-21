#!/usr/bin/env python

from utils_plotsM3 import *
#from utils_plots_Copy1 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import ROOT
from root_numpy import fill_hist  as fh
import warnings
warnings.filterwarnings('ignore')
import os
import gc

path="/home/bcifuentes/SEP_outp/2025_SCORES_exp/"
#path="/home/bcifuentes/JUL_ONNX/"
#path="/home/bcifuentes/MAY_outp/2025_SCORES/"
path="/home/bcifuentes/OCT_outp/2025_SCORES/"
#path="/home/bcifuentes/SEPT_ONNX/export/FINALSC/"
path="/home/bcifuentes/outp/2025_scores_1qcd_5s/HADEADOS/"
#path="/home/bcifuentes/outp/2025_scores_1qcd_10w/"
Do_primary = False

for carpeta in os.listdir(path):
    Tpath = os.path.join(path, carpeta)
    
    print("Carpeta encontrada: {:}".format(os.path.basename(Tpath)))
    taggerpath= os.path.basename(Tpath)

    
    
    taggersC1 = {}
    taggersC2 = {}
    taggersC3 = {}
    taggersC4 = {}
    taggersC5 = {}
    taggersC6 = {}
    taggersC7 = {}
    taggersC8 = {}
    taggersMix = {}
    
    tagger_filesC1 = {}
    tagger_filesC2 = {}
    tagger_filesC3 = {}
    tagger_filesC4 = {}
    tagger_filesC5 = {}
    tagger_filesC6 = {}
    tagger_filesC7 = {}
    tagger_filesC8 = {}
    tagger_files_Mix = {}
    
    other_MC_tagger_files = {}
    #if "SC1"  in taggerpath:  
    if taggerpath=="pythiaTopScores":
        outdir='./outp_'+taggerpath

        CUTS=["None"]
            
        if not Do_primary:
            #tagger_filesC1["LundNet_class"]    = path+ taggerpath+ "/LundNetScores.root"
            tagger_filesC1["LundNet_class"]    = path+ taggerpath+ "/Pythia.root"
            #tagger_filesC1["LundNet_class"]    = path+ taggerpath+ "/LundNet_R22.root"
            tagger_filesC1["SherpaCluster"]      = path+ taggerpath+ "/SherpaCluster.root"
            tagger_filesC1["SherpaLund"]         =  path+ taggerpath+ "/SherpaLund.root"
            tagger_filesC1["HerwigDipole"]       = path+ taggerpath+ "/HerwigDipole.root"
            tagger_filesC1["HerwigAngular"]     = path+ taggerpath+ "/HerwigAngular.root"
        
        else:
            pass

    else:
        print("Saltado")
        continue

    try:
        os.system("mkdir {}".format(outdir))

    except:
        print("{} already exists".format(prefix))
    
    
    
    try:
        for cut in CUTS:
            os.system("mkdir {}/{}".format(outdir,cut))
    except:
        print("Cuts folders already exists")
        
    
       
    working_point = 0.5
    #'''
    if not Do_primary:

        TAGGERS=[taggersC1]
        TAGGER_FILES=[tagger_filesC1]
        
    else:
        pass

    
    for tagger_files,taggers in zip(TAGGER_FILES,TAGGERS):
        for t in tagger_files:
            print("init",t)
            taggers[t] = tagger_scores(t,tagger_files[t], working_point)
    

    for taggers in TAGGERS:
        pol_func = get_wp_tag_pol_func(taggers["LundNet_class"], working_point)
        
        for t in taggers:
            if taggers[t].name == "3var":
                continue
            if taggers[t].name == "HerwigAngular" or taggers[t].name == "HerwigDipole" or taggers[t].name == "SherpaCluster"  or taggers[t].name == "SherpaLund" : 
                get_tag_other_MC(taggers[t], pol_func, working_point)
            else:
                get_wp_tag(taggers[t],working_point, prefix=outdir)  ## smooth function

    if not Do_primary:     
        
        for taggers,cut in zip(TAGGERS,CUTS):
            #make_rocs(cut,taggers,prefix=outdir)
            make_efficiencies_all(cut,taggers, prefix=outdir)
            make_roc_otherMC(cut,taggers, prefix=outdir)
            #mass_sculpting(cut,taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)
            #make_efficiencies_all_2(taggers,[taggers[t].name for t in taggers], prefix=outdir)
            pt_bgrej_otherMC(cut,taggers,weight="fjet_weight_pt", prefix=outdir, wp=working_point)    
            #plotAlternative(cut,taggers, weight="fjet_weight_pt", prefix=outdir, NNorANN='NN', wp=working_point)
            #make_efficiencies_3var(cut,taggers, prefix=outdir)

    else:
        pass
    gc.collect()
