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
import json
import re

path="/home/bcifuentes/MAY_outp/2025_SCORES/"
        
outdir='./outp_ALLMIX/'

#BS
#STUDIES=["BS= 4800","BS= 5600","BS= 1024","BS= 4800mix"]
#SZ
#STUDIES=["5%-1%","7.5%-1.5%","9%-1.8%"]
#LR
#STUDIES=["lr= 4e-4","lr= 2e-4","lr= 1e-4","lr= 5e-6"]
#EP
#STUDIES=["Every 3 epoch ","Every 4 epoch","Every 5 epoch"]
#TS
#STUDIES=["TS=0.1, BS=4800","TS=0.2, BS=4800","TS=0.1, BS=5600","TS=0.2, BS=5600"]

#ALL

#MANUAL
#STUDIES=["Usual","Manual"]

'''
path_tagger1="Hadd15P8QCD_75P8TOP_BS4800_LR0002_E3MoreDataTest"
path_tagger2="Hadd15P8QCD_75P8TOP_BS4800_LR0002_E4MoreDataTest"
path_tagger3="Hadd15P8QCD_75P8TOP_BS4800_LR0002_E5MoreDataTest"
path_tagger4="Hadd15P8QCD_75P8TOP_BS5600_LR0002_E4MoreDataTest"
path_tagger5="Hadd15P8QCD_75P8TOP_BS1024_LR0002_E4MoreDataTest"
path_tagger6="Hadd15P8QCD_75P8TOP_BS4800_LR0001_E4MoreDataTest"
path_tagger7="Hadd15P8QCD_75P8TOP_BS4800_LR0004_E4MoreDataTest"
'''

path_tagger1="Hadd1MIXQCD_5P8TOP_BS4800_LR0002_E4MoreDataTest"
path_tagger2="Hadd15MIXQCD_75P8TOP_BS4800_LR0001_E4MoreDataTest"
path_tagger3="Hadd15MIXQCD_75P8TOP_BS4800_LR0002_E4MoreDataTest"
path_tagger4="Hadd15MIXQCD_75P8TOP_BS4800_LR0004_E4MoreDataTest"
path_tagger5="Hadd18MIXQCD_9P8TOP_BS4800_LR0002_E4MoreDataTest"


path_taggers = [path_tagger1, path_tagger2, path_tagger3,path_tagger4,path_tagger5]

STUDIES=["1","2","3","4","5"]


file_types = ["LundNet_class", "SherpaLund", "SherpaCluster","HerwigDipole","HerwigAngular"]
gen_types = ["Pythia", "SherpaLund", "SherpaCluster","HerwigDipole","HerwigAngular"]

#file_types = ["LundNet_class"]
#gen_types = ["Pythia"]


# Load the dictionary with the best models (generated earlier)
with open("better_models.json", "r") as f:
    best_models_dict = json.load(f)

# Function to normalize tagger names
def normalize_tagger(path_tagger):
    path_tagger = path_tagger.replace("Hadd", "")
    path_tagger = path_tagger.replace("MoreDataTest", "")
    return path_tagger

# Build the list of (epoch, val_loss) tuples
MODELS = []
for pt in path_taggers:
    normalized = normalize_tagger(pt)
    if normalized in best_models_dict:
        info = best_models_dict[normalized]
        MODELS.append((info["epoch"], info["val_loss"]))

print(MODELS)

TAGGER_FILES=[{} for _ in range(len(path_taggers))]
TAGGERS=[{} for _ in range(len(path_taggers))]

for file,gen in zip(file_types,gen_types):
    #tagger_filesC1[file]    = path+ "Hadd_1P8QCD_5P8TOP_BS2000_LR0002_E4_MANUAL"+ "/USUAL.root".format(gen)
    #tagger_filesC2[file]    = path+ "Hadd_1P8QCD_5P8TOP_BS2000_LR0002_E4_MANUAL"+ "/MANUAL.root".format(gen)
    
    for i,pathtagger in zip(range(len(path_taggers)),path_taggers):
        TAGGER_FILES[i][file] = path + pathtagger + "/{:}.root".format(gen)  


try:
    os.system("mkdir {}".format(outdir))
    
except:
    print("{} already exists".format(prefix))
    
       
working_point = 0.5
#'''

#TAGGER_FILES=[tagger_filesC1,tagger_filesC2,tagger_filesC3,tagger_filesC4]
#TAGGERS=[taggersC1,taggersC2,taggersC3,taggersC4]
    
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
            get_wp_tag(taggers[t],working_point, prefix=outdir) 

     
#pt_bgrej_comp(TAGGERS,STUDIES,file_types, weight="fjet_weight_pt", prefix=outdir, wp=working_point)
#plotAlternativeFinal(TAGGERS,STUDIES,TAGGERS[0],file_types,weight="chris_weight", prefix=outdir, NNorANN='NN',wp=working_point)
pt_bgrej_envelope_total_NV(TAGGERS,STUDIES, weight="fjet_weight_pt", prefix=outdir, wp=working_point)
pt_bgrej_Vall_Loss(TAGGERS,STUDIES,MODELS, weight="fjet_weight_pt", prefix=outdir, wp=working_point)

'''
for taggers,cut in zip(TAGGERS,CUTS):
    make_rocs(cut,taggers,prefix=outdir)
    make_efficiencies_all(cut,taggers, prefix=outdir)
    mass_sculpting(cut,taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  
    pt_bgrej_otherMC(cut,taggers,weight="fjet_weight_pt", prefix=outdir, wp=working_point)    
    plotAlternative(cut,taggers, weight="fjet_weight_pt", prefix=outdir, NNorANN='NN', wp=working_point)
    make_efficiencies_3var(cut,taggers, prefix=outdir)
'''

