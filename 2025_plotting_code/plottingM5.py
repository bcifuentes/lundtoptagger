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
        

#BS
#STUDIES=["BS= 4800","BS= 5600","BS= 1024","BS= 4800mix"]
#SZ
#STUDIES=["5%-1%","7.5%-1.5%","9%-1.8%","BEST"]
#LR
#STUDIES=["lr= 1e-4","lr= 2e-4","lr= 4e-4","BEST"]
#EP
#STUDIES=["Every 3 epoch ","Every 4 epoch","Every 5 epoch"]
#TS
#STUDIES=["TS=0.1, BS=4800","TS=0.2, BS=4800","TS=0.1, BS=5600","TS=0.2, BS=5600"]

#ALL

#MANUAL
#STUDIES=["Usual","Manual"]

'''
Hadd15MIXQCD_75P8TOP_BS4800_LR000005_E4MoreDataTest/  
Hadd15MIXQCD_75P8TOP_BS4800_LR0001_E4MoreDataTest/    
Hadd15P8QCD_75P8TOP_BS5600_LR000005_E4MoreDataTest/
Hadd15MIXQCD_75P8TOP_BS4800_LR0002_E4MoreDataTest/    
Hadd15P8QCD_75P8TOP_BS5600_LR0002_E4MoreDataTest/
Hadd15MIXQCD_75P8TOP_BS4800_LR0004_E4MoreDataTest/    
Hadd15P8QCD_75P8TOP_BS5600_LR0002_E4_TS2MoreDataTest/
Hadd15P8QCD_75P8TOP_BS1024_LR0002_E4MoreDataTest/     
Hadd18MIXQCD_9P8TOP_BS1024_LR0004_E5MoreDataTest/
Hadd15P8QCD_75P8TOP_BS4800_LR000005_E4MoreDataTest/   
Hadd18MIXQCD_9P8TOP_BS4800_LR0002_E4MoreDataTest/
Hadd15P8QCD_75P8TOP_BS4800_LR0001_E4MoreDataTest/     
Hadd18MIXQCD_9P8TOP_BS4800_LR0004_E5MoreDataTest/
Hadd15P8QCD_75P8TOP_BS4800_LR0002_E3MoreDataTest/     
Hadd18MIXQCD_9P8TOP_BS5600_LR0005_E5MoreDataTest/
Hadd15P8QCD_75P8TOP_BS4800_LR0002_E4MoreDataTest/     
Hadd18MIXQCD_9P8TOP_BS5600_LR0006_E5MoreDataTest/
Hadd15P8QCD_75P8TOP_BS4800_LR0002_E4_TS2MoreDataTest/ 
Hadd1MIXQCD_5P8TOP_BS4800_LR0002_E4MoreDataTest/
Hadd15P8QCD_75P8TOP_BS4800_LR0002_E5MoreDataTest/     
Hadd15P8QCD_75P8TOP_BS4800_LR0004_E4MoreDataTest/

'''
OUTDIR=[]
PATH_TAGGERS=[]
IND=[]
SSTUDIES=[]
'''

#lr studies (BS=4800 , E4 ,15P8QCD_75P8TOP):
outdir='./OUTPUTS/outp_lr_4800E415P8QCD75P8TOP/'
path_taggers = ["Hadd15P8QCD_75P8TOP_BS4800_LR0001_E4MoreDataTest/", 
                "Hadd15P8QCD_75P8TOP_BS4800_LR0002_E4MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS4800_LR0004_E4MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS4800_LR000005_E4MoreDataTest/"
]
ind=2
STUDIES=["lr= 1e-4","lr= 2e-4","lr= 4e-4","lr = 5e-6"]

OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)



#whole Pythia studie:
outdir='./OUTPUTS/ALL_PYTHIA'

path_taggers = ["Hadd15P8QCD_75P8TOP_BS5600_LR000005_E4MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS5600_LR0002_E4MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS5600_LR0002_E4_TS2MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS1024_LR0002_E4MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS4800_LR000005_E4MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS4800_LR0002_E3MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS4800_LR0002_E5MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS4800_LR0004_E4MoreDataTest/"]
ind=-1
STUDIES=["1","2","3","4","5","6","7","8"]


OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)

#bs studies (lr=0002 , E4 ,15P8QCD_75P8TOP):
outdir='./OUTPUTS/outp_bs_0002E415P8QCD75P8TOP/'

path_taggers = ["Hadd15P8QCD_75P8TOP_BS4800_LR0002_E4MoreDataTest/", 
                "Hadd15P8QCD_75P8TOP_BS5600_LR0002_E4MoreDataTest/", 
                "Hadd15P8QCD_75P8TOP_BS1024_LR0002_E4MoreDataTest/"
]
ind=0
STUDIES=["BS= 4800","BS= 5600","BS= 1024"]

OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)


#size studies (lr=0002, E4 , MIX, BS=4800):
outdir='./OUTPUTS/outp_sz_4800E4MIX0002/'

path_taggers = ["Hadd1MIXQCD_5P8TOP_BS4800_LR0002_E4MoreDataTest/", 
                "Hadd15MIXQCD_75P8TOP_BS4800_LR0002_E4MoreDataTest/",
                "Hadd18MIXQCD_9P8TOP_BS4800_LR0002_E4MoreDataTest/",
]
ind=1
STUDIES=["5%-1%","7.5%-1.5%","9%-1.8%"]

OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)

#MIX log data studies (E4, 18MIXQCD_9P8TOP)
outdir='./OUTPUTS/outp_MIX_E418MIXQCD9P8TOP/'

path_taggers = ["Hadd18MIXQCD_9P8TOP_BS1024_LR0004_E5MoreDataTest/", 
                "Hadd18MIXQCD_9P8TOP_BS5600_LR0006_E5MoreDataTest/",
                "Hadd18MIXQCD_9P8TOP_BS5600_LR0005_E5MoreDataTest/",
                "Hadd18MIXQCD_9P8TOP_BS4800_LR0004_E5MoreDataTest/"
]
ind=-1
STUDIES=["1024/0004","5600/0005","5600/0006","4800/0004"]

OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)

#Epoch studies (15P8QCD_75P8TOP_BS4800_LR0002)
outdir='./OUTPUTS/outp_ep_15P8QCD75P8TOPBS4800LR0002/'

path_taggers = ["Hadd15P8QCD_75P8TOP_BS4800_LR0002_E3MoreDataTest/",
                "Hadd15P8QCD_75P8TOP_BS4800_LR0002_E4MoreDataTest/" ,
                "Hadd15P8QCD_75P8TOP_BS4800_LR0002_E5MoreDataTest/"]
ind=1
STUDIES=["Every 3 epoch ","Every 4 epoch","Every 5 epoch"]

OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)
'''

'''
path= "/home/bcifuentes/outp/2025_scores_1qcd_10w/"
outdir='./OUTPUTS/outp_comp_mix_p8_W/'

path_taggers = ["mixWScores/","pythiaWScores/"]

ind=1
STUDIES=["Mix Tagger","Pythia Tagger"]

OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)
'''

'''
path= "/home/bcifuentes/outp/2025_scores_1qcd_5s/HADEADOS/top/"
outdir='./OUTPUTS/outp_comp_mix_p8_Top/'

path_taggers = ["mixTopScores/","pythiaTopScores/"]

ind=1
STUDIES=["Mix Tagger","Pythia Tagger"]

OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)


outdir="./OUTPUTS/MANUAL/"

path_taggers=["USUAL/","MANUAL/"]
ind=0
STUDIES=["Standard","Exportable"]

OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)


path= "/home/bcifuentes/outp/2025_scores_1qcd_5s/HADEADOS/"
outdir='./OUTPUTS/outp_comp_TOPKTCUTS_HA/'

path_taggers = ["pythiaTopScores/","cut_-1/","cut_-05","cut_0","cut_05","cut_1","cut_2"]
ind=0
STUDIES=["None","-1.0","-0.5","0.0","0.5","1.0","2.0"]

'''
path= "/home/bcifuentes/SEPT_ONNX/export/FINALSC/"
outdir='./onnxporfin/'

path_taggers = ["onnxSC1","torchSC1"]
ind=1
STUDIES=["ONNX", "PyTorch"]

'''


path= "/home/bcifuentes/outp/2025_scores_1qcd_10w/"
outdir='./OUTPUTS/outp_comp_WKTCUTS_HA/'

path_taggers = ["pythiaWScores/","cut_-1/","cut_-05","cut_0","cut_05","cut_1","cut_2"]
ind=0
STUDIES=["None","-1.0","-0.5","0.0","0.5","1.0","2.0"]

path="/home/bcifuentes/JUL_outp/2025_SCORES/"
outdir="./OUTPUTS/outp_FLATANN/"

path_taggers = ["USUAL/","ANN198/","ANN320","FLAT/"]
ind=0
STUDIES=["LundNetNN","LundNetANNe198","LundNetANNe320","LundNetNN (FlatMass)"]


outdir='./OUTPUTS/outp_new_cuts/'
path=""
path_taggers= ["/home/bcifuentes/JUL_outp/2025_SCORES/hadds_18MIXQCD_9P8TOP_BS5600_LR0005/","/home/bcifuentes/outp/2025_scores_1qcd_5s/HADEADOS/mixTopScores/"]
ind=0
STUDIES=["New Cuts", "Old tagger"]
'''

OUTDIR.append(outdir)
PATH_TAGGERS.append(path_taggers)
IND.append(ind)
SSTUDIES.append(STUDIES)


for outdir,path_taggers,ind,STUDIES in zip(OUTDIR,PATH_TAGGERS,IND,SSTUDIES):
    #file_types = ["LundNet_class", "SherpaLund", "SherpaCluster","HerwigDipole","HerwigAngular"]
    #gen_types = ["Pythia", "SherpaLund", "SherpaCluster","HerwigDipole","HerwigAngular"]
    
    file_types = ["LundNet_class"]
    gen_types = ["SCORES"]
    #file_types = ["LundNet_class","HerwigAngular"]
    #gen_types = ["Pythia","HerwigAngular"]
    
    # Load the dictionary with the best models (generated earlier)
    with open("./better_models.json", "r") as f:
        best_models_dict = json.load(f)
    
    # Function to normalize tagger names
    def normalize_tagger(path_tagger):
        path_tagger = path_tagger.replace("Hadd", "")
        path_tagger = path_tagger.replace("MoreDataTest", "")
        path_tagger = path_tagger.replace("/", "")
        return path_tagger
    
    # Build the list of (epoch, val_loss) tuples
    MODELS = []
    for pt in path_taggers:
        normalized = normalize_tagger(pt)
        print(normalized)
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
    plotAlternativeFinal(TAGGERS,STUDIES,TAGGERS[ind],file_types,weight="chris_weight", prefix=outdir, NNorANN='NN',wp=working_point)
    #pt_bgrej_envelope_total_NV(TAGGERS,STUDIES, weight="fjet_weight_pt", prefix=outdir, wp=working_point)
    #pt_bgrej_Vall_Loss(TAGGERS,STUDIES,MODELS, weight="fjet_weight_pt", prefix=outdir, wp=working_point)
    
    make_efficiencies_all_2(TAGGERS,STUDIES, prefix=outdir)
    #make_rocs_2(TAGGERS,STUDIES,prefix=outdir)
    
    '''
    for taggers,cut in zip(TAGGERS,CUTS):
        make_rocs(cut,taggers,prefix=outdir)
        make_efficiencies_all(cut,taggers, prefix=outdir)
        mass_sculpting(cut,taggers, weight="fjet_weight_pt", prefix=outdir, wp=working_point)  
        pt_bgrej_otherMC(cut,taggers,weight="fjet_weight_pt", prefix=outdir, wp=working_point)    
        plotAlternative(cut,taggers, weight="fjet_weight_pt", prefix=outdir, NNorANN='NN', wp=working_point)
        make_efficiencies_3var(cut,taggers, prefix=outdir)
    '''
    
