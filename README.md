# lundtoptagger

samples and flat weights are here : 
```
/data/jmsardain/LJPTagger/FullSplittings/SplitForTopTagger/ 
```

For the training, the main changes one should do are in the configuration file: configs/config_class_train_top.yaml .
In this file you will define the learning rate, batch size, the input files, the model to use, the repo to save your ckpts. 

To run the training: 
```
## if you are not working on UChicago, do not do the first 2 lines
source /data/jmsardain/LJPTagger/JetTagging/miniconda/bin/activate
conda activate rootenv

python weight_class_train.py configs/config_class_train_top.yaml
```

For the testing, you should run the final_makescores notebook. The only changes you should do are under the conditions in the for loop. The different variables should point to your test files, the ckpt you want to use and the repo to save your output root files 

At the end, when you are done with the testing, make sure you hadd all the root files together: 
```
hadd -f tree.root user.*root
```

## Plotting

To generate the tagger plots, first open a clean terminal and set up the environment:

```bash
setupATLAS -c centos7
source setup.sh
```


### Main tagger plots: `plottingM4.py`

This script generates the standard tagger plots:

- ROC curves
- Signal efficiency vs background rejection
- Background rejection vs jet transverse momentum (pT), comparing the tagger's performance across different Monte Carlo generators
- Envelope plots for each tagger (e.g. useful for studying kt cut effects)

To run it:

```bash
python plottingM4.py
```

The script automatically reads scores from a group of folders organized by taggers and MC generators.

### Comparative studies: `plottingM5.py`

This script is aimed at broader comparative studies. It generates:

- Background rejection vs pT plots where the MC generators are averaged to absorb generator-specific behaviors
- Envelope plots for each model
- Validation loss plots (including envelope versions)

Currently, it is configured for hyperparameter optimization studies.

To run it:

```bash
python plottingM5.py
```

### Notes

- Make sure the paths to the `.root` score files are correctly set in the script, usually at the top of `plottingM4.py` and `plottingM5.py`.
- Use the same environment setup (`source setup.sh`) for both merging score files with `hadd` and running these plotting scripts to prevent segmentation faults and ensure proper library loading.


