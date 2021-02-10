# deepTCR_PyTorch
Here, initially a revamp of DeepCAT in PyTorch and Python scripting implementing an entire pipeline, from https://github.com/s175573/DeepCAT


### Folder structure:
```
── DeepTCR_PyTorch
    ├── ./src
    |   |── `models.py`: Architectures, etc
    |   |── `plots.py` : Helper methods for plotting (AUCs, PPV, Train loss, etc)
    |   |── `preprocessing.py` : Reading sequences, AA encoding, etc.
    |   |── `train_eval_helpers.py` : Training, evaluating, batch_training, prediction methods
    |   └── `util/` : various utility methods.
    |
    ├── ./notebooks/
    |   |── `various notebooks for draft, and tests and plots
    |
    ├── ./output/
    |   |── ./run_output/ : outputs of predict_cancer_score.py
    |   |── ./training_output/ : outputs of train.py
    |   |── ./tsv_output/ : outputs of prepareAdaptiveFile.py
    |
    ├── prepareAdaptiveFile.py : Parses raw .tsv files
    ├── iSMARTm.py : taken directly from Beshnova et al's repo
    ├── predict_cancer_score.py : Loads a trained model and predicts cancer_score
    └── train.py : Trains a model from scratch
```
