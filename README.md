# deepTCR_PyTorch
Here, initially a revamp of DeepCAT in PyTorch and Python scripting implementing an entire pipeline, from https://github.com/s175573/DeepCAT

For everything to run, install the environment using the .yml file with  ```conda env create -f deeptcr.yml``` 

Training from scratch : Check all available options using ```python train.py -h``` in the terminal
An example of execution :
```python train.py -lr 0.0001 -nb_epochs 300 -outdir 'my_directory' -keys 12 13 14 -valmode naive -ratio 0.25 -test True -plots True -arch deepcat``` will run train the model with sequence lengths 12, 13 and 14 for 300 epochs with learning rate 0.0001, then evaluate the model on the test set and save plots, and all results & weights in a directory `output/training_output/my_directory`

```python predict_cancer_score.py -indir ./SampleData/ -outdir 'my_results/' -weightdir 'output/training_output/my_directory'``` will load the weights located in my_directory, and then run the model and evaluate patient cancer scores located in the folder ./SampleData/


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
