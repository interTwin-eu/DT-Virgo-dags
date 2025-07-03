# DT Virgo Use Case

This repository contains the code for the Glitchflow digital twin for simulating transient noise 
in gravitational waves interferometer.

## Requirements

- itwinai==0.3.0
- torch==2.4.1
- torchaudio==2.4.1
- torchvision==0.19.1
- torchmetrics==1.6.2
- gwpy==3.0.12
- h5py==3.13.0
- numpy==1.26.4
- matplotlib==3.10.1
- pandas==2.2.3
- scipy==1.14.1
- tensorboard==2.19.0
- mlflow==2.20.3
- ml4gw==0.7.4
- scikit-image==0.25.2
- scikit-learn==1.6.1
- tensorflow==2.19.0

For itwinai follow the official page <br>
https://itwinai.readthedocs.io/latest/installation/user_installation.html <br>

Other packages that could require the official documentation <br>
https://github.com/ML4GW/ml4gw <br>
https://www.mlflow.org/docs/latest/ml/tracking/quickstart <br>
https://www.tensorflow.org/install?hl=it <br>

## Installation and configuration

Execute setup.sh to create the directory tree in your working directory.

> chmod +x setup.sh

> ./setup.sh

Then your current directory will look like

    .
    ├── current dir/
        └── annalisarun  #Saved Annalisa Data
        └── datasets     #Processed dataset
        └── QTdatasets   #Spectrograms
        └── temp         #Data saved during training

The saveconf.yaml file contained inside the conf directory can be used to adjust the directory tree to the user setup.

The other files contained inside the conf directory define the processing pipeline parameters

- datasets.yaml <br>
Defines the locations of the datasets and their processing parameters <br>
- scan.yaml <br>
Defines the parameters for Annalisa <br>
- process.yaml <br>
Defines the qtransform during the transormation of the dataset's timeseries into spectrograms
- whiten.yaml  <br>
Defines the timeseries whitening during spectrograms creation

## Annalisa package

The Annalisa Package containes the pipeline classes for processing datasets, scanning them with the Annalisa tool, and transform them into spectrograms. <br>

- Data.py: module containing data structures and methods for data manipulations used in the pipeline
- Dataloader.py: Itwinai's classes for data loading steps
- Scanner.py: Itwinai's classes selecting channels containing glitches and producing a dataset made of spectrograms

## Glitchflow package

The Glitchflow package contains the pipeline classes for training the DT neural network, collecting metrics using mlflow and tensorboard , 
and then make inference for the trained model and generating synthetic glitches . The model is logged to mlflow.

- Data.py: module containing data structures and methods for data manipulations used in the pipeline
- Dataloader.py: Itwinai's classes for data loading steps. In particular dataset splitting and preprocessing.<br>
  During the inference step the model is retrieved from the mlflow catalogue. <br>
- Model.py: module containing the neural network definition and the metrics used during the training and inference step.
- Trainer.py: TorchTrainer class used for model training. See itwinai documentation for more details.
- Inference.py: Module containing the inference step and a class for generating a synthetic dataset

 ## Pipeline execution

 To execute a pipeline use itwinai syntax. Assuming the working directory is the same of the config.yaml file

 >itwinai exec-pipeline +pipe_key="pipeline name" +pipe_steps=[List containing the steps to execute]

if the pipe_step argument is not given the whole pipeline will be executed. For example consider the preprocessing pipeline of the config.yaml file.
Suppose you want to scan a dataset for correlated channels and the producing a spectrogram dataset, the syntax will be:

>itwinai exec-pipeline +pipe_key=preproc_pipeline +pipe_steps=[Annalisa-scan,QT-dataset]

Instead to train the model

>itwinai exec-pipeline

and here we have the pipeline definition for the training section

     training_pipeline:
      
       Trainer:
        _target_: Glitchflow.Trainer.GlitchTrainer
        #training parameters section
        num_epochs: 100
        acc_freq: 1 # accuracy logging frequency
        config: 
         #passed to itwinai configuration class 
         _target_: itwinai.torch.config.TrainingConfiguration 
         batch_size: 10
         #optimizer parameters
         optim_lr: 1.0e-4
         optim_momentum: 0.9
         optim_weight_decay: 1e-4
         #scheduler parameters
         cschd_mode: 'min'
         cschd_patience: 7
         cschd_min_lr: 1e-7
         cschd_verbose: Yes 
         cschd_factor: 0.5
        #logging section
        tensorboard_root: '/home'
        logger: 
         _target_: itwinai.loggers.MLFlowLogger
         experiment_name: 
         log_freq: 'batch'
         tracking_uri: 'http://localhost:5005'

Parameters  are defined using the syntax of a yaml file. The _target_ expression is used when you need to pass a python class.

## Logging

The application uses Mlflow and Tensorboard for logging. For installation refers to the official documentation. <br>
In case of a local setup the python installation should be enough. To launch mlflow 

> mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host ip adress --port 5005

- --backend-store-uri: the database that will be used to store data. sqlite is the default db but other databases can  be used.
- --default-artifact-root: mlflow directory where data will be stored. Logged model will be found here
- --host: the ip adress of the server. Put it to 0.0.0.0 if you work behind a proxy.
- --port: 5005 is the default port.

To launch tensorboard

> tensorboard --logdir logdir --host ip --port 6000

- --logdir: tensorboard root directory.
- --host: the ip adress of the server. Put it to 0.0.0.0 if you work behind a proxy.
- --port: 6000 is the default port.














