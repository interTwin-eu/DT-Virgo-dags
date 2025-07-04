# Index
* [DT Virgo Use Case](#dt-virgo-use-case)
    * [The Training DT Subsystem](#the-training-dt-subsystem)
    * [The Inference DT Subsystem](#the-inference-dt-subsystem)
* [Technical documentation](#technical-documentation)
    * [Requirements](#requirements)
    * [Installation and configuration](#installation-and-configuration)
    * [ANNALISA module](#annalisa-module)
    * [Glitchflow module](#glitchflow-module)
    * [Pipeline execution](#pipeline-execution)
    * [Logging](#logging)
 
# DT Virgo Use Case

The sensitivity of Gravitational Wave (GW) interferometers is limited by noise. We have been using Generative Neural Networks (GenNNs) to produce a Digital Twin (DT) of the Virgo interferometer to realistically simulate transient noise in the detector. We have used the GenNN-based DT to generate synthetic strain data (a channel that measures the deformation induced by the passage of a gravitational wave). Furthermore, the detector is equipped with sensors that monitor the status of the detector’s subsystems as well as the environmental conditions (wind, temperature, seismic motions) and whose output is saved in the so-called auxiliary channels. Therefore, in a second phase, also from the perspective of the Einstein Telescope, we will use the trained model to characterise the noise and optimise the use of auxiliary channels in vetoing and denoising the signal in low-latency searches, i.e., those data analysis pipelines that search for transient astrophysical signals in almost real time. This will allow the low-latency searches (not part of the DT) to send out more reliable triggers to observatories for multi-messenger astronomy.	
Figure 1 shows the high-level architecture of the DT. Data streams from auxiliary channels are used to find the transfer function of the system producing non-linear noise in the detector output. The output function compares the simulated and the real signals in order to issue a veto decision (to further process incoming data in low-latency searches) or to remove the noise contribution from the real signal (denoising).

<p align="center">
  <img src="https://github.com/user-attachments/assets/8a191145-b771-4ee1-9ba0-a687301e48c2" alt="High-level architecture of the DT">
  <br>
  Figure 1: High-level architecture of the DT.
</p>

Figure 2 shows the System Context diagram of the DT for the veto and denoising pipeline. 
Two main subsystems characterise the DT architecture: the Training DT subsystem and the Inference DT subsystem. The Training DT subsystem is responsible for the periodical re-training of the DT model on a buffered subsample of the most recent Virgo data. The DT model needs to be updated to reflect the current status of the interferometer, so continuous retraining of the GenNN needs to be carried out. tThe Inference DT subsystem is responsible for the low-latency vetoing and denoising of the detector’s datastream.
All modules within both subsystems are implemented as itwinai plugins. Itwinai offers several key features that are beneficial to the DT, including distributed training capabilities, a robust logging and model catalog system, enhanced code reusability, and a user-friendly configuration interface for pipelines.

<p align="center">
  <img src="https://github.com/user-attachments/assets/968a9d39-ceda-4106-adff-2729dcea3d0e" alt="System context diagram of the DT.">
  <br>
  Figure 2: System context diagram of the DT.
</p>

## The Training DT Subsystem

Operators initiate the **Training Subsystem**. The ANNALISA module first selects relevant channels for network training by analyzing time-frequency data (Q-Transform) to find correlations measured as coincident spikes in signal energy above a threshold.

After this initial step, operators preprocess data retrieved from the Virgo Data Lake. ANNALISA handles this preprocessing, which includes data resampling, whitening, spectrogram generation, image cropping, and loading into a custom PyTorch dataloader. This dataloader then feeds a Generative Neural Network (GenNN) during training.

The chosen neural network is a **Convolutional U-net**, featuring residual blocks and attention gates with enhanced skip connections for better data complexity capture. Other architectures are available for the user to experiment. GlitchFlow manages both the model definition and training. As the model trains, its weights and performance metrics are systematically logged into a dedicated model registry on MLFlow, making it accessible for the Inference Subsystem. Itwinai facilitates this logging and offloading to computing infrastructure during training.


## The Inference DT Subsystem

Users, typically GW detector characterization or data analysis experts, activate the **Inference Subsystem**. They start by selecting the data for analysis, which then undergoes the same preprocessing steps as those applied during the training phase. Subsequently, a trained model is loaded from the model catalog and utilized to perform inference on the chosen data.

The output of this process comprises "clean" data, ideally free of glitches, and metadata containing veto flagging information, which identifies glitch instances. Both the cleaned data and metadata are logged, offering a complete record of the denoising and vetoing operations.

Logged details, including images of the real, generated, and cleaned data, are accessible on **TensorBoard**. Metadata containing veto flag information, organized by the GPS time of the analyzed data, is also logged. Furthermore, metadata for any data that failed to be cleaned is recorded, including the area and Signal-to-Noise Ratio (SNR) of glitches still visible after cleaning. To access this information, users can launch TensorBoard and navigate through the logged events, which are categorized by run and timestamp, allowing for detailed visualization and analysis of the inference results. The entire pipeline, encompassing data selection, inference, and logging, is configurable via a YAML file, enabling users to specify modules to execute, preprocessing parameters, dataset specifics, network architecture, and paths for saving results.

# Technical documentation
The following shows how to set up and run the  Virgo DT pipeline.

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
Defines the qtransform during the transformation of the dataset's timeseries into spectrograms
- whiten.yaml  <br>
Defines the timeseries whitening during spectrograms creation

## ANNALISA module

The ANNALISA module containes the pipeline classes for processing datasets and compute correlations for channel selection. The module includes:<br>

- Data.py: package containing data structures and methods for data preprocessing used in the pipeline such as:
    - The TFrame class used for reading pytorch tensors and relative metadata during the pipeline workflow
    - Method for reading and processing gw data
    - Methods and classes for working with different data format like yaml and json
    - Methods used for preprocessing the dataset before model training
    - Various custom matplotlib plotting functions 

- Dataloader.py: Itwinai's classes for data loading steps. It provides:
    - Processing of gw data
    - Dataset splitting and preprocessing before training
    - Loading data for inference
    - Spectrogram dataset visualization utility.

- Scanner.py: Itwinai's class selecting relevant channels for network training by analyzing time-frequency data (Q-Transform) to find correlations measured as  coincident spikes in signal energy above a threshold. Parameters can be defined via scan.yaml file. Results are stored locally, path can be configured by user.
- Spectrogram.py: Itwinai's class for transforming a dataset of timeseries into a dataset of spectrograms via Q-transform.  Parameters can be defined via process.yaml file for the Q-transform and for the whitening of data the whiten.yaml file is read. Results are stored locally, path can be configured by user.

 

## GlitchFlow module

The GlitchFlow module contains the pipeline classes for training the DT's Neural Network, collecting metrics using MLflow and TensorBoard, making inferences with the trained model, and generating synthetic glitches. The model is logged to MLflow. The module contains:

- Data.py: same as for ANNALISA
- Dataloader.py: Itwinai's classes for data loading steps. In particular dataset splitting and preprocessing.<br>
  During the inference step the model is retrieved from the MLFlow catalogue. <br>
- Model.py: package containing the Neural Network architecture definition and the metrics used during the training and inference step.
- Trainer.py: TorchTrainer class used for model training. See itwinai documentation for more details https://itwinai.readthedocs.io/latest/how-it-works/training/training.html#itwinai-torchtrainer.
- Inference.py: package containing the inference step and a class for generating a synthetic dataset.

 ## Pipeline execution

 To execute a pipeline use itwinai syntax. Assuming the working directory is the same of the config.yaml file

 >itwinai exec-pipeline +pipe_key="pipeline name" +pipe_steps=[List containing the steps to execute]

if the pipe_step argument is not given the whole pipeline will be executed. For example consider the preprocessing pipeline of the config.yaml file.
Suppose you want to preprocess  a dataset, search for correlated channels and then producing a spectrogram dataset, the syntax will be:

>itwinai exec-pipeline +pipe_key=preproc_pipeline 

where preproc_pipeline is the pipeline's name. Itwinai will execute the following steps, see the config.yaml file:

- The preprocessing step named Data-processor
- A scan with Annalisa, the step named Annalisa-scan
- The spectrogram production step named QT-dataset

Instead suppose the dataset has been been already preprocessed, you need only the latter two steps. So type 

> itwinai exec-pipeline +pipe_key=preproc_pipeline +pipe_steps=[Annalisa-scan,QT-dataset]

where we have provided a list of steps for execution.

For the training of the model the syntax is simplified

>itwinai exec-pipeline

because the pipeline named training_pipeline is considered the default. 

Now we analyze how a pipeline written inside the config.yaml is configured.
We consider the training pipeline.



     training_pipeline:
     _target_: itwinai.pipeline.Pipeline

       Splitter:
        _target_: Glitchflow.Dataloader.QTDatasetSplitter
        train_proportion: 0.9
        images_dataset: '/home/mydataset.pt'

       Processor:  
        _target_: Glitchflow.Dataloader.QTProcessor 
        
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

         

Pipelines are defined using the yaml markup language. The first two lines are 

    training_pipeline:
     _target_: itwinai.pipeline.Pipeline

in the first line we define the name of the pipeline that will be used by the command line . 
The second line ,the target _directive_ is standard and we will always use it to define the pipeline as an itwinai object.
The _target_ directive defines a python object in general.  <br>

Now we define the first step of the pipeline:

    Splitter:
      _target_: Glitchflow.Dataloader.QTDatasetSplitter
      train_proportion: 0.9
      images_dataset: '/home/mydataset.pt'

The first two lines follows the same logic used to define the pipeline. The name followed by the _target_ directive wich indicates
a class of the Glitchflow module. QTDatasetSplitter will split data into the train and test subsets and pass them to the next step.

Next we define some parameters using the yaml syntax

      train_proportion: 0.9
      images_dataset: '/home/mydataset.pt'
We have defined the the train proportion of the dataset and its path. These parameters will be passed to the class constructor.
The second step is defined with the same logic.

      Processor:  
       _target_: Glitchflow.Dataloader.QTProcessor 

 The training step has more parameters. Some of them are simple scalars

     Trainer:
        _target_: Glitchflow.Trainer.GlitchTrainer
        #training parameters section
        num_epochs: 100
        acc_freq: 1 # accuracy logging frequency
but others are python classes. We configure it  with _target_ like in the previous steps

     config: 
         #passed to itwinai configuration class 
         _target_: itwinai.torch.config.TrainingConfiguration 
         batch_size: 10
         #optimizer parameters
         optim_lr: 1.0e-4
         optim_momentum: 0.9
all the parameters in the config section will be passed to the itwinai TrainingConfiguration class.
         
 




## Logging

### MLFlow

The application uses MLFlow and TensorBoard for logging. For installation refers to the official documentation. <br>
In case of a local setup the python installation should be enough. To launch MLFlow:

> mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host ip adress --port 5005

- --backend-store-uri: the database that will be used to store data. sqlite is the default db but other databases can  be used.
- --default-artifact-root: MLFlow directory where data will be stored. Logged model will be found here
- --host: the ip adress of the server. Put it to 0.0.0.0 if you work behind a proxy.
- --port: 5005 is the default port.

### TensorBoard

To launch TensorBoard:

> tensorboard --logdir logdir --host ip --port 6000

- --logdir: tensorboard root directory.
- --host: the ip adress of the server. Put it to 0.0.0.0 if you work behind a proxy.
- --port: 6000 is the default port.

User can launch TensorBoard and navigate through the logged events, which are categorized by run and timestamp, allowing for detailed visualization and analysis of the inference results comprising of:
-  images of the real, generated, and cleaned data
-  Metadata containing veto flag information, organized by the GPS time of the analyzed data
-  metadata for any data that failed to be cleaned is recorded, including the area and Signal-to-Noise Ratio (SNR) of glitches still visible after cleaning
-  accuracy metrics for both denosing and vetoing task

Examples are reported in the figures below:

<p align="center">
  <img src="https://github.com/user-attachments/assets/13b645f5-1ce8-415d-8eb0-f9ca45d70eeb" alt="Training accuracy">
  <br>
  Figure 3: Training accuracy as a function of learning epochs for different values for fixed $SNR^2$ threshold.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/a7d37a27-4f58-47a3-8635-7ee422c29b1f" alt="Denoising inference">
  <br>
  Figure 4:On the left: Denoising inference. Real, generated spectrograms of data used for inference and their absolute difference.
On the center and right: Denoising and vetoing accuracy as a function of different $SNR^2$ threshold after training.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/ed0d994e-0903-48de-ab5e-cdee1978ad77" alt="Training loss">
  <br>
  Figure 5: Training and validation loss as a funciton of learning epochs.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/884447df-25b0-41bc-8fb7-4f616e670581" alt="Veto metadata">
  <br>
  Figure 6: Denoising metadata. In the table are reported the gps time of the data used for inference, a binary flag to indicate if the data was succesfully cleaned, the maximum $SNR^2$ for uncleaned data, and the area (in pixels) of residual glitch after (failed) cleaning.
</p>













