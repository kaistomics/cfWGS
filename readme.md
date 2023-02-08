# Integrative modeling of tumor genomes and epigenomes for enhanced cancer diagnosis by cell-free DNA

-----


## Contents
- [Description](##Description)
- [Repo contents](##Repo-contents)
- [System requirements](##System-requirements)
- [Installation](##Installation)
- [Demo](##Demo)
- [Results](##Results)

-----

## Description
Multi-cancer early detection remains a key challenge in cell-free DNA (cfDNA)-based liquid biopsy. Here, we perform cfDNA whole-genome sequencing to generate two test datasets covering 2,125 patient samples of 9 cancer types and 1,241 normal control samples, and also a reference dataset for background variant filtering based on 20,529 low-depth healthy samples. An external cfDNA dataset consisting of 208 cancer and 214 normal control samples is used for additional evaluation. Unprecedented accuracy for cancer detection and tissue-of-origin localization is achieved using our algorithm, which incorporates cancer type-specific profiles of mutation distribution and chromatin organization in tumor tissues as model references. Remarkably, our integrative model detects early-stage cancers, including those of pancreatic origin, with high sensitivity that is comparable to that of late-stage detection. Model interpretation reveals the contribution of the cancer type-specific genomic and epigenomic features. Our methodologies may lay the groundwork for accurate cfDNA-based cancer diagnosis, especially at early stages. 

-----

## Repo contents
- [data](./data): example dataset.
- [model_genome.cancer_detection.py](model_genome.cancer_detection.py): Genome model training for cancer detection.
- [model_genome.too.py](model_genome.too.py): Genome model training for tissue of origin prediction.
- [model_epigenome.cancer_detection.py](model_epigenome.cancer_detection.py): Epigenome model training for cancer detection.
- [model_epigenome.too.py](model_epigenome.too.py): Epigenome model training for tissue of origin prediction.


-----

## System requirements
#### Hardware requirements
Training the genome, epigenome model require sufficient RAM and GPU acceleration. We trained our model using following specs:

RAM: 378GB
CPU: 56 cores
GPU: NVIDIA Titan Xp

#### Software requirements
The genome and epigenome model were trained on Linux operating system (Ubuntu 18.04.6 LTS).
We recommend to use miniconda to install the required packages in the virtual environment (https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh).
We trained out model in the virtual environment with python version 3.8.12

-----

## Installation
Genome, epigenome model were built on Tensorflow and Keras.
The specific version of the packages we used are listed below.
~~~
conda install -c anaconda tensorflow-gpu==2.4.1
conda install -c anaconda scikit-learn==1.0.1
conda install -c conda-forge keras-tuner==1.1.0
conda install -c conda-forge pandas
conda install -c conda-forge scikit-optimize==0.9.0
pip install tensorflow-addons==0.14.0
~~~
-----
## Demo
Constructing genome model for cancer detection and TOO prediction.
Argument 1: input file localization
Argument 2: hyperparameter searching trial (we set this value to 200 in the paper).
Argument 3: GPU number  
~~~
python model_genome.cancer_detection.py ./data/input_genome.cancer_detection.npz 11 1
python model_genome.too.py ./data/input_genome.too.npz 11 2
~~~  
Using the given data, the genome model training for cancer detection takes 22 minutes on our system.
Using the given data, the genome model training for TOO prediction takes 13 minutes on our system.


Constructing epigenome model for cancer detection and TOO prediction.
Argument 1: input file localization
Argument 2: hyperparameter searching trial (we set this value to 200 in the paper).
Argument 3: GPU number
~~~
python model_epigenome.cancer_detection.py ./data/input_epigenome.cancer_detection.npz 11 1
python model_epigenome.too.py ./data/input_epigenome.too.npz 11 2
~~~
Using the given data, the epigenome model training for cancer detection takes 16 minutes on our system.
Using the given data, the epigenome model training for TOO prediction takes 12 minutes on our system.

-----

## Results
When training is completed, final output model is created in TensorFlow SavedModel format(h5 format), which stores the model structure and weight at once.

output of "model_genome.cancer_detection.py" code
- model_genome.cancer_detection.best.h5

output of "model_genome.too.py" code
- model_genome.too.best.h5

output of "model_epigenome.cancer_detection.py" code
- model_epigenome.cancer_detection.best.h5

output of "model_epigenome.too.py" code
- model_epigenome.too.best.h5
