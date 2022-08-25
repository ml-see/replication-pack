
# Introduction

This replication pack replicates the experimental work and reproduces the ML models and experts' prediction performance matrices. It consists of two parts front-end and backend files.


## Front-end Files

- `ml_predictions.py`: this is the main file for ML models. This file outputs information about the progress of building the ML models, their predictions, and the prediction metrics. The metrics include F-Score, AUC-ROC, and confusion matrix.

- `output_parser.py`: Since `ml_predictions.py` produces a large sum of information, this file can parse the output and produce a table that summarises the ML prediction measurements (F-Score and AUC-ROC).

- `expert_fscore.py`: this reads the expert estimates from the JOSSE data set and calculates their F-Score. It outputs a table of the projects along with the issues number and expert F-Score.

- `main.py`: this is a python file that runs the front-end files listed above in the correct order, so you can have a single python command to get all the experiment results.

## Back-end Files

The backend files are responsible for reading the data set and building the ML models. They are as the following:

- `dataset.py`: This class handles and manages different data sets. It reads the data set from their folder `datasets`. If the required data set is not available in the folder, it downloads it from the data set repository.

- `se_issue_project.py`: Since data sets may contain several projects, this class represents these projects and offers tools to manipulate them while training ML models.

- `se_issue_bert.py`: this class loads BERT and fine-tunes it. It also predicts issue estimates using BERT fine-tuned model.

- `bert_embbedings.py`: this file contains helping functions to manipulate and store BERT embeddings to use them in the RF model

- `se_issue_rf.py`: this class build an RF model using `sklearn` implementation. It also predicts issues' estimates using the RF model.

- `datasets`: this is the folder where we store the data sets.


# How to Obtain the Replication Pack
The replication pack is publicly accessible using its public Github repository. The Github repository URL is: 
https://github.com/ml-see/replication-pack

You also can get an archived version of the replication pack that is stored in Zenodo using it DOI: 10.5281/zenodo.6427092


# How to Run The Pack

## Option 1: Using Docker
`Dockerfile` was created to ease building the docker container image and running the container. You need to install the Docker first, and then you need to do the following steps:
1. Build the container image from the `Dockerfile`. First, unzip the replication pack, enter the replication pack and then build the image. Use the following commands to do so:
`> unzip ml_replication.zip`
`> cd ml_replication`
`> docker build --tag ml-replication .`

2. After building the container image, you can run the container using the following command:
`> docker run ml-replication`

Note: `Dockerfile` uses `main.py` to run all the front-end files.

## Option2: Using Manual Python Installation
First, you need to have Python 3.8.12 installed (not a newer version or older one! Just to avoid dependencies compatibility issues). You can follow the instructions from Python official website here [BeginnersGuide/Download - Python Wiki](https://wiki.python.org/moin/BeginnersGuide/Download). You also need to install `pip`, a command line program that helps install all required packages listed in `requirements.txt` file. You can follow the official pip installation instruction from their website here: [Installation - pip documentation v21.0 (pypa.io)](https://pip.pypa.io/en/stable/installing/). You also need to install Java 11.

After having Java, Python and pip installed, you can install the required packages with this command:

`> pip install -r requirements.txt`

Then you need to download NLTK stopwords using the following command:
`python -m nltk.downloader stopwords`

Then you can run `main.py`, which will run the three front-end python scripts explained above. Alternatively, you can run the front-end python scripts in the following order:

1. Run the `expert_fscore.py` to get expert performance measures: using this command:
`> python expert_fscore.py`

2. Run `ml_predictions.py` to get ML models performance measures: using this command: `> python ml_predictions.py > output.txt`. It is important to save the standard output of the command in a file to feed it later to the parser to get the nice presented table of ML models' measures.

3. Run `output_parser.py` to get the ML models' performance measures in a table format without other information from the `ml_predictions.py` output. Use this command:
`> python output_parser.py output.txt`.