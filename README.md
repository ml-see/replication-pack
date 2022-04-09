# Introduction

This replication pack replicates the experimental work and reproduces the ML models and experts prediction performance matrices. It consists of two parts a front-end and backend files.

## Front-end Files

- `ml_predictions.py`: this is the main file for ML models. This file outputs information about the progress of building the ML models, their predictions, and the prediction metrics. The metrics include F-Score, AUC-ROC, and confusion matrix.

- `output_parser.py`: Since `ml_predictions.py` produces a large sum of information, this file can parse the output and produce a table that summarises the ML prediction measurements (F-Score and AUC-ROC).

- `expert_fscore.py`: this reads the expert estimates from the JOSSE data set and calculates their F-Score. It outputs a table of the projects along with the issues number and expert F-Score.

## Back-end Files

The backend files are the ones responsible for reading the data set and building the ML models. They are as the following:

- `dataset.py`: this is a class that handle and manage different data sets. It reads the data set from their folder `datasets`. If the required data set is not available in the folder, it downloads it from the data set repository.

- `se_issue_project.py`: Since data sets may contain several projects, this class represents these projects and offers tools to manipulate them during the process of training ML models.

- `se_issue_bert.py`: this class loads BERT and fine-tune it. It also predicts issues estimates using BERT fine-tuned model.

- `bert_embbedings.py`: this file contains helping functions to manipulates and store BERT embeddings to use them in the RF model

- `se_issue_rf.py`: this class build an RF model using `sklearn` implementation. It also predicts issues estimates using the RF model.

- `datasets`: this is the folder where we store the data sets.

-

# How to Install and Run The Pack

First, you need to have Python 3 installed you can follow the instructions from Python official website here [BeginnersGuide/Download - Python Wiki](https://wiki.python.org/moin/BeginnersGuide/Download). You also need to install `pip`, a command line program that helps install all required packages listed in `requirements.txt` file. You can follow the official pip installation instruction from their website here: [Installation - pip documentation v21.0 (pypa.io)](https://pip.pypa.io/en/stable/installing/).

After having python and pip installed, you can install the required packages this command:

`> pip install -r requirements.txt`

Then you can run the pack in the following order:

1. Run the `expert_fscore.py` to get expert performance measures: using this command:
`> python expert_fscore.py`

2. Run `ml_predictions.py` to get ML models performance measures: using this command: `> python ml_predictions.py > output.txt`. It is important to save the standard output of the command in a file to feed it later to the parser to get the nice presented table of ML models measures.

3. Run `output_parser.py` to get the ML models performance measures in a table format without other information from the `ml_predictions.py` output. use this command:
`> python output_parser.py output.txt`.