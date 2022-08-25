# How to Install and Run The Pack

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

Then you can run `main.py`, which will run the three front-end python scripts explained in README.md. Alternatively, you can run the front-end python scripts in the following order:

1. Run the `expert_fscore.py` to get expert performance measures: using this command:
`> python expert_fscore.py`

2. Run `ml_predictions.py` to get ML models performance measures: using this command: `> python ml_predictions.py > output.txt`. It is important to save the standard output of the command in a file to feed it later to the parser to get the nice presented table of ML models' measures.

3. Run `output_parser.py` to get the ML models' performance measures in a table format without other information from the `ml_predictions.py` output. Use this command:
`> python output_parser.py output.txt`.