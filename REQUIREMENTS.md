# Replication Pack Requirements
## Hardware Requirements
Running the replication pack needs at least 32 GB of ram and 4X 3.0 GHz Intel Scalable Processors. Ideally, this pack should be executed using GPUs to get accelerated computing and fast results. Running this pack using CPUs can take at least 4-5 hours. However, the pack will produce the result in an hour or less using GPUs. In case no result was produced, this is an indicator of running this pack using low computing resources.

## Software Requirements
The pack requires Python 3.8.12 to be installed installed (not a newer version or older one! Just to avoid dependencies compatibility issues). You also need to install `pip`, a command line program that helps install all required packages listed in `app/requirements.txt` file. Finally, you also need to install Java 11.

After having Java, Python and pip installed, you need install the required packages listed in `app/requirements.txt` file