# UECAD-Public
Public repository for Unsupervised Electrical Consumption Anomaly Detection

First, download the following file and place it within the repository: https://drive.google.com/file/d/1fN1UJrVborJgwnQ42yDcwVqmpOIKYR8-/view?usp=drive_link

You need to run the notebook Make-dataset.ipynb once during the first time you use this repository.

While running experiments, you can change latent_dim directly from the script aswell as the models and buildings to concider. For specific model options (like the enabling of LSTM layers) you can check the default options in get_model.py and change them directly from there.

After running an experiment, execute its appropriate cell in get_metrics.ipynb. The final results will be stored under Experiment_#/result/#.csv
