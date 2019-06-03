# Data science bowl 2018
This repository contains my project folder for the [kaggle data science bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018/overview). It is coded using **pytorch** and the **fastai** library. 

## Requirements
To get an environment with all requirements to have the project running, please use anaconda and run:
```shell
conda env create -f environment.yml
```
Optionally, you can find all packages and their version in `requirements.txt`.

## Getting the data
To download the data, I recommend using the [kaggle API](https://www.kaggle.com/docs/api#interacting-with-competitions). You can then simply run:
```shell
kaggle competitions download -c data-science-bowl-2018 -p data
```
Unzip the files in folders with corresponding names (for instance, `stage1_train.zip` goes to folder `stage1_train`).

## Project structure
The project can contain up to 4 folders once you started using it:
* The `data` folder contains the dataset divided in 3 main subfolders:
    * stage1_train that contains the trainset
    * stage1_test that contains the testset for stage1 (to compare with results stored in stage1_solution.csv)
    * stage2_test_final that contains the testset for stage2 that you can still test on kaggle
* The `models` folder is empty to begin with and will be used to store models when training
* The `submissions` folder is also empty and will store the csv submission files
* The `dsbowl` folder contains the source code, organised as such:
    * `train.py` contains the script for training
    * `predict.py` contains the script to make predictions using an existing model
    * `config.py` contains the parameters for training. You can change them to make some tests.
    * `modules` contains various python files for the different modules used.
Besides, the notebook explains how the project works as a whole.
## How to use
To train, simply run:
```shell
python -m dsbowl
```
Note that the script automatically creates a submission after training. 

To make a submission, get the model filename you want to use in `models`, then run:
```shell
python -m dsbowl -p <model_name>
```
Finally, check the latest submission file in `submissions` and run:
```shell
kaggle competitions submit data-science-bowl-2018 -f submissions/<submission>.csv -m "<your message>"
```

## Notes
This is a work in progress, there are still lot of things to improve but it works fine for now.
