# CS-433 Machine Learning Project 1
### Abstract
In 2013, the discovery of Higgs boson is acknowledged by the Nobel prize in physics. With data provided by the ATLAS experiment at CERN, Higgs classification task became an  interesting  challenge.  For  this  purpose,  several  machine learning  models  are  tested.  Our  best  model  achieves  0.8  of accuracy.
### Prerequisites

This project runs on python3 and requires as external library: numpy, matplotlib

```
pip install numpy 
pip install matplotlib
```

### Installing prerequisites

```
pip install -r requirements.txt
```
### Usage 
```
Please put train.csv and test.csv inside a data directory at same level than the script directory.
|-data
|  |--train.csv
|  |--test.csv
|-script
|  |--run.py
|  |--implementations.py
|  |--proj1_helpers.py
|  |--parameters.json

#cd scripts
#python run.py [model]
(model by default is logistic regression)

Submissin file will be created in data/


see options with
#python run.py -h
```
## Structure
The project is composed of 4 main files: 
```
Scripts:
-implementations.py: All the regressions methods implementations
-run.py: Contains the code to run the training using a wanted regression method
-parameters.json: The parameters used for the different methods
-proj1_helpers: All others helpful functions 

Data:
-train.csv: train set containing 250'000 samples of 30 features
-test.csv: test set containing 568'238 samples of 30 features
-sample-submission.csv: example of submission

```

    
### Data Process
```
- undefined values : undefined entries of value -999 replaced by the median
- outliers removal : outliers removed using IQR
- standardization : classic standardization to achieve zero mean and unit variance on the data

- label data : from initial labels of {-1, 1} to {0, 1} labels for logistic regression
```

### Training
```
- By default, logistic regression model is trained on the pre-processed train data
- Hard-coded tuned hyperparameter used from parameters.json
- Output weights of the trained model
```

### Testing
```
- Pre-processing of the test data
- Prediction using trained model written in sample_submission.csv 
```


## Authors

**Martin Cibils** -**Maxime Fellrath** -  **Yura Tak**  

