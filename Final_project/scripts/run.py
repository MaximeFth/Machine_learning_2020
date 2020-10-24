from proj1_helpers import *
from implementations import *
import matplotlib.pyplot as plt
import json
import sys
####################################ARGS PROCESSING ###############################################
# Argparse banned :(  used sys instead.
#import argparse
#parser = argparse.ArgumentParser(description='Process arguments')
#parser.add_argument('Method', action="store", type=str, nargs = '?',default="LR", const = "LR")
#args = parser.parse_args()
#Method = args.Method

if len(sys.argv) > 1:
	Method = str(sys.argv[1])
	print(Method)
else:
	Method = "LR"
assert Method in ['LR', 'R_LR', 'LS_SGD', 'LS_GD', 'RR', 'LS'], "method name not correct, please choose one of the following: LR, R_LR, LS_SGD, LS_GD, RR, LS"

# loading parameters
with open('parameters.json') as param:
        parameters = json.load(param)
if parameters[Method]["degree"] == 0:
	parameters[Method]["degree"] = None
####################################DATA PRE PROCESSING ############################################

print("loading data...", end = " ")
"load the data"
DATA_TRAIN_PATH = '../data/train.csv' 

idx = 0

y, tX, ids = load_csv_data(DATA_TRAIN_PATH)


"put the labels between 0 and 1"
y_std = (y+1)/2

"replace invalid value"
tX = replace(tX, -999)

tX_std, tX_mean, tX_stdev  = standardize(tX)



"remove outiers"
tX_no_outliers, y_no_outliers = remove_outliers_IQR(tX,y_std, 0.87, 0)

"standardize the features"
tX_no_outliers_std, _, _ = standardize(tX_no_outliers,tX_mean,tX_stdev)



####################################Main function ####################################################

def train_w():
	print("training using: ",parameters[Method]["f_name"])
	final_weights = train(globals()[parameters[Method]["f_name"]],y_no_outliers,tX_no_outliers_std,tX_std,y_std,seed=0,**parameters[Method])
	return final_weights

def test_w(final_weights):
	"""
	create submission by predicting the labels given on the test set by the weights
	:param weights: weights to test.
	"""
	weights, _, _ = final_weights
	#load testing data
	print("loading test data...", end = " ")
	DATA_TEST_PATH = '../data/test.csv' 
	_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

	#replacing the -999 values with the median
	tX_test = replace(tX_test, -999)
	
	#standardize the testing data wrt to the train mean and standard deviation
	tX_test_std,_,_ = standardize(tX_test, tX_mean, tX_stdev)

	#expand the test features to the same degree as the train were expanded
	if parameters[Method]["degree"] is not None:
		tX_test_std =build_poly(tX_test_std, parameters[Method]["degree"])

	#specify output path
	OUTPUT_PATH = '../data/sample_submission.csv' 

	#predict the label
	y_pred = predict_labels(np.mean(weights,axis=0), tX_test_std)

	#create submission
	create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
	print("Submission successfully created!")

          
if __name__ == '__main__':
    weights = train_w()
    test_w(weights)
    # load test data
