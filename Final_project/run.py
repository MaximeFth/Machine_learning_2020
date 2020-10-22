from proj1_helpers import *
from implementations import *
import json
import argparse

####################################ARGS PROCESSING ###############################################


parser = argparse.ArgumentParser(description='Process arguments')
parser.add_argument('Method', action="store", type=str, nargs = '?',default="LR", const = "LR")

args = parser.parse_args()
Method = args.Method
print(Method)

assert Method in ['LR', 'R_LR', 'LS_SGD', 'LS_GD', 'RR', 'LS'], "method name not correct, please choose one of the following: LR, R_LR, LS_SGD, LS_GD, RR, LS"
####################################DATA PRE PROCESSING ############################################

print("loading data...")
"load the data"
DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

"put the labels between 0 and 1"
y_std = (y+1)/2

"replace invalid value"
#tX = replace(tX, -999)

"remove outiers"
tX_no_outliers, y_no_outliers = remove_outliers_IQR(tX,y_std, 0.85, 0)

"standardize the features"
tX_no_outliers_std = standardize(tX_no_outliers)

####################################Main function ####################################################
def run():
    with open('parameters.json') as param:
        parameters = json.load(param)
    train("LR",y_no_outliers,tX_no_outliers_std,seed=0,**parameters[Method])

          
if __name__ == '__main':
    run()
