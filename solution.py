#github 

import sys
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)

def dataset_stat(dataset_df: pd.DataFrame):
    n_feats = dataset_df.drop(columns='target').shape[1]
    temp_sum =  sum(dataset_df['target'])
    n_class0 = len(dataset_df['target']) - temp_sum
    n_class1 = temp_sum
    return n_feats, n_class0, n_class1

def split_dataset(dataset_df: pd.DataFrame, testset_size: float):
    x_train, x_test, y_train, y_test = train_test_split(
        dataset_df.drop(columns='target'), dataset_df['target'],test_size=testset_size,
        shuffle=True
    )
    return x_train, x_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)

    acc = accuracy_score(y_test, predict)
    prec = precision_score(y_test, predict)
    recall = recall_score(y_test, predict)

    return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)

    acc = accuracy_score(y_test, predict)
    prec = precision_score(y_test, predict)
    recall = recall_score(y_test, predict)

    return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
    
    model = make_pipeline(
        StandardScaler(), 
        SVC()
    )
    model.fit(x_train, y_train)
    predict = model.predict(x_test)

    acc = accuracy_score(y_test, predict)
    prec = precision_score(y_test, predict)
    recall = recall_score(y_test, predict)

    return acc, prec, recall

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)