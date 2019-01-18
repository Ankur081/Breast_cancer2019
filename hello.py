#!/usr/bin/env python3

import pandas as pd
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import cgi,cgitb
cgitb.enable()
print("content-type:text/html")
print(" ")

web=cgi.FieldStorage()
a=web.getvalue("manopause")
b=web.getvalue("tumor-size")
c=web.getvalue("inv-nodes")

#reading csv file and converting into dataframe
df=pd.read_csv('breast-cancer_csv.csv')
brstcancer_target=df['irradiat']
brstcancer_data=[df.coloumns[:-1]]
split_data=train_test_split(brstcancer_target,brstcancer_data,test_size=0.1)
train_data,test_data,train_target,test_target=split_data

#descision tree classfication aldo
dsc_algo=DecisionTreeClassifier()
trained_dsc=dsc_algo.fit(train_data,train_target)
output_dsc=trained_dsc.predict([1,2,3])

acc_dsc=accuracy_score(test_target,output_dsc)
print("Accuracy score using Decision tree classifier:",acc_dsc)


