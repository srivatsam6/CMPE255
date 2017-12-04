
# coding: utf-8

# In[42]:

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import sklearn
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[43]:

test_percentage = 0.2 #populate from request
options_checked = [0,1,2] #populate from request
df = pd.read_csv('covtype.csv')
#print(df.describe(include = 'all'))


# In[44]:

def test_split(test_percentage,dataframe):
    train,test = train_test_split(df,test_size=test_percentage,random_state=999)
    print("test_split")
    return test


# In[45]:

def train_split(test_percentage,dataframe):
    train,test = train_test_split(df,test_size=test_percentage,random_state=999)
    print("train_split")
    return train


# In[46]:

def feature_selection_model():
    print("feature_selection_model")
    train = train_split(test_percentage,df)
    features = train.iloc[:,0:54]
    label = train['Cover_Type']
    clf = ExtraTreesClassifier()
    clf = clf.fit(features, label)
    model = SelectFromModel(clf, prefit=True)
    return model


# In[47]:

def compute_accuracy(classifier):
    #New_features = select_train_features()
    #Test_features = select_test_features()
    test = test_split(test_percentage,df)
    train = train_split(test_percentage,df)
    
    model = feature_selection_model()
    
    New_features = model.transform(train.iloc[:,0:54])
    Test_features = model.transform(test.iloc[:,0:54])
    
    label = train['Cover_Type']
    
    fit=classifier.fit(New_features,label)
    pred=fit.predict(Test_features)
    Model.append(classifier.__class__.__name__)
    Accuracy.append(accuracy_score(test['Cover_Type'],pred))
    print('Accuracy of '+classifier.__class__.__name__ +' is '+str(accuracy_score(test['Cover_Type'],pred)))


# In[48]:

#classifiers = ["DecisionTreeClassifier","RandomForestClassifier","LogisticRegression"]
Model = []
Accuracy = []
def DecisionTree():
    print("DecisionTree")
    clf = DecisionTreeClassifier()
    compute_accuracy(classifier = clf)
    
def RandomForest():
    print("RandomForest")
    clf = RandomForestClassifier(n_estimators=200)
    compute_accuracy(classifier = clf)
    
def Logistic():
    print("Logistic")
    clf = LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200)
    compute_accuracy(classifier = clf)
    
for option in options_checked:
    print(option)
    if(option == 0):
        DecisionTree()
    elif(option == 1):
        RandomForest()
    else:
        Logistic()
    



# In[ ]:



