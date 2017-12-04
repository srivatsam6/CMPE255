import sklearn
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import json
import logging
import os
from flask import Flask,render_template
from flask import request
from flask import jsonify
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask_cors import CORS
from flask_cors import cross_origin
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper


from flask import Flask, session
#from flask.ext.session import Session
from flask_session import Session

from crossorigin import *

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})

#@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
app.config['CORS_HEADERS'] = 'Content-Type'

app.config['SECRET_KEY'] = 'my_secret_key'

# Check Configuration section for more details
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

test_percentage = 0.2 #populate from request
options_checked = [0,1,2] #populate from request

UPLOAD_FOLDER = 'F:/MS/Third Semester/255/255 group project/project code/saved/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/')
# @crossdomain(origin='*')
# def homepage():
#     return render_template('index.html')

@app.route('/set/')
@crossdomain(origin='*')
def set():
    session['fileName'] = 'covtype.csv'
    return 'ok'

@app.route('/get/')
@crossdomain(origin='*')
def get():
    return session.get('fileName', 'not set')


@app.route('/value', methods=['GET', 'POST','OPTIONS'])
#@crossdomain(origin='*')
@crossdomain(origin='*', headers='Content-Type')
#@cors.crossdomain(origin='*')
def getData():
  content = request.get_json(silent=True)
  print content
  return json.dumps(content);

@app.route('/value1', methods=['GET', 'POST','OPTIONS'])
#@crossdomain(origin='*')
@crossdomain(origin='*', headers='Content-Type')
#@cors.crossdomain(origin='*')
def getData1():
  content = request.get_json(silent=True)
  print content
  return json.dumps(content);


@app.route('/updatefeatures', methods=['POST','OPTIONS'])
#@cors.crossdomain(origin='*')
@crossdomain(origin='*',headers=['Content-Type','Authorization'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def getUpdatedFeatures():
  content = request.get_json(silent=True)
  print("hello")
  print content
  return json.dumps(content);

@app.route('/updatefeatures1', methods=['POST','OPTIONS'])
#@cors.crossdomain(origin='*')
@crossdomain(origin='*',headers=['Content-Type','Authorization'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def getUpdatedFeatures1():
  content = request.get_json(silent=True)
  print("hello")
  print content
  return json.dumps(content);

@app.route('/submittedfeatures', methods=['POST','OPTIONS'])
#@cors.crossdomain(origin='*')
@crossdomain(origin='*',headers=['Content-Type','Authorization'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def allFeatures1():
  content = request.get_json(silent=True)
  print("hello")
  print content
  return json.dumps(content);


#logging.getLogger('flask_cors').level = logging.DEBUG
    
@app.route('/features', methods=['GET', 'POST','OPTIONS'])
@crossdomain(origin='*')
def postData():
  #content = [ { "Column": "height" }, { "Column": "weight" }, { "Column": "humidity"}, { "Column": "soilType" }, { "Column": "rainfall"}]
  df = pd.read_csv('F:/MS/Third Semester/255/255 group project/project code/saved/covtype.csv')
  if 'dataframe' in session:
    print(session['dataframe'])
  content = []
  print('sess::',session.pop('dataframe',None))
  columnList = list(df.columns.values)
  print('columnslst:',columnList)
  col = {}
  return jsonify(columnList)


  return jsonify(content)
  

@app.route('/hello')	
@crossdomain(origin='*')
def hello():
    return "Hello World!"


@app.route("/uploadFile", methods=['GET', 'POST'])
@crossdomain(origin='*')
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            df = pd.read_csv('F:/MS/Third Semester/255/255 group project/project code/saved/covtype.csv')
            session['fileName'] = filename
            print("in upload file")
            print session['fileName']
            print("out upload file")
            return "successfully uploaded"




def test_split(test_percentage,dataframe):
    train,test = train_test_split(df,test_size=test_percentage,random_state=999)
    print("test_split")
    return test


def train_split(test_percentage,dataframe):
    train,test = train_test_split(df,test_size=test_percentage,random_state=999)
    print("train_split")
    return train

def feature_selection_model():
    print("feature_selection_model")
    train = train_split(test_percentage,df)
    features = train.iloc[:,0:54]
    label = train['Cover_Type']
    clf = ExtraTreesClassifier()
    clf = clf.fit(features, label)
    model = SelectFromModel(clf, prefit=True)
    return model

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

if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)


# for option in options_checked:
#     print(option)
#     if(option == 0):
#         DecisionTree()
#     elif(option == 1):
#         RandomForest()
#     else:
#         Logistic()






