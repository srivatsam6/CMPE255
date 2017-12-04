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
from flask import request,session
from flask import jsonify
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask_cors import CORS
from flask import request, redirect, url_for
from werkzeug import secure_filename



app = Flask(__name__)

#CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


#UPLOAD_FOLDER = 'D:/Masters/SJSU/Semester 2/255/Project/project code/project code/saved/'
df = pd.DataFrame()
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv'])
Model = []
Accuracy = []
test_percentage = 0.2 #populate from request
options_checked = [0,1,2] #populate from request




#app.config['UPLOAD_FOLDER'] = get_upload_folder()

def get_upload_folder():
    curr_dir = os.getcwd()
    upload_dir = os.path.join(curr_dir,'saved')
    print ('directory---')
    
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    print (upload_dir)
    return upload_dir

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/value', methods=['GET', 'POST'])
def getData():
  content = request.get_json(silent=True)
  print (content)
  return json.dumps(content);

@app.route('/updatefeatures', methods=['GET', 'POST'])
def getUpdatedFeatures():
  content = request.get_json(silent=True)
  print (content)
  return json.dumps(content);
    
@app.route('/features', methods=['GET', 'POST'])
def postData():
  #content = [ { "Column": "height" }, { "Column": "weight" }, { "Column": "humidity"}, { "Column": "soilType" }, { "Column": "rainfall"}]
 # print(session['dataframe'])
  if 'dataframe' in session:
      print(session['dataframe'])
  content = []
  print('sess::',session.pop('dataframe',None))
  columnList = list(df.columns.values)
  print('columnslst:',columnList)
  col = {}
  for i in (columnList):
      col["Column"] = str(i)
      content.append(col)
  #content = ["height", "weight", "humidity", "soilType", "rainfall"]
  #return json.dumps(content);
  print(content)
  return jsonify(content)
  

@app.route('/hello')	
def hello():
    return "Hello World!"


@app.route("/uploadFile", methods=['GET', 'POST'])
def index():
    print('upload')
    
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('filename:'+filename)
            upload_folder = get_upload_folder()
            #app.config['UPLOAD_FOLDER'] = upload_folder
            #app.config['FILE_NAME'] = filename
            file.save(os.path.join(upload_folder, filename))
            df = pd.read_csv(os.getcwd()+'/saved/'+filename)
            
            print(df.describe(include = 'all'))
     #       sess = requests.session()
      #      sess.post('dataframe', df.columns.values)
            session['dataframe'] = df
            return redirect(url_for('postData'))







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
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='localhost', port=5000, debug=True)

