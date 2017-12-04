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
# from sklearn.cross_validation import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask_cors import CORS
# from flask_cors import cross_origin
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename

from flask import Flask, session
#from flask.ext.session import Session
from flask_session import Session

from crossorigin import *

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

#@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])


app.config['SECRET_KEY'] = 'my_secret_key'

# Check Configuration section for more details
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

kjdsfjsd
@app.route('/vadfjkdsflue', methods=['GET', 'POST'])
def getData():
  content = request.get_json(silent=True)
  print content
  return json.dumps(content);



if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)