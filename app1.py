from flask import Flask, session
#from flask.ext.session import Session
from flask_session import Session

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_secret_key'

# Check Configuration section for more details
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)

@app.route('/set/')
def set():
    session['key'] = 'value'
    return 'ok'

@app.route('/get/')
def get():
    return session.get('key', 'not set')


if __name__ == '__main__':
     app.run(debug=True)









# from flask import Flask, redirect, render_template, request, session, url_for
# import pandas
# import StringIO

# app = Flask(__name__)

# #app.config['SECRET_KEY'] = 'my_secret_key'


# @app.route('/', methods=['GET', 'POST'])
# def hello():
#     if request.method == 'POST':
#         file = request.files['file']
#         if file:
#             df = pandas.io.parsers.read_csv(StringIO.StringIO(file.read()))
#             session['data'] = df.to_json()
#             return redirect(url_for('world'))
#     return '''
#     <!doctype html>
#     <title>Upload new File</title>
#     <h1>Upload new File</h1>
#     <form action="" method=post enctype=multipart/form-data>
#       <p><input type=file name=file>
#          <input type=submit value=Upload>
#     </form>
#     '''

# @app.route('/world/')
# def world():
#     df = pandas.io.json.read_json(session['data'])
#     return "Hello World"

# if __name__ == '__main__':
#     app.run(debug=True)