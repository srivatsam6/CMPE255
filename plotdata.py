#!/usr/bin/env python

# Install dependencies, preferably in a virtualenv:
#
#     pip install flask matplotlib
#
# Run the development server:
#
#     python app.py
#
# Go to http://localhost:5000/plot.png and see a plot of random data.
#
# based on https://gist.github.com/rduplain/1641344
 
import random
import cStringIO
 
from flask import Flask, make_response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
 
 
app = Flask(__name__)


@app.route('/')
def root() :
    return "<form action='/plot.png' method='post'><input type='text' name='data'><input type='submit' value='submit'>"

@app.route('/plot.png', methods = ['GET', 'POST'])
def plot() :
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
 
    if request.method == 'GET' :
        xs = range(100)
        ys = [random.randint(1, 50) for x in xs]
    
    if request.method == 'POST' :
        # assuming posted data looks like '3.3,5.3,6.13,33.4'
        ys = map( float, request.form['data'].strip().split(',') )
        xs = range(len(ys))

    axis.plot(xs, ys)
    canvas = FigureCanvas(fig)
    output = cStringIO.StringIO()
    canvas.print_png(output)
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response

 
if __name__ == '__main__':
    app.run(debug=True)