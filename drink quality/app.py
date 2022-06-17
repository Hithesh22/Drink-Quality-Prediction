
#import the requirements
from flask import Flask,render_template,json,jsonify,request
import pickle
import numpy as np
import sklearn


#initialize the app

app=Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        alcohol=float(request.form['alcohol'])
        sulphates=float(request.form['sulphates'])
        volatile_acidity=float(request.form['volatile_acidity'])
        #load the pickle file
        filename='drink.pickle'
        loaded_model=pickle.load(open(filename,'rb'))
        data=np.array([[volatile_acidity,sulphates,alcohol]])
        my_prediction=loaded_model.predict(data)
        #get the result template
        return render_template('index.html',pred=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
