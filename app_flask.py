from flask import Flask , request , jsonify,render_template,url_for
from src.exception import CustomException
import json
import sys
from  src.pipeline import predict_pipeline
import string
from src.exception import CustomException
import pandas as pd
import altair as alt


     
app = Flask(__name__)

@app.route('/', )
def start():
    return render_template('home.html')

@app.route('/submit')
def predicthome():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

    
@app.route('/submit', methods=['POST','GET'])
def test():
    try:
        
        raw_text = request.form['news']
        PredictPipeline=predict_pipeline.PredictPipeline(raw_text)
        pred,probability=PredictPipeline.predict()
        
        return render_template('predict.html', results=pred,tables=[probability.to_html(classes='data',index=False, header=False)])
    
    except Exception as e:
        raise CustomException(e,sys)
        
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=int("3000"),debug =True)