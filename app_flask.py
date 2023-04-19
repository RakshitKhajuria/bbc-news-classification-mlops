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
    return render_template('index.html')

@app.route('/reset', methods=['POST','GET'])
def reset():
    try:
        return render_template('index.html')
    
    except Exception as e:
        raise CustomException(e,sys)

    
@app.route('/submit', methods=['POST','GET'])
def test():
    try:
        
        raw_text = request.form['news']
        PredictPipeline=predict_pipeline.PredictPipeline(raw_text)
        pred,probability=PredictPipeline.predict()
        fig = alt.Chart(probability).mark_bar().encode(x='Category',y='Probability', color='Category')
        
        return render_template('index.html', results="RESULT - "+pred,tables=probability.to_html(classes="table table-striped",index=False), chart=fig.to_json())
    
    except Exception as e:
        raise CustomException(e,sys)
        
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)