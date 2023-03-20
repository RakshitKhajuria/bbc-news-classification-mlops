from  src.pipeline import predict_pipeline


text=input("Enter the text")
PredictPipeline=predict_pipeline.PredictPipeline(text=text)

pred=PredictPipeline.predict()
print(pred)