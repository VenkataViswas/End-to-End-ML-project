import numpy as np
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline

application = Flask(__name__)
app = application

@app.route('/')
def home():
    return render_template('index.html')  # Optional home page if you have it

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(features=pred_df)
        return render_template('predict.html', results=round(results[0], 2))
    else:
        return render_template('predict.html')

if __name__ == "__main__":
    # Optional training on startup (can be removed after first run)
    train_pipeline = TrainPipeline()
    train_pipeline.run()
    app.run(host='0.0.0.0')
