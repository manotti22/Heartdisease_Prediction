from flask import Flask, request, render_template,url_for
import pickle
import numpy as np
import pandas as pd
from loggerMain import HeartFailurePredictLogger
from MongoDbManagement import MongoDbUtils


app = Flask( __name__)
pickle_reg = pickle.load(open('Cholest_reg.pkl','rb'))
pickle_model = pickle.load(open('Class_Heart_predict.pkl','rb'))
@app.route('/')
def index():
    try:
        log = HeartFailurePredictLogger.ineuron_scrap_logger()
        log.info("Index initialization successfull")
        return render_template('index.html')
    except Exception as e:
        log.exception(" Something went wrong on initiation process")


@app.route('/single_classification',methods=['POST',"GET"])
def single_classification():
    try:
        log = HeartFailurePredictLogger.ineuron_scrap_logger()
        log.info("single_classification initialization successfull")
        return render_template('single_classification.html')
    except Exception as e:
        log.exception(" Something went wrong on single_classification process")


@app.route('/predict_classification',methods=['POST',"GET"])
def predict_classification():
    try:
        log = HeartFailurePredictLogger.ineuron_scrap_logger()
        Age = request.form.get('Age')
        Sex = request.form.get('Sex')
        ChestPainType = request.form.get('ChestPainType')
        RestingBP = request.form.get('RestingBP')
        Cholesterol = request.form.get('Cholesterol')
        FastingBS = request.form.get('FastingBS')
        RestingECG = request.form.get('RestingECG')
        MaxHR = request.form.get('MaxHR')
        ExerciseAngina = request.form.get('ExerciseAngina')
        Oldpeak = request.form.get('Oldpeak')
        ST_Slope= request.form.get('ST_Slope')
        dict_test = {'Age': Age, 'Sex': Sex, 'ChestPainType': ChestPainType, 'RestingBP': RestingBP,
                     'MaxHR':MaxHR, 'Cholesterol': Cholesterol, 'FastingBS': FastingBS,
                     'RestingECG': RestingECG, 'ExerciseAngina': ExerciseAngina, 'Oldpeak': Oldpeak,
                     'ST_Slope': ST_Slope }


        input = np.array(list(dict_test.values())).reshape(1, 11)
        predict = pickle_model.predict(input)[0]
        log.info("Pridication successfull with value",predict)
        if predict == 1:
            return render_template('heart Failure.html')
        else:
            return render_template('not_heart Failure.html')
        return render_template('predict_classification.html')
    except Exception as e:
        log.exception(" Something went wrong on predict_classification process")


@app.route('/batch_classification',methods=['POST',"GET"])
def batch_classification():
    try:
        log = HeartFailurePredictLogger.ineuron_scrap_logger()
        log.info("batch_classification initialization successfull")
        db_obj = MongoDbUtils()
        if db_obj.isDatabasePresent('batch_data'):
            if db_obj.isCollectionPresent('classification_batch'):
                response = db_obj.getRecords('classification_batch')
                if response is not None:
                    batch = [i for i in response]
                    log.info("db batch_classification initialization successfull")
                    batch_reg = pd.DataFrame(batch)
                    test_data = batch_reg.drop(columns='_id')
                    test_data.to_html("class_batch.html")
                    data = pickle_model.predict(test_data.values)
                    log.info("Batch Pridiction successfull",)
                    return render_template('batch_classification.html', data=data)
        return render_template('single_classification.html')
    except Exception as e:
        log.exception(" Something went wrong on batch_classification process")


@app.route('/single_regression',methods=['POST',"GET"])
def single_regression():
    try:
        log = HeartFailurePredictLogger.ineuron_scrap_logger()
        log.info("single_regression initialization successfull")
        return render_template('single_prediction.html')
    except Exception as e:
        log.exception(" Something went wrong on single_regression process")


@app.route('/predict',methods=['POST',"GET"])
def predict():
    try:
        log = HeartFailurePredictLogger.ineuron_scrap_logger()
        Age = request.form.get('Age')
        Sex = request.form.get('Sex')
        ChestPainType = request.form.get('ChestPainType')
        RestingBP = request.form.get('RestingBP')
        Cholesterol = request.form.get('Cholesterol')
        FastingBS = request.form.get('FastingBS')
        RestingECG = request.form.get('RestingECG')
        ExerciseAngina = request.form.get('ExerciseAngina')
        Oldpeak = request.form.get('Oldpeak')
        ST_Slope = request.form.get('ST_Slope')
        Disease = request.form.get('Disease')
        dict_test = {'Age': Age, 'Sex': Sex, 'ChestPainType': ChestPainType, 'RestingBP': RestingBP, 'Cholesterol': Cholesterol, 'FastingBS': FastingBS,
                     'RestingECG':RestingECG , 'ExerciseAngina': ExerciseAngina , 'Oldpeak': Oldpeak, 'ST_Slope':ST_Slope, 'Disease':Disease}

        input = np.array(list(dict_test.values())).reshape(1, 11)
        predict = pickle_reg.predict(input)[0]
        log.info("predict initialization successfull with value",predict)
        return str(predict)
    except Exception as e:
        log.exception(" Something went wrong on predict process")

@app.route('/batch_regression',methods=['POST',"GET"])
def batch_regression():
    try:
        log = HeartFailurePredictLogger.ineuron_scrap_logger()
        db_obj = MongoDbUtils()
        if db_obj.isDatabasePresent('batch_data'):
            if db_obj.isCollectionPresent('regression_batch'):
                response = db_obj.getRecords('regression_batch')
                if response is not None:
                    log.info("db batch_regression initialization successfull")
                    batch = [i for i in response]
                    batch_reg = pd.DataFrame(batch)
                    test_data = batch_reg.drop(columns='_id')
                    test_data.to_html("reg_batch.html")
                    data = pickle_reg.predict(test_data.values)
                    log.info("batch_regression successfull", )
                    return render_template('batch_regression.html', data=data)
    except Exception as e:
        log.exception(" Something went wrong on batch_regression process")

if __name__ == "__main__":
    app.run(debug = True)