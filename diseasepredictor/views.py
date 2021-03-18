from django.shortcuts import render
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import time
import joblib
import matplotlib.pyplot as plt




def Earthquake(request):
    def gen_features(X):
        strain = []
        #strain.append(X.mean())
        strain.append(X.std())
        strain.append(X.min())
        strain.append(X.max())
        #strain.append(X.kurtosis())
        #strain.append(X.skew())
        #strain.append(np.quantile(X,0.01))
        #strain.append(np.quantile(X,0.05))
        #strain.append(np.quantile(X,0.95))
        #strain.append(np.quantile(X,0.99))
        #strain.append(np.abs(X).min())
        strain.append(np.abs(X).max())
        #strain.append(np.abs(X).mean())
        strain.append(np.abs(X).std())
        return pd.Series(strain)
    train_df = pd.read_csv('/home/jishnusaurav/Downloads/LANL-Earthquake-Prediction/train.csv', nrows = 6000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    train = pd.read_csv('/home/jishnusaurav/Downloads/LANL-Earthquake-Prediction/train.csv', iterator = True, chunksize = 150000,dtype = {'acoustic_data': np.int16, 'time_to_failure': np.float64}) 
    X_train = pd.DataFrame()
    y_train = pd.Series()
   

    if request.method == 'POST':
        
        age = float(request.POST['age'])
        sex = float(request.POST['sex'])
        cp = float(request.POST['cp'])
        trestbps = float(request.POST['trestbps'])
        chol = float(request.POST['chol'])
        data = np.array(
            (age,
             sex,
             cp,
             trestbps,
             chol
            )
        ).reshape(1, 5)
        rand_forest=joblib.load("/home/jishnusaurav/Downloads/COVID-19-Predictor-master/COVID-19-Predictor-master/diseasepredictor/final_model.sav")
        predictions = rand_forest.predict(data)
        x=str(predictions[0])
        print(predictions[0])
        print("123")
        return render(request,
                  'heart.html',
                  {
                      'context': x
                  })
    else:
        return render(request,
                  'heart.html',
                  {
                      'context': "No data"
                  })






def breast(request):



    if request.method == 'POST':

        radius = float(request.POST['radius'])
        texture = float(request.POST['texture'])
        perimeter = float(request.POST['perimeter'])
        area = float(request.POST['area'])
        smoothness = float(request.POST['smoothness'])

        rf = RandomForestClassifier(
            n_estimators=16, criterion='entropy', max_depth=5)
        rf.fit(np.nan_to_num(X), Y)

        user_data = np.array(
            (radius,
             texture,
             perimeter,
             area,
             smoothness)
        ).reshape(1, 5)

        predictions = rf.predict(user_data)
        print(predictions)

        if int(predictions[0]) == 1:
            value = 'may have covid-19, do get tested'
        elif int(predictions[0]) == 0:
            value = "don\'t have covid-19"

    return render(request,
                  'breast.html',
                  {
                      'context': value
                  })


def home(request):

    return render(request,
                  'home.html')

# def handler404(request):
#     return render(request, '404.html', status=404)
