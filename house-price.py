import xgboost
import streamlit as slt
import numpy as np
import pickle

slt.title('California house price prediction by XGBoost Regressor')
user = slt.text_area('Enter the housing configuration')
if slt.button('Predict the house price'):
    with open('Booster.pkl','rb') as file:
        XGB = pickle.load(file)
    twod_arr = np.array([user.split(',')],dtype=float)
    prediction = XGB.predict(twod_arr)
    slt.header(prediction)
