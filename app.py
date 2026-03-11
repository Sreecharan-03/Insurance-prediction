# load the insurance_model.pkl and scaler.pkl files
import pickle
import numpy as np
class InsurancePremiumPredictor:
    def __init__(self):
        with open('insurance_model.pkl','rb') as f:
            self.model=pickle.load(f)
        with open('scaler.pkl','rb') as f:
            self.scaler=pickle.load(f)
    def prediction(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        input=np.array([[Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
        scaled_input=self.scaler.transform(input)
        result=self.model.predict(scaled_input)
        return result[0]
    
import streamlit as st
import pandas as pd
df=pd.read_csv('insurance.csv')
st.write(df.sample(5))
st.title('Insurance Premium Prediction')
Age=st.number_input('Age',min_value=18,max_value=100)
Annual_Income_LPA=st.number_input('Annual Income (LPA)',min_value=0.0)
Policy_Term_Years=st.number_input('Policy Term (Years)',min_value=1)
Sum_Assured_Lakhs=st.number_input('Sum Assured (Lakhs)',min_value=0.0)  
if st.button('Predict Premium'):
    predictor=InsurancePremiumPredictor()
    premium=predictor.prediction(Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs)
    st.write(f'Predicted Annual Premium (Thousands): {premium:.2f}')