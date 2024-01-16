#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self,columns):
        self.scaler = StandardScaler()
        self.columns = columns


    def fit(self, X):
        self.scaler.fit(X[self.columns])
        return self
    
    def transform(self, X):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis = 1) [init_col_order]
    
    
class absenteeism_model():
        
        def __init__(self, model_file, scaler_file):
            with open("model","rb") as model_file, open("scaler", "rb") as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
                
        def load_and_clean_data(self, data_file):
            df = pd.read_csv(data_file, delimiter=",")
            # store original data for later use
            self.df_with_predictions = df.copy()

            df = df.drop(["ID"], axis=1)
            df["Absenteeism Time in Hours"] = "NaN"
            
            # 4 NEW COLUMNS: Reason type columns
            df['Reason_1'] = ((df['Reason for Absence'] >= 1) & (df['Reason for Absence'] <= 14)).astype(int)
            df['Reason_2'] = ((df['Reason for Absence'] >= 15) & (df['Reason for Absence'] <= 17)).astype(int)
            df['Reason_3'] = ((df['Reason for Absence'] >= 18) & (df['Reason for Absence'] <= 21)).astype(int)
            df['Reason_4'] = ((df['Reason for Absence'] >= 22) & (df['Reason for Absence'] <= 28)).astype(int)

            # Display the updated DataFrame
            new_columns_order = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4'] + [col for col in df.columns if col not in ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']]

            df = df[new_columns_order]

            df = df.drop(columns=["Reason for Absence"])
            
            # DATE COLUMN CHANGES
            
            # Convert 'Date' column to datetime with day/month/year format
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

            # Extract month value and day of the week
            df['Month Value'] = df['Date'].dt.month
            df['Day of the Week'] = df['Date'].dt.dayofweek

            date_column_index = df.columns.get_loc("Date")
            # 3 new columns are created, I am getting all columns - 3 and placing the new ones in the correct order, that's why the -2 are used in the below code at the end.
            new_columns_order = list(df.columns[:date_column_index + 1]) + ["Month Value", "Day of the Week"] + list(df.columns[date_column_index + 1:-2])
            df = df[new_columns_order]

            # Drop the 'Date' column
            df.drop('Date', axis=1, inplace=True)
            
            # EDUCATION COLUMN CHANGE (Making binary)
            df["Education"] = (df["Education"] > 1).astype(int)
            
            # Other changes that won't matter in dataframe anymore
            df = df.fillna(value=0)
            df = df.drop(["Absenteeism Time in Hours"], axis=1)
            df = df.drop(["Day of the Week", "Daily Work Load Average", "Distance to Work"], axis=1)
            self.preprocessed_data = df.copy()
            self.data = self.scaler.transform(df)
            
            
        def predicted_probability(self):
            if (self.data is not None):
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
            
            
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
            
            
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data["Probability"] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data["Prediction"] = self.reg.predict(self.data)
                return self.preprocessed_data


# In[ ]:




