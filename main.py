
# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns



df = pd.read_csv('diabetes.csv')

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())


# X AND Y DATA
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# FUNCTION
def user_report():
  Pregnancies = st.sidebar.slider('Pregnancies', 0,20, 3 )
  Glucose = st.sidebar.slider('Glucose', 0,200, 120 )
  BloodPressure = st.sidebar.slider('Blood Pressure', 0,122, 70 )
  SkinThickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
  Insulin = st.sidebar.slider('Insulin', 0,846, 79 )
  BMI = st.sidebar.slider('BMI', 0,67, 20 )
  DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
  Age = st.sidebar.slider('Age', 21,88, 33 )

  user_report_data = {
      'Pregnancies':Pregnancies,
      'Glucose':Glucose,
      'BloodPressure':BloodPressure,
      'SkinThickness':SkinThickness,
      'Insulin':Insulin,
      'BMI':BMI,
      'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
      'Age':Age,
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data




# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)




# MODEL
rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)



# VISUALISATIONS
st.title('Visualised Patient Report')



# COLOR FUNCTION
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

 
 

# OUTPUT
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'You are not Diabetic'
else:
  output = 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')

st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')


# #adding a button

# if st.button('Wish Good morning'):

#     st.write('Good Morning') #displayed when the button is clicked

# else:

#     st.write('Have a great day') #displayed when the button is unclicked

# # Or even better, call Streamlit functions inside a "with" block:
 
