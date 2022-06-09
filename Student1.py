import numpy as np  #For handling numeric
import pandas as pd  #For handling data
import matplotlib.pyplot as plt #For visualisation
import seaborn as sb

df = pd.read_csv('StudentsPerformance.csv')  #read_csv makes it available


#Since the column names are large and with spaces, we shall rename as follows
df.columns = ['gender', 'race', 'parentDegree', 'lunch', 'course', 'mathScore', 'readingScore', 'writingScore']
df['TotalScore']=df['mathScore']+df['readingScore']+df['writingScore']


en_data=df
en_data['gender'].replace(['male','female'],
                        [0,1], inplace=True)
en_data['lunch'].replace(['standard','free/reduced'],
                        [0,1], inplace=True)
en_data['course'].replace(['none','completed'],
                        [0,1], inplace=True)
en_data['race'].replace(['group A','group B','group C','group D','group E'],
                        [0, 1,2,3,4], inplace=True)
en_data['parentDegree'].replace(["bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school', 'some high school'],
                        [0, 1,2,3,4,5], inplace=True)


Y = en_data['TotalScore']
X = en_data[['mathScore','readingScore','writingScore']]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#Multiple Regression
from sklearn.linear_model import LinearRegression
# creating an object of LinearRegression class
LR = LinearRegression()
# fitting the training data
LR.fit(X_train,Y_train)



#"""Multiple Regression Accuracy"""

import sklearn.metrics as sm
Y_Pred3 = LR.predict(X_test)
#print(Y_Pred3)
#rerror=sm.r2_score(Y_test, Y_Pred3)
#print("R Square Error of regression : ",rerror)

#"""An R2 of 1.0 indicates that the data perfectly fit the linear model."""

#plt.scatter(Y_test,Y_Pred3);
#plt.xlabel('Actual');
#plt.ylabel('Predicted');



#"""Thus, the Exploratory Analysis on Dataset and its results are presented."""

#sb.regplot(x=Y_test,y=Y_Pred3,ci=None,color ='red');

#"""STREAMLIT FOR WEB APP"""

import joblib
joblib.dump(LR,'studentPerf_model.pkl')

import pickle
import streamlit as st

# loading the trained model
pickle_in = open('studentPerf_model.pkl', 'rb')
classifier = joblib.load(pickle_in)


@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction(ReadScore, WriteScore, MathScore):
    # Pre-processing user input

    # Making predictions
    prediction = classifier.predict([[ReadScore, WriteScore, MathScore]])

    return prediction


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:grey;padding:13px"> 
    <h1 style ="color:black;text-align:center;">STUDENT PERFORMANCE PREDICTION</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # following lines create boxes in which user can enter data required to make prediction
    ReadScore = st.number_input("Reading Score",min_value=0, max_value=100,format="%d")
    WriteScore = st.number_input("Writing Score Score",min_value=0, max_value=100,format="%d")
    MathScore = st.number_input("Maths Score",min_value=0, max_value=100,format="%d")

    result = ""

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        result = prediction(ReadScore, WriteScore, MathScore)
        st.success('Your total score is {}'.format(result))
        if result>250:
            st.success("The student score grade S - First Class")
        if result>180:
            st.success("The student score grade A - Second Class")
        if result>100 and result<180:
            st.success("The student score grade B - Third Class")
        if result<100:
            st.success("The student is poor in learning! Please provide additional care!!")

if __name__ == '__main__':
    main()