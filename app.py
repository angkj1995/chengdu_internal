import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt

# reduce output
st.set_option('deprecation.showPyplotGlobalUse', False)

## Title of app
st.title('Company Bankruptcy Prediction')

uploaded_file = st.file_uploader("Upload company dataset")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file, index_col='company_name')
    st.write(dataframe)
    

## The model
@st.cache_resource
def readmodel():
    cat_model = joblib.load('cat_model')
    return cat_model
cat_model = readmodel()

## Sample data
@st.cache_data
def readtestX():
    df = pd.read_csv('testX.csv')
    return df
testX = readtestX()

## Full data
@st.cache_data
def readfull():
    df = pd.read_csv('data.csv')
    return df
fulldf = readfull()


if not (uploaded_file is None):
    ## Model prediction
    st.subheader("Company predictions")
    cat_pred = cat_model.predict_proba(dataframe)[:,1]
    cat_english = []
    cat_confidence = []

    #Threshold
    for i in range(0,cat_pred.size):
        if cat_pred[i]>=0.14020593633115097:
            cat_english.append('Bankrupt risk')
            cat_confidence.append( (cat_pred[i] - 0.14020593633115097) / (1-0.14020593633115097) )
        else:
            cat_english.append('No Bankrupt risk')
            cat_confidence.append( -(cat_pred[i]-0.14020593633115097)/0.14020593633115097)

    predictions = pd.DataFrame(cat_english, index=dataframe.index, columns=['Risk'])
    predictions['Confidence'] = cat_confidence

    left_column, right_column = st.columns(2)
    left_column.dataframe(predictions.style.highlight_min(axis=0))
    ##How many bankrupt?
    right_column.write('Distribution of bankruptcy risk in uploaded dataset: ')
    right_column.write(predictions['Risk'].value_counts())





    ## Shapley values (cache)
    @st.cache_resource
    def makeshap():
        explainer = shap.Explainer(cat_model.predict, testX)
        shap_values = explainer(dataframe)
        return shap_values

    



    st.subheader('Basic Corporate Information')
    # Evaluate
    shap_values = makeshap()
    coy = st.selectbox('View most explanatory features of individual company',
             dataframe.index)

    #Get company index
    coy_index = dataframe.index.get_loc(coy)
    st.write('Blue features make a company lower risk, Red features make a company higher risk')
    st.pyplot(fig=shap.plots.bar(shap_values[coy_index]))





    #Get company variable
    coy_variable = st.selectbox('View Corporate Information Distribution',
             np.sort(dataframe.columns.values))

    #Point data to plot
    v = dataframe.loc[coy,coy_variable]
    #Point prediction to plot
    if cat_english[coy_index]=='Bankrupt risk':
        coyx=1
    else:
        coyx=0

    fig1 = plt.figure()
    sns.boxplot(x='Bankrupt?', y=coy_variable, data=fulldf, showfliers=False)
    plt.scatter(coyx, v, marker='x', s=100,c='red')
    st.write(fig1)


    





    
