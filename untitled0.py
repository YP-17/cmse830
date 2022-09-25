# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:54:33 2022

@author: dell
"""

import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


siteHeader = st.container()
dataExploration = st.container()
newFeatures = st.container()

with siteHeader:
    st.title('Web App of Wisconsin Cancer Dataset!')

with dataExploration:
    st.header('Dataset: Wisconsin cancer dataset')
    st.text('I found this dataset at Kaggle ')

df_data = pd.read_csv("C:/Users/dell/Downloads/ICA6/data.csv")
print(df_data)


plot =['scatter', 'histogram', 'violin', '3d scatter']
selected_option_0= st.selectbox("Which Plor?", plot)
st.write('You selected:', selected_option_0)



if selected_option_0 == plot[1]:
    #choosing option-1 from the dropdown

    selected_option_1= st.selectbox("x axis feature?", df_data.columns)
    st.write('You selected:', selected_option_1)
    #choosing option-2 from the dropdown
    
    # choosing which plot
    
    selected_option_2= st.selectbox("y axis feature?", df_data.columns)
    st.write('You selected:', selected_option_2)
    
    fig=plt.figure(figsize=(9,7))
    sns.histplot(data=df_data, x=selected_option_1,y=selected_option_2, bins=20, hue="diagnosis")
    st.pyplot(fig)
        

if selected_option_0 == plot[0]:
    #choosing option-1 from the dropdown

    selected_option_1= st.selectbox("x axis feature?", df_data.columns)
    st.write('You selected:', selected_option_1)
    #choosing option-2 from the dropdown
    
    # choosing which plot
    
    selected_option_2= st.selectbox("y axis feature?", df_data.columns)
    st.write('You selected:', selected_option_2)
    
    fig=plt.figure(figsize=(9,7))
    sns.scatterplot(data=df_data, x=selected_option_1, y=selected_option_2, hue="diagnosis")
    st.pyplot(fig)
    
if selected_option_0 == plot[2]:
    #choosing option-1 from the dropdown
    selected_option_1= st.selectbox("x axis feature?", df_data.columns)
    st.write('You selected:', selected_option_1)
    #choosing option-2 from the dropdown
    # choosing which plot
    #selected_option_2= st.selectbox("y axis feature?", df_data.columns)
    #st.write('You selected:', selected_option_2)
    fig=plt.figure(figsize=(9,7))
    sns.catplot(data=df_data, x=selected_option_1, bins=20, hue="diagnosis",kind="violin")
    st.pyplot(fig)
