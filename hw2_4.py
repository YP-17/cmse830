# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:00:51 2022

@author: Pavan
"""

import streamlit as st
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns



df_geyser = sns.load_dataset('geyser')
fig1 = px.scatter(df_geyser, x="duration", y="waiting", color='kind')
st.plotly_chart(fig1, use_container_width=True)

df_exercise = sns.load_dataset('exercise')
fig2 = px.scatter_3d(df_exercise, x= 'pulse', y= 'time', z= 'diet', color='kind')
st.plotly_chart(fig2, use_container_width=True)

df_healthexp = sns.load_dataset('healthexp')
fig3 = px.scatter_3d(df_healthexp, x= 'Year', y='Spending_USD', z= 'Life_Expectancy', color= 'Country')
st.plotly_chart(fig3, use_container_width=True)




