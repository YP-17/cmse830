# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 19:29:07 2022

@author: Pavan
"""

# import libraries
import streamlit as st
import pandas as pd
import altair as alt




data_ghg = pd.read_csv("https://raw.githubusercontent.com/YP-17/cmse830/main/ghg-emissions-by-sector.csv")

st.sidebar.title("""
# By selecting the country and Timeline, it is easy to visualize how different countries emmit pollution in the specific sectors
""")

# allow user to choose which portion of the data to explore
choice_country = st.sidebar.selectbox(
    "x axis",
    data_ghg["Entity"].unique())
    

cols = ['Agriculture', 'Land-use change and forestry',
       'Waste', 'Industry', 'Manufacturing and construction', 'Transport',
       'Electricity and heat', 'Buildings', 'Fugitive emissions',
       'Other fuel combustion', 'Aviation and shipping']

choice_sector = st.sidebar.selectbox(
    "y axis",
    cols)


color = st.select_slider(
    'Select the timeline',
    options=data_ghg["Year"].unique())

st.write('Selected till', color)



timeline = []
temp_data = data_ghg.where(data_ghg['Entity'] == choice_country).dropna()
req_data = temp_data.where(temp_data['Year'] <= color).dropna()

# do the visuslization for req_data as time series plots

cols = ['Agriculture', 'Land-use change and forestry',
       'Waste', 'Industry', 'Manufacturing and construction', 'Transport',
       'Electricity and heat', 'Buildings', 'Fugitive emissions',
       'Other fuel combustion', 'Aviation and shipping']

data_ghg['total'] = data_ghg[cols].sum(axis=1)
new_ghg = data_ghg.sort_values(by='total')


choose_sector = st.button('Compare countries by Sectors')   
choose_countries = st.button('Compare countries total pollution')


def stage1(data):
    reqchart1 = alt.Chart(req_data).mark_area().encode(
    x='Year:N',
    y= choice_sector,
    tooltip=["Year"],
    ).interactive()
    
    return reqchart1



reqchart1 = stage1(req_data)
#sorted_total = data_ghg['Entity','sum'].sort_values(ascending = True)




# Display both charts together
st.altair_chart((reqchart1).interactive(), use_container_width=True)


