import streamlit as st
import pandas as pd
import altair as alt
from PIL import Image

data_ghg = pd.read_csv("https://github.com/YP-17/cmse830/blob/main/ghg-emissions-by-sector.csv")


st.subheader("""
# ✨ GHG Emissions ✨. 
## The goal is to visualize, compare how different countries emitt Green House Gases .
""")

option = st.selectbox(
    "What would you like to do",
    ('Instructions','Year V/s Sector','Year Vs Country','Country Vs % Change in year'))


st.markdown('**Note:** 3 variables defines this project')
         
st.write(' **1. Year:** This is a time line from 1990 to 2019')
         
st.write('**2. Country:** Selects both countries and continents')
         
st.write('**3. Sectors:** This consists of 11 different sectors that boosts a country GDP')

cols = ['Agriculture', 'Land-use change and forestry',
           'Waste', 'Industry', 'Manufacturing and construction', 'Transport',
           'Electricity and heat', 'Buildings', 'Fugitive emissions',
           'Other fuel combustion', 'Aviation and shipping']

if st.button('Individual Sector Details?'):
    st.write(cols)


if option =='Instructions':
    st.subheader(""" This Project has 3 visualizations, follow the instructions to get more clarity""")
    st.write(""" **Visualization 1** describes how GHG emissions from different countries evolve over timeline """)
    st.write(""" **Visualization 2** conveys top 20 most pollution emission countries with their share to world pollution""")
    st.write(""" **Visualization 3** is more detailed picture of selected countries percentage change of pollution WRT previous """)


if option=='Year V/s Sector':
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
    
    if st.button('Click here for observations'):
        st.write("""By playing around with different countries, it is evident
                 that every on an average doubles or tripled its ghg emissions
                 over 29 years""")
    
    
    
    
    
    # Display both charts together
    st.altair_chart((reqchart1).interactive(), use_container_width=True)

elif option == 'Year Vs Country':
    cols = ['Agriculture', 'Land-use change and forestry',
           'Waste', 'Industry', 'Manufacturing and construction', 'Transport',
           'Electricity and heat', 'Buildings', 'Fugitive emissions',
           'Other fuel combustion', 'Aviation and shipping']
    
    
    st.subheader("Visualizing countries and Continents")
    
    st.sidebar.title("""
    # Select the specific sector 
    """)
    
    
    choice_sector = st.sidebar.selectbox(
        "y axis",
        cols)
    
    color = st.select_slider(
        'Select the timeline',
        options=data_ghg["Year"].unique())
    st.write('Selected till', color)
    
    data_ghg['total'] = data_ghg[cols].sum(axis=1)
    temp_data = data_ghg[['Entity','Year',choice_sector,'total']].dropna()
    req_data = temp_data.where(temp_data['Year'] == color).dropna().sort_values(by='total',ascending=False)
    req_data = req_data.head(25)
    
    #new_ghg = data_ghg.where(data_ghg['Year'] == color).dropna()
    
    
    
    
    reqchart1 = alt.Chart(req_data).mark_bar().encode(
        y = "Entity",
        x= choice_sector,
        color=choice_sector,
        size=alt.Color(choice_sector, scale = alt.Scale(scheme='plasma')),
        tooltip=["Entity",choice_sector,"Year"]
    ).interactive()
    
    if st.button('Click here for observations'):
        st.write(""" Over that span of 29 years similar countries are responsible
                 for 85% of pollution and the net ghg emmisions are increasing""")
        image = Image.open('https://github.com/YP-17/cmse830/blob/main/global_emissions_sector_2015.png')
        st.image(image, caption='Data of 1990-2019')
    st.altair_chart((reqchart1).interactive(), use_container_width=True)
    
elif option == 'Country Vs % Change in year' :  
    data_ghg = pd.read_csv("C:/Users/dell/Desktop/main project/ghg-emissions-by-sector.csv")
    data_ghg = data_ghg.drop('Code', axis=1)
    data_ghg = data_ghg.dropna()
    cols = ['Agriculture', 'Land-use change and forestry',
           'Waste', 'Industry', 'Manufacturing and construction', 'Transport',
           'Electricity and heat', 'Buildings', 'Fugitive emissions',
           'Other fuel combustion', 'Aviation and shipping']
    
    
    
    options_country = st.multiselect(
        'Select a countries',
        data_ghg["Entity"].unique())
    
    for i in options_country:
        st.write('You selected:', i)
        
        
    # options_sector = st.multiselect(
    #     'Select a sector',
    #     cols)
    
    
    # for i in options_sector:
    #     st.write('You selected:', i)
        
        
    
    
    
    data_ghg['total'] = data_ghg[cols].sum(axis=1)
    data_ghg['change'] = 0*data_ghg['total']
    
    
    def stage1(data):
        interval = alt.selection_interval(encodings=['x'])
        line = alt.Chart(data).mark_line(point={"filled": False,"fill": "white"}).encode(
            x = 'Year:N',
            y= 'change:Q',
            tooltip=["Entity","change"],
            color=alt.condition(interval, 'Entity', alt.value('red'))
        ).add_selection(
        interval)
        
        return line
    
    
    
    # dummy_data = data_ghg.where(data_ghg["Entity"]=="India").dropna()
    # dummy_data['change'] = dummy_data['total'].diff()
    # dummy_data['change'] = (dummy_data['change']/dummy_data['total'])*100 
    
    
    reqchart1 = alt.Chart(data_ghg).mark_line().encode(
    ).interactive()
    
    
    
    for i in options_country:
        dummy_data = data_ghg.where(data_ghg["Entity"]==i).dropna()
        dummy_data['change'] = dummy_data['total'].diff()
        dummy_data['change'] = (dummy_data['change']/dummy_data['total'])*100
        reqchart1 = reqchart1+stage1(dummy_data)
        
        
        
    if st.button('Click here for observations'):
        st.write("""Net change for each country before 2015 is on rise but
                 Paris agreement made it to become lower""")
        image = Image.open('https://github.com/YP-17/cmse830/blob/main/1578429041326.png')
        st.image(image, caption='Goals of Paris Agreement')
                 
    st.altair_chart((reqchart1).interactive(), use_container_width=True)
