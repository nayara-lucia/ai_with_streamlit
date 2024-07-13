import streamlit as st
import pandas as pd
import plotly.express as px
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.2f}'.format)

st.set_page_config(
    page_title="Análise Exploratória",
    page_icon="🧠",
    layout="wide"
)

st.header("Análise Exploratória - PIB")
tab1, tab2, tab3 = st.tabs(['📝 Resumo', '📊 Visualização', '💹 Maiores Valores'])

# @st.cache_data
df = pd.read_csv("dados.csv", sep=';')
df['PROPORCAO'] = df["VALOREMPENHO"] / df["PIB"]

with tab1:
    desc = df.describe(include=['float64'])
    desc.loc['var'] = desc.loc['std'] ** 2
    desc.loc['amplitude'] = desc.loc['max'] - desc.loc['min']
    st.dataframe(desc)

with tab2:
    col1, col2 = st.columns(2)
    columns = [col1, col2]
    
    float_columns = df.select_dtypes(include='float64').columns[:2]
    
    for column, column_st in zip(float_columns, columns):
        with column_st:
            fig = px.histogram(df, x=column, title=f"Histograma: {column}")
            st.plotly_chart(fig)
            
    for column, column_st in zip(float_columns, columns):
        with column_st:
            fig = px.box(df, x=column, title=f"Boxplot: {column}")
            st.plotly_chart(fig)
            
with tab3:
    top_n = st.number_input("Selecione o Top valores:",min_value=2,max_value=10,value=5,step=1)
    
    col1, col2,col3 = st.columns(3,gap='large')
    columns = [col1, col2,col3]
    
    float_columns = df.select_dtypes(include='float64')
    
    
    for column, column_st in zip(float_columns, columns):
        with column_st:
            fig = px.pie(df.nlargest(top_n,column), values=f'{column}', names='MUNICIPIO',title=f"{column}")
            st.plotly_chart(fig)
            
    
    