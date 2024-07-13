import streamlit as st
import pandas as pd
import scipy.stats as stats
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
plt.style.use('dark_background')


st.set_page_config(page_title="Tests for normality",page_icon="📊",
    layout="wide")

st.header("Normality Check")

def read_file_transform(file):
    decoder = StringIO(file.read().decode("utf-8"))
    df = pd.read_csv(decoder, header=None)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    return df

def histogram(data):
    fig, ax = plt.subplots(figsize=(20,15))
    ax.hist(x=data.iloc[:,0],bins='auto',rwidth=0.9,facecolor='darkblue')
    ax.set_title("Histogram",fontsize=35 )
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    return fig

def qq_plot(data):
    fig, ax = plt.subplots(figsize=(20, 15))
    stats.probplot(data.iloc[:, 0], dist='norm', plot=ax)
    ax.set_title("QQ-Plot",fontsize=35 )
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    return fig

def shapiro_test(data):
    p_value = stats.shapiro(data.iloc[:,0]).pvalue
    if p_value < 0.05:
        message = "The data does not follow a normal distribution"
    else:
        message = "There is not enough evidence to reject the normality of the data"
    return p_value, message


with st.sidebar:
    st.markdown(":green[**Settings**]")
    file_upload = st.file_uploader("Upload CSV file",type=['csv'],accept_multiple_files=False)
    if file_upload is not None:
        file = read_file_transform(file_upload)
    process = st.button("Process",disabled=(file_upload is None))
    
col1, col2 = st.columns(2)

if process:
    with col1:
        st.pyplot(histogram(file))
    with col2:
        st.pyplot(qq_plot(file))
    
    p, result = shapiro_test(file)
    if p < 0.05: 
        st.write(f":red[Shapiro-Wilk test p-value: **{p}**]")
        st.warning(result)
    else:
        st.write(f":green[Shapiro-Wilk test p-value: **{p}**]")
        st.success(result,icon="✅")
    
    