import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


st.set_page_config(page_title="Poisson Distribuition",page_icon="📊",
    layout="centered")

st.header("Probability of Equipment Failure")

def calc(type,mean):
    lambda_ = mean
    p_intervals = np.arange(lambda_ - 2, lambda_ + 3) # array([0, 1, 2, 3, 4])
    
    if type == "Probability mass function":
        prob = {i: poisson.pmf(i, lambda_) for i in p_intervals}
        
    elif type == "Cumulative distribution function":
        prob = {i: poisson.cdf(i, lambda_) for i in p_intervals}
        
    else:
        prob = {i: poisson.sf(i, lambda_) for i in p_intervals}
    
    return prob
    

with st.sidebar:
    st.markdown(":green[**Settings**]")
    selected = st.radio("Select the calculation type",options=["Probability mass function","Cumulative distribution function","Survival function"]) 
    occur = st.number_input("Type the mean occurrence of equipment failure",min_value=1,max_value=100,value=2)
    process = st.button("Process")

if process:
        prob = calc(selected,occur)     
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(20,20))
        ax.bar(*zip(*sorted(prob.items())),color='grey')
        ax.set_xlabel("Number of Occurrences",fontsize=18)
        ax.set_ylabel("Probability",fontsize=18)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        st.pyplot(fig)

            

            
        
        
