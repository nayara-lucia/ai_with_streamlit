import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from datetime import date
from io import StringIO

st.set_page_config(page_title="Milk Production Forecast",
    page_icon= "🥛",
    layout="wide")

st.title("Time Series Forecasting - Milk Production")

st.image("https://th.bing.com/th/id/OIG3.nuGsCDfi0UOM6MHszd.P?pid=ImgGn",width=300)

with st.sidebar:
    st.markdown(":green[**Please, fill in the fields for the forecast**]")
    
    upload_file = st.file_uploader("Choose CSV file", type=['csv'])
    if upload_file is not None:
        decoder = StringIO(upload_file.read().decode("utf-8"))
        df = pd.read_csv(decoder, header=None)
        
    date_init = date(2000,1,1)
    period = st.date_input(label="Initial period of the Series", value=date_init)
    period_pred = st.number_input(label="Forecast period in months", min_value=1, max_value=36,value= 12)
    
    process = st.button("Process", disabled=(upload_file is None))
    
if upload_file is not None and process:
    try:
        #Data transform
        new_data = pd.Series(df.iloc[:,0].values, index= pd.date_range(start=period, periods=len(df),freq='m'))

        #Decompose
        decompose = seasonal_decompose(new_data,model='additive')
        fig_decompose = decompose.plot()
        fig_decompose.set_size_inches(10,8)
    
        #Model
        model = SARIMAX(new_data,order=(2,0,0), seasonal_order=(0,1,1,12))
        model_fit = model.fit()
        fore_cast = model_fit.forecast(steps=period_pred)
        
        #Forecast
        fig_forecast, ax = plt.subplots(figsize=(10,8.1))
        ax.plot(new_data)
        ax.plot(fore_cast, color='red')
        
        #Columns
        col1,col2,col3 = st.columns([3,3,2]) #colum size
        
        with col1:
            st.subheader("Decompose")
            st.pyplot(fig_decompose)
        with col2:
            st.subheader("Forecast")
            st.pyplot(fig_forecast)
        with col3:
            st.subheader("Forecast Dataframe")
            st.dataframe(fore_cast)
            
    except Exception as e:
        st.error(f"Data error: {e}")

        
    