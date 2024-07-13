import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Investment Forecast")

st.markdown("""---""")

data = pd.read_csv('slr12.csv',sep=';')

def train_reglin(x,y):
    model = LinearRegression()
    model.fit(x,y)
    return model
    
X = data.iloc[:, [0]]
y = data.iloc[:,1]

model = train_reglin(X,y)

col1, col2 = st.columns(2) #vertical_alignment="bottom"

with col1:
    st.subheader("Data (limit 5)")
    st.table(data.head(9))
    
with col2:
    st.subheader("Scatter Plot")
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax1.scatter(data['FrqAnual'],data['CusInic'],color='skyblue')
    ax1.plot(X, model.predict(X),color='blue')
    st.pyplot(fig)
    
st.markdown("""---""")

st.subheader("Annual fee")
new_value = st.number_input("Type here",min_value=1.0,max_value=9999999.0,value=1000.0,step=100.0)
process = st.button("Process")

def format_(p):
    return f"$ {p[0]:.2f}"

if process: 
    new_data = pd.DataFrame([[new_value]], columns=['FrqAnual'])
    pred = format_(model.predict(new_data[['FrqAnual']]))
    st.subheader(f"The approximate initial investment is: :green[{pred}]")
    

    
