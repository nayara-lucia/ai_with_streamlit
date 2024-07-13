import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


st.set_page_config(
    page_title="Cars App",
    page_icon="🚗",
    layout="centered",
)

@st.cache_data
def transform_columns(base, column):
    return base[column].astype('category')

def encoder_model(x, y):
    encoding_X = OrdinalEncoder()
    encoding_y = LabelEncoder()
    X_transform = encoding_X.fit_transform(x)
    Y_transform = encoding_y.fit_transform(y)
    return X_transform, Y_transform, encoding_X, encoding_y

def encoder_new_data(x, encoding_X):
    return encoding_X.transform(x)

def tree_model(x, y):
    model = DecisionTreeClassifier()
    model.fit(x, y)
    return model

def predict(model, x):
    return model.predict(x)

def acc_metric(y_true, y_pred):
    result = accuracy_score(y_true, y_pred)
    return "{:.2f}%".format(result * 100)

def pipe(base):
    for column in base.columns:
        base[column] = transform_columns(base, column)

    X = base.iloc[:, 0:6]
    y = base.iloc[:, -1]

    X, y, encoding_X, encoding_y = encoder_model(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = tree_model(X_train, y_train)
    y_pred = predict(model, X_test)
    result = acc_metric(y_test, y_pred)
    return result, model, encoding_X, encoding_y

base = pd.read_csv("car.csv", sep=',')
result, model, encoding_X, encoding_y = pipe(base)

#----------------------------------------------------------------------------#
st.title("Classifying car quality")
st.image('https://maserati.scene7.com/is/image/maserati/maserati/amricas/LIPMAN_JL16928_V3.jpg?$1920x2000$&fit=constrain', width=300)
st.markdown("""---""")

st.write(f"Model Accuracy: :green[{result}]")

new_data_selections = [st.selectbox(f"{i}", base[i].unique()) for i in base.columns.drop('class')]

if st.button("Process"):
    new_data = pd.DataFrame([new_data_selections], columns=base.columns.drop('class'))
    new_data_trans = encoder_new_data(new_data, encoding_X)
    new_data_predict = predict(model, new_data_trans)
    res = encoding_y.inverse_transform(new_data_predict)
    res_formatted = ''.join(res)
    st.subheader(f"Prediction: :orange[{res_formatted}]")
