import streamlit as st
import pandas as pd
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
import json
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Multipage App",
)
@st.cache()
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"

lottie_ny = load_lottiefile("./90724-travel-world-usa.json")
lottie_uber = load_lottiefile("./34273-mercedes.json")




st.sidebar.success("Select a page above")

st.markdown("<h1 style='text-align: center; color: white;'>TP3 STREAMLIT</h1>", unsafe_allow_html=True)

df_ny = pd.read_csv("pages/ny-trips-data.csv")
df_tips=pd.read_csv("pages/tips.csv", delimiter=";")




col1, col2 = st.columns(2)


with col1:
    st_lottie(lottie_ny, key="1")
with col2:
    st_lottie(lottie_uber, key="2")

st.markdown("***")

col1, col2 = st.columns(2)

with col1:
    st.title("ny")
    st.dataframe(df_ny)
with col2:
    st.title("tips")
    st.dataframe(df_tips)




st.write("Please choose one of the TP in the sidebar")