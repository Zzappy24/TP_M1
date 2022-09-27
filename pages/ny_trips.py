import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import plotly.graph_objects as go
import plotly


def lecture(link):
    return pd.read_csv(link)

df_ny = lecture("pages/ny-trips-data.csv")
st.title("NY DATA")
st.dataframe(df_ny)

st.markdown("***")

st.title("NY DATA longitude and latitude")
st.dataframe(df_ny.iloc[:, 5:7])

st.markdown("***")



def display_map(df):
    return st.map(df)

def convert_time(df):
    j = 0
    for i in df.columns.str.contains("time"):
        if i:
            df.iloc[:, j] = pd.to_datetime(df.iloc[:, j])
        j += 1
    return df


def get_dom(dt):
    return dt.day


def get_weekday(dt):
    return dt.weekday()


def get_hour(dt):
    return dt.hour


def get_year(dt):
    return dt.year


def apply_dwhy(df):
    j = 0
    for i in df.columns.str.contains("time"):
        if i:
            df['day'] = df.iloc[:, j].map(get_dom)
            df['week_day'] = df.iloc[:, j].map(get_weekday)
            df['hour'] = df.iloc[:, j].map(get_hour)
            df['year'] = df.iloc[:, j].map(get_year)
        j += 1
    return df


def nulll(df):
    return f"le nombre de valeurs NaN dans le DataFrame : {df.isna().sum().sum()}"

def min(df):
    return float(df.min())
def max(df):
    return float(df.max())
def minI(df):
    return int(df.min())
def maxI(df):
    return int(df.max())
    
convert_time(df_ny)
st.dataframe(apply_dwhy(df_ny))

#df_map = df_ny.iloc[:, 5:7]
#df_map.rename(columns={"pickup_longitude": "longitude"}, inplace=True)
#df_map.rename(columns={"pickup_latitude": "latitude"}, inplace=True)

def Choix_utilisateur(df_ny):
    Vendor = list(df_ny["VendorID"].drop_duplicates())
    Vendor_choice = st.sidebar.multiselect('Choose Vendor:', Vendor, default=Vendor[0],key="Vendor")
    passenger = st.sidebar.slider(
        'passenger', min(df_ny["passenger_count"]), max(df_ny["passenger_count"]),step=1.0, key="Ca")
    trip_distance = st.sidebar.slider(
        'trip_distance', min(df_ny["trip_distance"]), max(df_ny["trip_distance"]), 2.0, key="Pr")
    day_pickup = st.sidebar.slider(
        'day_pickup',minI(df_ny["day"]), maxI(df_ny["day"]),minI(df_ny["day"]) , key="day")

    df_ny2 = df_ny.copy()
    df_ny2 = df_ny[df_ny['VendorID'].isin(Vendor_choice)]
    df_ny2 = df_ny2[df_ny2["passenger_count"] <= passenger]
    df_ny2 = df_ny2[df_ny2["trip_distance"] <= trip_distance]
    df_ny2 = df_ny2[df_ny2["day"] <= day_pickup]
    df_ny2 = df_ny2.sort_values('trip_distance', ascending=True).reset_index(drop=True)
    df_ny2.rename(columns={"pickup_latitude":'latitude',"pickup_longitude":"longitude"}, inplace=True)
    return df_ny2


st.title("Location on a map")

st.sidebar.header("Choose parameters for the Map :")

df_ny2 = Choix_utilisateur(df_ny)
st.dataframe(df_ny2)
display_map(df_ny2)

st.markdown("***")

st.write(f"{nulll(df_ny)}")

st.markdown("***")
def map_pydeck(df_ny2):
    return st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=40.7,
            longitude=-73.99,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
            'HexagonLayer',
            data=df_ny2,
            get_position='[longitude, latitude]',
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=df_ny2,
                get_position='[longitude, latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))

st.subheader("Map with PyDeck")

map_pydeck(df_ny2)

st.markdown("***")

st.sidebar.markdown("***")


st.sidebar.header("COMPARE group by Passenger_count")


def tracer_spyder_graphe(df_ny2):
    categories = ['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount', 'total_amount']
    fig = go.Figure()
    t = df_ny.groupby("VendorID").mean()
    t = t[['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount', 'total_amount']]
    st.table(t)


    fig.add_trace(go.Scatterpolar(
    r=t.iloc[0,:].values.tolist(),
    theta=categories,
    fill='toself',
    name="Vendeur 1"))

    fig.add_trace(go.Scatterpolar(
        r=t.iloc[1,:].values.tolist(),
        theta=categories,
    fill='toself',
    name="Vendeur 2"
    ))
    col1, col2, col3 = st.columns(3)
    return col1.plotly_chart(fig)
st.title("Comparison between Vendor")
tracer_spyder_graphe(df_ny2)

value = df_ny.groupby("passenger_count").mean()

def bar_plot(df_ny,nom_colonne,a):
    #st.write("Select 2 Passenger and 2 things to compare at least ")
    Class = st.sidebar.multiselect(
    f"number {nom_colonne} do you want ?",
    options = df_ny[nom_colonne].unique(),
    default = df_ny[nom_colonne].unique()[0:6], key=a)
    things = st.sidebar.multiselect(
    "What do you want to compare ?",
    options = value.columns.values,
    default = value.columns.values[3:6], key=a+"2")
    l=[]
    colonne = []
    for i in value.index:
        if i not in Class:
            l.append(i)
    for j in value.columns:
        if j not in things:
            colonne.append(j)
    value2 = value.drop(colonne, axis=1)
    value2 = value2.drop(l)
    st.bar_chart(value2)
st.markdown("***")
st.title("COMPARE group by Passenger_count")
bar_plot(df_ny, "passenger_count","pass")
st.sidebar.markdown("***")

st.markdown("***")
st.title("COMPARE group by VendorID")

st.sidebar.header("COMPARE group by VendorID")
bar_plot(df_ny, "VendorID","vend")


    


st.title("Price by hour")

df_ny_group_by_hour = df_ny.groupby("hour").mean()
st.line_chart(df_ny_group_by_hour[['fare_amount', 'tip_amount', 'total_amount']])




                  

