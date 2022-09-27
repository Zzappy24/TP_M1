import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

def Machine_Learning(df_uber):
    st.markdown("<h1 style='text-align: center; color: white;'>Matrice de Corrélation</h1>", unsafe_allow_html=True)
    import seaborn as sn
    fig, ax = plt.subplots(figsize = (50,50))
    df_uberll = df_uber.copy()
    corrM = df_uberll.corr()
    sn.heatmap(round(corrM, 2), annot=True, annot_kws={"size": 32})
    plt.xticks(rotation=74, fontsize=30)
    plt.yticks(fontsize=30) 
    st.pyplot(fig)


    df_uberll = pd.get_dummies(df_uber, columns = ['day','smoker','sex'])

    x = df_uberll.drop(["time"], axis = 1)
    y = df_uberll["time"].copy()
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)

    # Splitting the dataset into training and testing set (50/20)
    x_train, x_test, y_train, y_test = train_test_split(x, y_transformed, test_size = 0.5, random_state = 0)

    logisticRegr = LogisticRegression(max_iter=1000)
    logisticRegr.fit(x_train, y_train)
    score = logisticRegr.score(x_test, y_test)
    y_pred_log = logisticRegr.predict(x_test)

    st.write("On obtient une précision de ",score)

    st.markdown("***")

    df_uberCoef2 = pd.DataFrame(index = list(x.columns)).copy()
    df_uberCoef2_exp = pd.DataFrame(index = list(x.columns)).copy()


    for k in range(0, len(logisticRegr.coef_)):
        df_uberCoef2.insert(len(df_uberCoef2.columns),f"Coefs {k}",logisticRegr.coef_[k])
        df_uberCoef2_exp.insert(len(df_uberCoef2_exp.columns),f"Coefs {k}",np.exp(logisticRegr.coef_[k]))

    st.markdown("<h3 style='text-align: center; color: white;'>Coefficient devant X</h3>", unsafe_allow_html=True)
    st.dataframe(df_uberCoef2)

    st.markdown("<h3 style='text-align: center; color: white;'>Coefficient exponentiel</h3>", unsafe_allow_html=True)
    st.dataframe(df_uberCoef2_exp)

    st.markdown("***")

    st.markdown("<h2 style='text-align: center; color: white;'>On inverse la procédure d'encodage</h2>", unsafe_allow_html=True)
    df_uberCoef3 = pd.DataFrame(index = list(x.columns)).copy()
    df_uberCoef3_exp = pd.DataFrame(index = list(x.columns)).copy()


    for k in range(0, len(logisticRegr.coef_)):
        df_uberCoef3.insert(len(df_uberCoef3.columns),f"Coefs {lab.inverse_transform(np.unique(y_transformed))[k]}",logisticRegr.coef_[k])
        df_uberCoef3_exp.insert(len(df_uberCoef3_exp.columns),f"Coefs {lab.inverse_transform(np.unique(y_transformed))[k]} exp",np.exp(logisticRegr.coef_[k]))

    st.markdown("<h3 style='text-align: center; color: white;'>Coefficient devant X</h3>", unsafe_allow_html=True)
    st.dataframe(df_uberCoef3)

    st.markdown("<h3 style='text-align: center; color: white;'>Coefficient exponentiel</h3>", unsafe_allow_html=True)  
    st.dataframe(df_uberCoef3_exp)

    st.markdown("***")

    st.markdown("<h2 style='text-align: center; color: white;'>Matrice de confusion</h2>", unsafe_allow_html=True)


    matrice_confusion = confusion_matrix(y_test, y_pred_log)
    fig2, ax1 = plt.subplots(figsize = (20,20))
    sns.heatmap(matrice_confusion, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Greens_r', annot_kws={"size": 30})
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')



    ax1.xaxis.set_ticklabels(lab.inverse_transform(np.unique(y_test)).tolist(), rotation=74, fontsize=20)
    ax1.yaxis.set_ticklabels(lab.inverse_transform(np.unique(y_test)).tolist(), rotation=0, fontsize=20)


    all_sample_title = f'Accuracy Score: {round(score, 4)}'
    plt.title(all_sample_title, size = 25);

    return st.pyplot(fig2)


def user_input(df):
    st.sidebar.markdown("***")
    st.sidebar.title("Choisir les paramètres du test de machine leaarning")


    total_bill = st.sidebar.slider('total_bill', df["total_bill"].min(), df["total_bill"].max(), 5.3, key="bill")
    tip = st.sidebar.slider('tip',0.0,df["tip"].max(),3.3, key="tip")
    #color = st.select_slider('Select a color of the rainbow', options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'],key="ddd")
    sex = st.sidebar.select_slider('sex',options=['Male','Female'],key="sex")
    #smoker = st.sidebar.slider('smoker', "Yes", "No", 2.3, key="Fi")
    smoker = st.sidebar.select_slider('smoker',options=df['smoker'].unique(), key="Sm")
    day = st.sidebar.select_slider('day',options=df['day'].unique(), key="day")
    size = st.sidebar.slider('size', int(df["size"].min()), int(df["size"].max()), 1, key="size")
    data = {'total_bill' : total_bill,
    'tip' : tip ,
    'sex' : sex,
    'smoker' : smoker,
    'day' : day,
    'size' : size},



    parametres=pd.DataFrame(data,index=[0])
    return parametres

def Démonstration_prédiction(df):
    lab = preprocessing.LabelEncoder()
    col1,col2 = st.columns(2)
    col1.subheader("Paramètre pour la prédictions :")

    df2 = user_input(df_uber)
    col2.dataframe(df2)

    if df2.iloc[0,2]=="Male":
        df2.iloc[0,2]=0
    if df2.iloc[0,2]=="Female"  :
        df2.iloc[0,2]=1
    if df2.iloc[0,3]=="No":
        df2.iloc[0,3]=0
    else :
        df2.iloc[0,3]=1
    if df2.iloc[0,4]=="Sun":
        df2.iloc[0,4]=0
    if df2.iloc[0,4]=="Sat":
        df2.iloc[0,4]=1
    if df2.iloc[0,4]=="Thur":
        df2.iloc[0,4]=2
    if df2.iloc[0,4]=="Fri":
        df2.iloc[0,4]=3
    
    dfz = df2["sex"]
    B = df2.to_numpy()
    B = np.array(df2)
    

   

    st.markdown("***")
    
    df['day'] = lab.fit_transform(df['day'])
    df['smoker'] = lab.fit_transform(df['smoker'])
    df['sex'] = lab.fit_transform(df['sex'])

    x = df.drop(["time"],axis=1)
    y = df_uber["time"].copy()
    y_transformed = lab.fit_transform(y)
    


    scaler = StandardScaler()


    # Splitting the dataset into training and testing set (50/50)
    x_train, x_test, y_train, y_test = train_test_split(x, y_transformed, test_size = 0.5, random_state = 0)


    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression(max_iter=1000)
    logisticRegr.fit(x_train, y_train)
    #predictions = logisticRegr.predict(x_test)
    score = logisticRegr.score(x_test, y_test)
    y_pred_log = logisticRegr.predict(B)
    st.subheader("Prédiction : ")
    st.write("repas labélisée : ",y_pred_log[0])
    reponse =lab.inverse_transform(np.unique(y_pred_log[0]))

    st.write("le repas est un : ")
    st.dataframe(reponse)
    
    return st.write("avec une précision de ",score)

def Tracer_Pie(df_fumeur):
        fig = go.Figure(
                    go.Pie(
                    hole = 0.5,
                    labels = [df_fumeur.index[0]+" fumeur ", df_fumeur.index[1]+" fumeur "],
                    values = [df_fumeur.iloc[0,:].values[0], df_fumeur.iloc[1,:].values[0]],
                    hoverinfo = "value",
                    textinfo = "label+percent"
                ))

        return col2.plotly_chart(fig)


st.markdown("<h1 style='text-align: center; color: white;'>Uber</h1>", unsafe_allow_html=True)
df_uber = pd.read_csv("tips.csv", delimiter=";")
col1,col2,col3 = st.columns(3)
col2.write(df_uber)

st.markdown("***")



radML = st.sidebar.radio("Modèle et Visualisation", ["Démonstration prédiction", "Logistic Regression plus en détail", "Analyse / Visualization"])

if radML == 'Démonstration prédiction':
    Démonstration_prédiction(df_uber)
if radML == "Logistic Regression plus en détail":
    Machine_Learning(df_uber)
if radML == "Analyse / Visualization":
    st.markdown("<h2 style='text-align: center; color: white;'>Nombre de fumeurs par genre</h2>", unsafe_allow_html=True)
    st.markdown("  ")
    df_fumeur = df_uber[["sex",'smoker']]
    df_fumeur['smoker'] = df_fumeur['smoker'].map({'No':0,'Yes':1})
    df_fumeur = df_fumeur.groupby("sex").count()
    st.dataframe(df_fumeur)
    col1, col2 = st.columns(2)
    col1.bar_chart(df_fumeur)
    
    Tracer_Pie(df_fumeur)

    st.markdown("***")
    st.markdown("<h2 style='text-align: center; color: white;'>total bill and tip per day</h2>", unsafe_allow_html=True)

    df_uber_group_by_day = df_uber.groupby("day").mean()
    st.line_chart(df_uber_group_by_day[['total_bill', 'tip']])
    
    st.markdown("***")
    st.markdown("<h2 style='text-align: center; color: white;'>size AVG by day hue by Dinner/Lunch</h2>", unsafe_allow_html=True)

    fig3 = plt.figure(figsize=(10, 4))
    sns.lineplot(data=df_uber, x="day", y="size", hue="time")
    st.pyplot(fig3)






    


