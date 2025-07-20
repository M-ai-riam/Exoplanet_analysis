# Exoplanet_analysis
A python built app, leveraging streamlit, panda, seaborn, matplotlib, numpy, and scikit.
# Project Name:
ASTROATA
# Description
This is a streamlit based app which integrates astronomy and data analysis. The sole purpose of this app is to dive into the world of research and analysis. Nowadays, there's this trend and in depth research of finding life out of earth, having the passion of astronomy this app leverages a begginner friendly and a basic user interface helping in finding whether or not the newly found planet can or cannot support life. 
# Feature:
* Predicts Habitability based on planet radii, stellar temperature and orbital period
Technologies Used:
Python
Streamlit
Panda
Seaborn
Matplotlib
Scikit
Numpy

# Installation
# Firstly install 
pip install streamlit
pip install panda
pip install scikit
pip install numpy
pip install seaborn
pip install matplotlib
# Download data from Nasa exopanet data 
# Code for analysis

import pandas as pd

exo= pd.read_csv('PS_2025.07.15_03.15.01.csv',delimiter=',',skiprows=97)

exo.head()

exo.info()
exo.describe()

col=['pl_name','pl_rade','pl_orbper','st_teff','pl_eqt']
exo=exo[col]

exo.isnull()

exo['pl_rade']=exo['pl_rade'].fillna(exo['pl_rade'].median())
exo['pl_orbper']=exo['pl_orbper'].fillna(exo['pl_orbper'].median())
exo['st_teff']=exo['st_teff'].fillna(exo['st_teff'].median())# not including pl_eqt cus it doesnt depend on median n using it could result in the wrong analysis of the data

exo['pl_eqt'].isnull().sum()


exo['pl_eqt'].dropna()
exo= exo.dropna()#dropping rows with too many nan values


import numpy as np
exo['Habitable']=np.where((exo['pl_rade'].between(0.5,1.5)) & (exo['st_teff'].between(4500,6500)),1,0)# binary column 0 for non habitable and 1 for habitable


exo_filtered = exo[(exo['pl_rade'] <= 20) & (exo['st_teff'] <= 10000)]


exo['Habitable'].value_counts()# tells the unique (distinct) count 


exo1_org=exo.copy()


from sklearn.preprocessing import StandardScaler

imp=['pl_rade','pl_orbper','st_teff']#standardizing the features so that every feature has the same range instead if a biased result
standard= StandardScaler()
exo[imp]=standard.fit_transform(exo[imp])

import seaborn as sns
import matplotlib.pyplot as mpl


sns.pairplot(exo,vars=['pl_rade','pl_orbper','st_teff'],hue='Habitable')
mpl.show()


sns.heatmap(exo[['pl_rade','pl_orbper','st_teff']].corr(),annot=True,cmap='coolwarm')
mpl.title('Correlation Grid')
mpl.show()


sns.scatterplot(x='pl_rade', y='st_teff', hue='Habitable', data=exo_filtered)
mpl.axvspan(0.5, 2, color='blue', alpha=0.1, label='Habitable Radius (0.5-2)')
mpl.axhspan(4500, 6500, color='green', alpha=0.1, label='Habitable Temp (4500-6500 K)')
mpl.title('Habitability Dependency on Planet Radius and Stellar Temperature')
mpl.xlabel('Planet Radii (Earth radii)')
mpl.ylabel('Stellar Temperature (K)')
mpl.legend()
mpl.xlim(0,2)
mpl.ylim(3000,7000)
mpl.show()


mpl.savefig('hab_rad_temp.png')


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
x=exo[['pl_rade','pl_orbper','st_teff']]
y=exo['Habitable']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)


y_pred=model.predict(x_test)
#a feature is a column


#accuracy and classification report
print('Accuracy report:',accuracy_score(y_pred,y_test))
print('Classification',classification_report(y_pred,y_test))

#find out which feature is important 
imp_fea=pd.DataFrame({'Feature':x.columns, 
                      'Importance':model.feature_importances_})#extracts the importance score from ml model
print(imp_fea.sort_values('Importance',ascending=False))
sns.barplot(x='Feature',y='Importance',data=imp_fea)
mpl.title("Important Features",None,'Left',)
mpl.show()

# Code for the app( 1st page)

import streamlit as st
st.set_page_config(layout="wide")
st.title("Exoplanet Habitability Predictor")
st.set_page_config(page_title="Professional App", layout="wide")

# Custom navigation bar
nav = st.radio("Go to:", ["Home", "Dashboard", "Settings"], index=0, horizontal=True)

if nav == "Home":
    st.title("Home Page")
    st.write("Welcome to the home page.")
elif nav == "Dashboard":
    st.title("Dashboard")
    st.write("View Prediction Panel.")

st.set_page_config(page_title="Professional App", layout="wide", initial_sidebar_state="expanded")
import streamlit as st

def set_background():
    st.markdown("""
        <style>
        .stApp {
            background-image: url("images.jpg");  # Replace with your image URL
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .stApp > header {
            background-color: rgba(0, 0, 0, 0.7);  # Semi-transparent header
        }
        .sidebar .sidebar-content {
            background-color: rgba(255, 255, 255, 0.9);  # Semi-transparent sidebar
        }
        h1, h2, h3 {
            color: #ffffff;  # White text for readability on dark background
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: red;
            border-radius: 5px;
            padding: 10px 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

#Call the function to apply the background
set_background()
# 2nd page 
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

exo=pd.read_csv('PS_2025.07.15_03.15.01.csv',skiprows=97)
exo['pl_rade']=exo['pl_rade'].fillna(exo['pl_rade'].median())
exo['pl_orbper']=exo['pl_orbper'].fillna(exo['pl_orbper'].median())
exo['st_teff']=exo['st_teff'].fillna(exo['st_teff'].median())
exo['Habitable']=np.where((exo['pl_rade'].between(0.5,1.5)) & (exo['st_teff'].between(4500,6500)),1,0)
from sklearn.model_selection import train_test_split
X=exo[['pl_rade','pl_orbper','st_teff']]
y=exo['Habitable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)  # Convert to NumPy array
X_test_scaled = scaler.transform(X_test.values)        # Convert to NumPy array

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

st.title("Prediction Panel")
st.write("Enter planet parameters for prediction")
radi=st.slider("Planet Radius (Earth Radii)",0.1,10.0,1.5)
orb_per=st.slider("Planet orbital period (Days)",0.1,1000.0,365.0)
st_temp=st.slider("Stellar Temperature (Kelvin)",2000,10000,5500)

input_fields=scaler.transform([[radi,orb_per,st_temp]])
pre_result=model.predict(input_fields)

if st.button("Predict"):
    final_result="Potential for sustaining life" if pre_result[0]==1 else "No possibility of Life development or sustainment"
    st.write(f'Prediction:{final_result}')
# Usage
Run the mentioned below command on the terminal with the name with whihc the python file is saved.
# Example command
Streamlit run astroata.py

Contributing
Contributions are welcome! Please read the contributing guidelines for details.
License
This project is licensed under the MIT License.
