import streamlit as st st.set_page_config(layout="wide") st.set_page_config(page_title="Professional App", layout="wide")

st.set_page_config(page_title="Professional App", layout="wide", initial_sidebar_state="expanded")

def set_background(): st.markdown( """ <style> .stApp { background-image: url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80"); background-size: cover; background-position: center; background-attachment: fixed; background-repeat: no-repeat; }

    /* Title color */
    h1 {
        color: white !important;
        text-shadow: 2px 2px 4px #000000;
        text-align: center;
    }
    h2 {
        color: white !important;
        text-shadow: 2px 2px 4px #000000;
        text-align: center;
    }
    /* Header background */
    .stApp > header {
        background-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Sidebar background */
    .sidebar .sidebar-content {
        background-color: rgba(25, 25, 112, 1.0) !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#Apply the background set_background()

st.title("Exoplanet Habitability Predictor") st.title("WANNA ESCAPE EARTH")

2nd page
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
