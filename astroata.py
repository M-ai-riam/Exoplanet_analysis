import streamlit as st
st.set_page_config(layout="wide")
st.set_page_config(page_title="Professional App", layout="wide")


st.set_page_config(page_title="Professional App", layout="wide", initial_sidebar_state="expanded")

def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }
        
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

#Apply the background
set_background()

st.title("Exoplanet Habitability Predictor")
st.title("WANNA ESCAPE EARTH")
