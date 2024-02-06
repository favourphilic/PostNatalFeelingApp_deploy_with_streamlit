import streamlit as st 
import pandas as pd
import numpy as np
import pickle 



st.set_page_config(
    page_title="PostNATAL-App",
    page_icon="🧊",
    #layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("# **_PostNatal-APP_**")


#---------


import base64 




page_bg_img =f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image:url('https://images.pexels.com/photos/3875218/pexels-photo-3875218.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1');
background-size: cover;
background-size: 90%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.write(page_bg_img, unsafe_allow_html=True)





#-----------






#Load the model
with open("knnmodel.pkl", "rb") as f:
    model = pickle.load(f)

with st.form("my_form"):
   st.write(f'**{"Fill In The Fileds  To Make prediction"}**')
   age = st.slider( f'**{"Please Select  Your Age"}**')
   sad_fear = st.number_input(f'**{"Do You Feel Sad or Tearful"}**')
   irritable = st.number_input(f'**{"Do You Feel Irritable towards baby & partner"}**')
   sleep = st.number_input(f'**{"Do You Have Trouble sleeping at night"}**')
   conc = st.number_input(f'**{"Do You Have Problems concentrating or making decision"}**')
   eat = st.number_input(f'**{"Do You Overeact or Experience loss of appetite"}**')
   guilt = st.number_input(f'**{"Do You Feel Guilt"}**')
   bond= st.number_input(f'**{"Do You Have Problems of bonding with baby"}**')
   suicide= st.number_input(f'**{"Do You Have Any Suicide attempt"}**')


   test= np.array([age, sad_fear, irritable, sleep, conc, eat,guilt,bond, suicide]).reshape(1, -1)
   # Every form must have a submit button.
   submitted = st.form_submit_button("Submit")
   if submitted:
       pred = model.predict(test)[0]
       if pred == 0:
          st.markdown("# **_You Are Not Feeling Anxious_** ")
       else:
           st.markdown("# **_You Are Feeling Anxious_** ")
           

# css="""
# <style>
#     [data-testid="stForm"] {
#         background: LightBlue;
#     }
# </style>
# """
# st.write(css, unsafe_allow_html=True)





#st.write("Outside the form")

#st.sidebar.header("Configuration")