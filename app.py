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
background-size: 180%;
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
#-------------Markdown


st.markdown("*Postpartum anxiety* is **excessive worrying that occurafter** ***childbirth adpotion***.")

multi = '''
 People with postpartum anxiety may feel consumed with worry and constantly nervous or panicked. 
 If you or someone you know has symptoms of postpartum anxiety,
   get help from a healthcare provider immediately. 
 Treatment for postpartum anxiety includes behavioral therapy or medication.
'''
st.markdown(multi)

#---------------End of Markdown-
#Load the model
with open("knnmodel.pkl", "rb") as f:
    model = pickle.load(f)

with st.form("my_form"):
   st.markdown("""
    0-No, 1=Others, 2=Yes
""")
   st.write(f'**{"Fill In The Fileds  To Make prediction"}**')
   age = st.slider( f'**{"Please Select  Your Age"}**')
   sad_fear = st.number_input(f'**{"Do You Feel Sad or Tearful"}**', min_value=0, max_value=2)
   irritable = st.number_input(f'**{"Do You Feel Irritable towards baby & partner"}**', min_value=0, max_value=2)
   sleep = st.number_input(f'**{"Do You Have Trouble sleeping at night"}**', min_value=0, max_value=2)
   conc = st.number_input(f'**{"Do You Have Problems concentrating or making decision"}**', min_value=0, max_value=2)
   eat = st.number_input(f'**{"Do You Overeact or Experience loss of appetite"}**', min_value=0, max_value=2)
   guilt = st.number_input(f'**{"Do You Feel Guilt"}**', min_value=0, max_value=2)
   bond= st.number_input(f'**{"Do You Have Problems of bonding with baby"}**', min_value=0, max_value=2)
   suicide= st.number_input(f'**{"Do You Have Any Suicide attempt"}**', min_value=0, max_value=2)


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