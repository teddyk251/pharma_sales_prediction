import sys
sys.path.append("../scripts/")
sys.path.append("../dashboard/")

import streamlit as st
from multiapp import MultiApp
from dashboard import data_analytics_page, inference_page
st.set_page_config(page_title="Rossmann Pharmaceuticals Data Analytics", layout="wide")

app = MultiApp()


st.sidebar.markdown("""
# Rossmann Pharmaceuticals Sales Forecast

""")

# Add all your application here
app.add_app("Data Analystics", data_analytics_page.app)
app.add_app("Inference Page", inference_page.app)
# app.add_app("User Experience Analysis", user_experience_page.app)
# app.add_app("User Satisfaction Analysis", user_satisfaction_page.app)

# The main app
app.run()
