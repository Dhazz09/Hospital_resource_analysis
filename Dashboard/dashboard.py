import streamlit as st
import requests
import pandas as pd

API = "http://backend:8000"

st.title("🏥 Hospital Resource Optimisation Dashboard")

if st.button("Load Historical Data"):
    r = requests.get(f"{API}/historical")
    df = pd.DataFrame(r.json())
    st.dataframe(df)
    if not df.empty:
        st.line_chart(df.set_index("date")[["bed_usage","oxygen_usage","staff_on_duty"]])

days = st.slider("Forecast Days Ahead", 3, 14, 7)
if st.button("Get Forecast"):
    r = requests.get(f"{API}/predict", params={"days": days})
    preds = r.json()["predictions"]
    st.json(preds)
