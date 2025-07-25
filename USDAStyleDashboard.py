#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import plotly.express as px


# In[2]:


df = pd.read_csv(r"C:\RITIK\Business Data Analytics\Thesis\ml_results_for_dashboard.csv")
df['Date'] = pd.to_datetime(df['Date'])


# In[3]:


# App title
st.set_page_config(page_title="Market Analytics Dashboard", layout="wide")
st.title(" Market Analytics Dashboard - Equipment Forecasting")

# Sidebar filters
st.sidebar.header("🔎 Filter Options")
years = sorted(df['Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years[-1:])
equipment_types = df['Equipment Type'].unique()
selected_type = st.sidebar.selectbox("Select Equipment Type", sorted(equipment_types))

# Model selector
model_cols = [col for col in df.columns if col.startswith("Prediction_")]
selected_model = st.sidebar.selectbox("Select Prediction Model", model_cols)

# Filtered DataFrame
filtered_df = df[(df['Year'].isin(selected_years)) & (df['Equipment Type'] == selected_type)]

# Tabs
tab1, tab2, tab3 = st.tabs([" Overview", " Model Forecast", " Model Comparison"])

# -------- TAB 1: Overview --------
with tab1:
    st.subheader(" Units Sold vs Predictions")
    fig = px.line(filtered_df, x='Date', y=['Units sold', selected_model],
                  title=f"Actual vs {selected_model} for {selected_type}",
                  labels={"value": "Units", "Date": "Time"}, markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(" Equipment Sales Table")
    st.dataframe(filtered_df[["Date", "Equipment Type", "Units sold", selected_model]].sort_values("Date"))

    # Download filtered data
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(" Download Filtered Data", data=csv, file_name="filtered_market_data.csv")

# -------- TAB 2: Forecast Trends --------
with tab2:
    st.subheader(" Time Series Forecast")
    monthly_avg = filtered_df.groupby([filtered_df['Date'].dt.to_period('M')])[['Units sold', selected_model]].mean().reset_index()
    monthly_avg['Date'] = monthly_avg['Date'].dt.to_timestamp()
    fig2 = px.line(monthly_avg, x='Date', y=['Units sold', selected_model],
                   title="Monthly Average Forecast", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# -------- TAB 3: Model Comparison --------
with tab3:
    st.subheader(" MAE Comparison Between Models")
    from sklearn.metrics import mean_absolute_error

    mae_data = {
        'Model': [],
        'MAE': []
    }
    for model in model_cols:
        mae = mean_absolute_error(filtered_df['Units sold'], filtered_df[model])
        mae_data['Model'].append(model)
        mae_data['MAE'].append(mae)

    mae_df = pd.DataFrame(mae_data).sort_values(by='MAE')
    fig3 = px.bar(mae_df, x='Model', y='MAE', text='MAE', title="Mean Absolute Error by Model")
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(mae_df.set_index("Model"))

