import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import statsmodels.api as sm

# Streamlit App Setup
st.set_page_config(page_title="Data Cleaning & Visualization App", layout="wide")
st.title("📊 Data Cleaning & Visualization Tool")

# Custom CSS
st.markdown("""
    <style>
        .stApp {background-color: #f8f9fa;}
        .css-1d391kg {max-width: 900px; margin: auto;}
        .css-18e3th9 {text-align: center;}
        .stButton>button {background-color: #007bff; color: white;}
    </style>
""", unsafe_allow_html=True)

# Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Handle Duplicates Per Column
def handle_column_duplicates(df):
    for col in df.columns:
        if df[col].duplicated().any():
            df[col] = df[col].drop_duplicates().reset_index(drop=True)
    return df

# Handle Missing Values
def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':  # Categorical Data
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif np.issubdtype(df[col].dtype, np.number):  # Numeric Data
                df[col].fillna(df[col].median(), inplace=True)
            elif np.issubdtype(df[col].dtype, np.datetime64):  # Date Data
                df[col].fillna(method='ffill', inplace=True)  # Forward Fill
    return df

# Convert Date Columns
def convert_dates(df):
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    return df

# Scale Numeric Data
def scale_numeric_data(df):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

# Scatter Plot with Trendline
def scatter_plot_with_trendline(df, x_col, y_col):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.5)
    
    # Trendline
    X = sm.add_constant(df[x_col])
    model = sm.OLS(df[y_col], X).fit()
    df['trendline'] = model.predict(X)
    
    plt.plot(df[x_col], df['trendline'], color='red', linestyle="dashed")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    st.pyplot(plt)

# Bar Chart with Trendline
def bar_chart_with_trendline(df, x_col, y_col):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=df[x_col], y=df[y_col], alpha=0.7)
    
    # Trendline
    X = sm.add_constant(df[x_col])
    model = sm.OLS(df[y_col], X).fit()
    df['trendline'] = model.predict(X)
    
    plt.plot(df[x_col], df['trendline'], color='red', linestyle="dashed")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Bar Chart of {x_col} vs {y_col}')
    st.pyplot(plt)

# Heatmap Correlation
def heatmap_correlation(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.subheader("📊 Initial Data Overview")
    st.write(df.head())
    
    # Data Cleaning
    df = handle_column_duplicates(df)
    df = handle_missing_values(df)
    df = convert_dates(df)
    df = scale_numeric_data(df)

    st.subheader("✅ Cleaned Data Preview")
    st.write(df.head())
    
    # Data Visualization
    st.subheader("📈 Data Visualizations")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 1:
        st.write("### Correlation Heatmap")
        heatmap_correlation(df)
        
        st.write("### Scatter Plot with Trendline")
        selected_x = st.selectbox("Select X-axis for Scatter Plot", numeric_cols)
        selected_y = st.selectbox("Select Y-axis for Scatter Plot", numeric_cols)
        scatter_plot_with_trendline(df, selected_x, selected_y)
        
        st.write("### Bar Graph with Trendline")
        selected_bar_x = st.selectbox("Select X-axis for Bar Graph", numeric_cols, key='bar_x')
        selected_bar_y = st.selectbox("Select Y-axis for Bar Graph", numeric_cols, key='bar_y')
        bar_chart_with_trendline(df, selected_bar_x, selected_bar_y)
    
    # Convert cleaned DataFrame to CSV for download
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    st.download_button(label="Download Cleaned CSV", data=output, file_name="cleaned_data.csv", mime="text/csv")