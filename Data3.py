import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.preprocessing import StandardScaler

# Streamlit App Setup
st.set_page_config(page_title="Data Cleaning & Visualization App", layout="wide")
st.title("📊 Data Cleaning & Visualization Tool")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Initial Data Overview")
    st.write(df.head())

    # Handle Missing Values
    def handle_missing_values(df):
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':  # Categorical Data
                    df[col] = df[col].fillna(df[col].mode()[0])  # Fill with most common value
                elif np.issubdtype(df[col].dtype, np.number):  # Numeric Data
                    if df[col].skew() > 1 or df[col].skew() < -1:
                        df[col] = df[col].fillna(df[col].median())  # Use median if skewed
                    else:
                        df[col] = df[col].fillna(df[col].mean())  # Use mean if normal distribution
        return df

    df = handle_missing_values(df)

    # Remove Duplicates (Entire Row)
    df.drop_duplicates(inplace=True)

    # Standardize Categorical Values
    def standardize_text_columns(df):
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip().str.lower().str.capitalize()
        return df

    df = standardize_text_columns(df)

    # Convert Dates Properly
    def convert_to_datetime(df):
        for col in df.columns:
            if df[col].dtype == 'object':  # Ensure only string-like columns are processed
                try:
                    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
                except Exception:
                    pass
        return df

    df = convert_to_datetime(df)

    # Remove Outliers Using IQR
    def remove_outliers(df):
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan  # Mark outliers as NaN
        df = handle_missing_values(df)  # Refill NaNs
        return df

    df = remove_outliers(df)

    # Convert Categorical Data to One-Hot Encoding (Better than LabelEncoder)
    df = pd.get_dummies(df, drop_first=True)

    # Scale Numerical Data
    def scale_numeric_data(df):
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    df = scale_numeric_data(df)

    # Display Cleaned Data
    st.subheader("✅ Cleaned Data Preview")
    st.write(df.head())

    # Data Visualization
    st.subheader("📈 Data Visualizations")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    if len(numeric_cols) > 1:
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(plt)

        st.write("### Scatter Plot with Trendline")
        selected_x = st.selectbox("Select X-axis for Scatter Plot", numeric_cols)
        selected_y = st.selectbox("Select Y-axis for Scatter Plot", numeric_cols)
        plt.figure(figsize=(8, 5))
        sns.regplot(x=df[selected_x], y=df[selected_y], scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
        st.pyplot(plt)

        st.write("### Bar Graph with Trendline")
        selected_bar_x = st.selectbox("Select X-axis for Bar Graph", numeric_cols, key='bar_x')
        selected_bar_y = st.selectbox("Select Y-axis for Bar Graph", numeric_cols, key='bar_y')
        plt.figure(figsize=(8, 5))
        sns.barplot(x=df[selected_bar_x], y=df[selected_bar_y])
        sns.lineplot(x=df[selected_bar_x], y=df[selected_bar_y], color='red', label='Trendline')
        st.pyplot(plt)

    # Convert cleaned DataFrame to CSV for download
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    st.download_button(label="Download Cleaned CSV", data=output, file_name="cleaned_data.csv", mime="text/csv")
