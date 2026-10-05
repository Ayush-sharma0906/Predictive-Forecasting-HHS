import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path
import joblib


st.write("Current Working Directory:", os.getcwd())

st.set_page_config(
    page_title="HHS Care Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

h1{
    color:#0F62FE;
}

div[data-testid="stMetric"]{
    background-color:white;
    padding:18px;
    border-radius:12px;
    border:1px solid #dddddd;
    box-shadow:0 2px 8px rgba(0,0,0,.08);
}

</style>
""", unsafe_allow_html=True)

# ------------------files loading---------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "HHS_Feature_Engineered.csv"

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

forecast_df = pd.read_csv(BASE_DIR / "data" / "forecast" / "future_forecast.csv")
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"],errors="coerce")



# ------------------model loading----------------

gb_model = joblib.load(
    BASE_DIR / "models" / "gradient_boosting_model.pkl"
)

# comperison metrics 

baseline_mae = 235.26
baseline_rmse = 285.70
baseline_mape = 11.12

arima_mae = 235.26
arima_rmse = 285.70
arima_mape = 11.12

sarima_mae = 643.9922792307239
sarima_rmse = 870.7843898148097
sarima_mape = 28.039218661417742

hw_mae = 843.79
hw_rmse = 939.78
hw_mape = 38.32

rf_mae = 67.36
rf_rmse = 89.08
rf_mape = 3.12

gb_mae = 58.79
gb_rmse = 79.32
gb_mape = 2.70

# features selected for the model

feature_cols = [
    "Children apprehended and placed in CBP custody*",
    "Children in CBP custody",
    "Children transferred out of CBP custody",
    "Children discharged from HHS Care",
    "Month",
    "Day",
    "Year",
    "Quarter",
    "Day_of_Week",
    "Is_Weekend",
    "HHS_Lag_1",
    "HHS_Lag_7",
    "HHS_Lag_14",
    "Rolling_Mean_7",
    "Rolling_Mean_14",
    "Rolling_STD_7",
    "Net_Pressure"
]

# Sidebar
st.sidebar.image(
    "https://streamlit.io/images/brand/streamlit-mark-color.png",
    width=90
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Navigation",
    (
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Historical Trend",
        "🔮 Future Forecast",
        "🏆 Model Comparison",
        "⭐ Feature Importance",
        "ℹ️ About"
    ),
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.success("Best Model\nGradient Boosting")

st.sidebar.info("Forecast Horizon\n30 Days")

# ---------------- HOME ----------------

if page == "🏠 Home":

    st.title("📈 HHS Care Demand Forecasting Dashboard")

    st.success("""
    🚀 Welcome!

    This dashboard predicts future HHS Care demand using
    Machine Learning and Time Series Forecasting models.

    The objective is to support operational planning
    through accurate forecasting.
    """)

    st.caption(
        "Machine Learning-Based Forecasting System for Operational Planning"
    )

    st.markdown("""
    Welcome to the **Predictive Forecasting of HHS Care Demand** Dashboard.

    This application forecasts future HHS Care demand using Machine Learning
    and Time Series Forecasting models.
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📊 Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "🧮 Features",
        len(feature_cols)
    )

    col3.metric(
        "🏆 Best Model",
        "Gradient Boosting"
    )

    col4.metric(
        "🎯 Best MAPE",
        "2.70%"
    )

    st.subheader("📌 Dashboard Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Dataset Period:**")
        st.write(f"{df['Date'].min().date()} → {df['Date'].max().date()}")

        st.write(f"**Forecast Days:**")
        st.write("30")

    with col2:
        st.write("**Target Variable:**")
        st.write("Children in HHS Care")

        st.write("**Forecasting Technique:**")
        st.write("Gradient Boosting")

    st.divider()

    st.subheader("📌 Project Summary")

    st.info("""
    This dashboard provides historical analysis and 30-day forecasting of
    Children in HHS Care.

    Models developed:

    • Baseline

    • ARIMA

    • SARIMA

    • Holt-Winters

    • Random Forest

    • Gradient Boosting ⭐
    """)
    
    st.subheader("📈 Dataset Snapshot")

    st.write(f"**Date Range:** {df['Date'].min().date()} → {df['Date'].max().date()}")

    st.write(f"**Forecast Horizon:** 30 Days")

    st.write(f"**Total Forecast Records:** {len(forecast_df)}")

    st.write(f"**Target Variable:** Children in HHS Care")

    with st.expander("📌 View Dataset Preview"):

        st.dataframe(df.head(), width="stretch")

    with st.expander("📊 Statistical Summary"):

        summary = df.describe(include="number")

        st.dataframe(summary, width="stretch")

# ---------------- DATASET ----------------

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Total Features", df.shape[1])

    with col3:
        st.metric(
            "Date Range",
            f"{df['Date'].min().date()} → {df['Date'].max().date()}"
        )

    st.divider()

    st.subheader("First 10 Records")

    st.dataframe(df.head(10), width="stretch")

    st.divider()

    st.subheader("Statistical Summary")

    summary = df.describe(include="number")

    st.dataframe(summary, width="stretch")

# ---------------- HISTORICAL ----------------

elif page == "📈 Historical Trend":

    st.title("📈 Historical Trend")

    st.markdown(
        "Explore the historical trend of **Children in HHS Care** over time."
    )

    fig = px.line(
        df,
        x="Date",
        y="Children in HHS Care",
        title="Historical HHS Care Trend"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Children in HHS Care",
        hovermode="x unified"
    )

    st.plotly_chart(fig, width="stretch")

# ---------------- FORECAST ----------------

elif page == "🔮 Future Forecast":

    st.title("🔮 30-Day Future Forecast")

    st.markdown(
        "Forecast generated using the **Gradient Boosting Model**."
    )

    import plotly.graph_objects as go

    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Children in HHS Care"],
            mode="lines",
            name="Historical"
        )
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted_HHS_Care"],
            mode="lines",
            name="Forecast",
            line=dict(dash="dash")
        )
    )

    # Forecast Start
    fig.add_vline(
        x=df["Date"].max(),
        line_dash="dot",
        line_color="red"
    )

    fig.update_layout(
        title="Historical vs Future Forecast",
        xaxis_title="Date",
        yaxis_title="Children in HHS Care",
        hovermode="x unified"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Forecast Data")

    st.dataframe(
        forecast_df,
        width="stretch"
    )


    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name="future_forecast.csv",
        mime="text/csv"
    )

# ---------------- MODEL ----------------

elif page == "🏆 Model Comparison":

    st.title("🏆 Model Performance Comparison")

    comparison_df = pd.DataFrame({

        "Model":[
            "Baseline",
            "ARIMA",
            "SARIMA",
            "Random Forest",
            "Gradient Boosting"
        ],

        "MAE":[
            baseline_mae,
            arima_mae,
            sarima_mae,
            rf_mae,
            gb_mae
        ],

        "RMSE":[
            baseline_rmse,
            arima_rmse,
            sarima_rmse,
            rf_rmse,
            gb_rmse
        ],

        "MAPE":[
            baseline_mape,
            arima_mape,
            sarima_mape,
            rf_mape,
            gb_mape
        ]

    })
    st.subheader("📊 Model Performance")

    st.dataframe(comparison_df, width="stretch")
    fig = px.bar(
        comparison_df,
        x="Model",
        y="MAPE",
        color="Model",
        text="MAE",
        title="Model Comparison (MAE)"
    )



    st.plotly_chart(
        fig,
        width="stretch"
    )
    best_model = comparison_df.loc[
        comparison_df["MAPE"].idxmin()
    ]

    fig = px.bar(
        comparison_df,
        x="Model",
        y="RMSE",
        color="Model",
        text="RMSE",
        title="Model Comparison (RMSE)"
    )

    st.plotly_chart(fig, width="stretch")

    fig = px.bar(
        comparison_df,
        x="Model",
        y="MAPE",
        color="Model",
        text="MAPE",
        title="Model Comparison (MAPE)"
    )

    st.plotly_chart(fig, width="stretch")

    best_model = comparison_df.loc[
    comparison_df["MAPE"].idxmin()
]

    st.success(
        f"""
        🏆 Best Model Selected

        **Model:** {best_model['Model']}

        **MAPE:** {best_model['MAPE']:.2f}%

        This model achieved the lowest forecasting error and was selected for generating the 30-day forecast.
        """
    )

    st.info("""
        ### 📌 Why was Gradient Boosting selected?

        - Lowest MAE
        - Lowest RMSE
        - Lowest MAPE
        - Captured non-linear relationships effectively
        - Produced the most accurate 30-day forecasts
    """)

# ---------------- FEATURE ----------------

elif page == "⭐ Feature Importance":

    st.title("⭐ Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": gb_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Gradient Boosting Feature Importance",
        color="Importance"
    )

    fig.update_layout(
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader("Feature Importance Table")

    st.dataframe(
        importance_df,
        width="stretch"
    )

    st.success("🏆 Top 5 Most Important Features")

    st.table(
        importance_df.head(5)
    )

# ---------------- ABOUT ----------------

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    # Predictive Forecasting of HHS Care Demand

    ## 🎯 Project Objective

    This project aims to forecast the number of children expected to remain in
    HHS Care using historical operational data and machine learning techniques.

    The objective is to assist decision-makers in resource planning, staffing,
    and capacity management through accurate short-term forecasting.

    ---

    ## 📂 Dataset

    The dataset contains historical records including:

    - Children apprehended and placed in CBP custody
    - Children in CBP custody
    - Children transferred out of CBP custody
    - Children discharged from HHS Care
    - Children currently in HHS Care

    ---

    ## 🤖 Models Implemented

    ✅ Baseline Forecast

    ✅ ARIMA

    ✅ SARIMA

    ✅ Holt-Winters

    ✅ Random Forest Regressor

    ✅ Gradient Boosting Regressor

    ---

    ## 🏆 Best Performing Model

    Gradient Boosting Regressor

    Lowest forecasting error based on:

    - MAE
    - RMSE
    - MAPE

    ---

    ## 🛠 Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Statsmodels
    - Plotly
    - Streamlit

    ---

    ## 📈 Dashboard Features

    - Dataset Overview
    - Historical Trend Analysis
    - 30-Day Future Forecast
    - Model Performance Comparison
    - Feature Importance Analysis
    - Forecast CSV Download

    ---

    Developed as part of a Data Science Forecasting Project.
    """)

    st.divider()

    st.caption(
        "Developed with ❤️ using Python, Streamlit and Machine Learning"
    )

st.divider()

st.caption(
    "© 2026 Predictive Forecasting Dashboard | Built with Python • Streamlit • Plotly • Scikit-learn"
)

print("Dashboard loaded successfully.")