@"
# Predictive Forecasting of Care Load & Placement Demand

## 📌 Project Overview

This project focuses on analyzing and forecasting the care load of unaccompanied children in HHS (Health and Human Services) care.

The project uses historical operational data to understand trends, patterns, and relationships in the care system and develops multiple forecasting and machine learning models to predict future care demand.

The final project also includes an interactive Streamlit dashboard for exploring historical data, model results, feature importance, and future forecasts.

---

## 🎯 Objectives

- Understand and clean the historical HHS dataset.
- Perform exploratory data analysis (EDA).
- Identify trends, patterns, correlations, and missing dates.
- Engineer time-series and operational features.
- Establish baseline forecasting performance.
- Develop and evaluate statistical forecasting models.
- Develop machine learning regression models.
- Compare model performance using MAE, RMSE, and MAPE.
- Generate future care-load forecasts.
- Build an interactive dashboard for visualization and analysis.

---

## 📊 Dataset

The dataset contains historical information related to unaccompanied children in the HHS care system.

### Main Variables

- Date
- Children apprehended and placed in CBP custody
- Children in CBP custody
- Children transferred out of CBP custody
- Children in HHS Care
- Children discharged from HHS Care

### Target Variable

**Children in HHS Care**

The target variable represents the number of children currently in HHS care and is used for forecasting and prediction.

---

## 🔧 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Statsmodels
- Joblib
- Plotly
- Streamlit

---

## 🧹 Data Processing

The preprocessing stage includes:

- Loading the raw dataset.
- Converting date values into datetime format.
- Cleaning numeric values containing commas.
- Checking missing values.
- Checking duplicate records.
- Sorting observations chronologically.
- Identifying missing dates.
- Preparing the dataset for time-series analysis.

Processed datasets are stored in:

```text
data/processed/