# Electric Power Consumption Forecasting

## 📌 Project Overview
This project focuses on **Time Series Forecasting** to predict future electric power consumption based on historical data. By analyzing over 2 million records of household energy usage, this project compares the performance of traditional statistical models (ARIMA, SARIMA) against modern Deep Learning architectures (RNN, LSTM, Bi-LSTM).

The goal is to identify the most accurate model for forecasting **Global Active Power** to aid in efficient energy management and demand planning.

## 📂 Dataset
* **Source:** [Individual Household Electric Power Consumption (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
* **Input File:** `ElectricityC.zip` (Contains CSV data)
* **Size:** ~2 million rows (minute-level resolution)
* **Key Features:**
  * `Global_active_power`: Household global minute-averaged active power (in kilowatt).
  * `Global_reactive_power`: Household global minute-averaged reactive power.
  * `Voltage`: Minute-averaged voltage (in volt).
  * `Sub_metering_1/2/3`: Energy sub-metering No. 1, 2, and 3 (in watt-hour).

## 🛠️ Technologies Used
* **Language:** Python 3.x
* **Libraries:**
  * **Data Manipulation:** `pandas`, `numpy`
  * **Visualization:** `matplotlib`, `seaborn`
  * **Statistical Modeling:** `statsmodels` (ARIMA, SARIMA, ADF Test, Seasonal Decompose)
  * **Deep Learning:** `tensorflow`, `keras` (LSTM, RNN, Bi-LSTM)
  * **Evaluation:** `scikit-learn` (Metrics)

## ⚙️ Project Workflow

### 1. Data Preprocessing & Cleaning
* **Handling Non-Numeric Data:** Converted `Global_active_power` to numeric, coercing errors to handle special characters (e.g., '?').
* **Missing Values:** Identified ~1.25% missing data and dropped these rows to ensure time series continuity.
* **Resampling:** Aggregated minute-level data to **Weekly** frequency for statistical modeling and **Hourly** frequency for Deep Learning to reduce noise and computational load.

### 2. Exploratory Data Analysis (EDA)
* **Decomposition:** Decomposed the time series into Trend, Seasonality, and Residuals using `seasonal_decompose`. Identified a clear **yearly seasonal pattern**.
* **Stationarity Check:** Performed the **Augmented Dickey-Fuller (ADF) Test**.
  * Result: p-value < 0.05, confirming the data is stationary and suitable for modeling.
* **Correlation Analysis:** Analyzed ACF (Auto-Correlation) and PACF (Partial Auto-Correlation) plots to determine optimal lag values for ARIMA.

### 3. Model Implementation
The project implements two distinct modeling approaches:

#### **Phase A: Statistical Models**
* **ARIMA (AutoRegressive Integrated Moving Average):** Used as a baseline model.
* **SARIMA (Seasonal ARIMA):** Implemented to specifically address the yearly seasonality observed during EDA.
* **Key Finding:** SARIMA significantly outperformed the standard ARIMA model by effectively capturing the seasonal components.

#### **Phase B: Deep Learning Models**
* **Simple RNN:** Basic Recurrent Neural Network used as a deep learning baseline.
* **LSTM (Long Short-Term Memory):** Implemented to capture long-term dependencies and mitigate the vanishing gradient problem common in simple RNNs.
* **Bi-LSTM (Bidirectional LSTM):** Tested to see if processing data in both directions (past-to-future and future-to-past) improved accuracy.
* **Key Finding:** The **LSTM model** achieved the lowest Mean Absolute Percentage Error (MAPE), outperforming both the Simple RNN and Bi-LSTM models.

## 📊 Results & Evaluation
Models were evaluated using **MAPE (Mean Absolute Percentage Error)** and **RMSE (Root Mean Squared Error)** to ensure robust performance comparison.

| Model Category | Best Model | Key Observation |
| :--- | :--- | :--- |
| **Statistical** | **SARIMA** | Superior to ARIMA due to handling of yearly seasonality. |
| **Deep Learning** | **BiLSTM** | Best overall accuracy; handled non-linear patterns effectively without overfitting. |

*Residual analysis was performed on the final models to ensure errors were random and normally distributed, validating the model's reliability.*
