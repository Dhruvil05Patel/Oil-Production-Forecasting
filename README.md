# Texas Crude Oil Production Forecasting 🛢️📈

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Time%20Series-lightgrey)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626)

## 📌 Project Overview
Crude oil production forecasting is a cornerstone of energy planning, underpinning decisions that range from strategic reserve management to state tax-revenue estimation. This repository contains a comprehensive comparative study of classical statistical and modern deep-learning time-series models applied to **monthly Texas crude oil production data** spanning from **January 1981 to January 2026** (541 observations, measured in thousands of barrels).

The forecasting task is framed as a supervised univariate time-series regression problem, navigating the complex non-linear dynamics and structural breaks introduced by the 2010s shale revolution and macroeconomic shocks.

## 🎯 Key Objectives
1. **Implement and Tune Models:** Deploy statistical (AR, MA, ARIMA, SARIMA) and deep-learning (RNN, GRU, LSTM) architectures on the production series.
2. **Comparative Evaluation:** Assess models on a held-out test set using RMSE, MAE, MAPE, and $R^2$.
3. **Probabilistic Forecasting:** Construct forecast intervals via adaptive conformal prediction on a Quantile Regression model and assess empirical coverage (PICP).
4. **Actionable Insights:** Translate point forecasts into recommendations for producers, regulators, midstream operators, and policymakers.

## 📊 Dataset
* **Source:** U.S. Energy Information Administration (EIA)
* **Series:** Texas Field Production of Crude Oil (`PET.MCRFPTX2.M`)
* **Frequency:** Monthly
* **Data Pre-processing:** The dataset undergoes 2nd-order differencing to achieve stationarity (confirmed via Augmented Dickey-Fuller tests) and Min-Max scaling to the `[-1, 1]` range for stable neural network training. Data is chronologically split into Train (70%), Validation (20%), and Test (10%) sets to prevent forward leakage.

## 🧠 Methodology
### 1. Statistical Models (Classical Econometrics)
* AutoRegressive **AR(3)**
* Moving Average **MA(11)**
* **ARIMA(1,2,2)**
* **SARIMA** (with explicitly modeled annual seasonality)

### 2. Deep Learning Models (Sequence Modeling)
* **Simple RNN** (Look-back: 11)
* **GRU** (Look-back: 10)
* **LSTM** (Look-back: 11)
* *Note: Hyperparameters (hidden units, look-back windows) were optimized using `keras-tuner`.*

### 3. Probabilistic Forecasting
* **Statistical Quantile Regression** with adaptive conformal calibration using a 12-month lag depth and a 6-month rolling volatility window.

## 🏆 Key Findings & Results
* **Best Overall Accuracy:** **SARIMA** achieved the highest test-set point accuracy (**MAPE = 1.68%**, $R^2 = 0.8455$), demonstrating that explicit seasonal differencing is highly effective for this specific series.
* **Top Deep Learning Model:** **GRU** led the neural network family (**MAPE = 2.55%**), effectively capturing regime transitions and seasonality without the parameter overhead of LSTM.
* **Uncertainty Quantification:** The adaptive Quantile Regression model produced the most efficient probabilistic intervals (mean width = 9,580 thousand barrels) while maintaining near-target coverage (**PICP = 78.18%** against an 80% target).

## 📂 Repository Structure
```text
├── data/
│   └── Texas_dataset.csv          # Raw EIA dataset
├── notebooks/
│   └── final_AFM_project.ipynb    # Main Jupyter Notebook with all code & analysis
├── docs/
│   ├── main_AFM.pdf           # Detailed project report
│   └── main_AFM.tex           # LaTeX source code for the report
├── images/                        # Visualizations generated during EDA and Evaluation
└── README.md                      # Project documentation
