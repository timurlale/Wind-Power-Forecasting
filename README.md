# Wind Power Forecasting using EDOA-Optimized TCN-BiGRU-MHA Model

This repository contains the **complete implementation and dataset** used in the study:

**"Enhanced Dhole Optimization Algorithm for Hyperparameter Tuning of TCN-BiGRU-MHA Hybrid Architecture for Wind Power Forecasting"**

---

## 📊 Dataset Description

The dataset used in this study is constructed by integrating:

- **SCADA data** from a real wind turbine  
- **Meteorological data** obtained from NASA MERRA-2  

### 📍 Data Source

- Wind turbine: **Nordex N117/3600**
- Location: Yalova, Türkiye  
- Coordinates: 40.59528° N, 28.99035° E  
- Time period: **01 January 2018 – 31 December 2018**

The final dataset consists of:

- **8,760 hourly samples**
- Originally:
  - SCADA data: 10-minute resolution (aggregated to hourly)
  - Meteorological data: hourly

---

### 🎯 Target Variable

- **LV Active Power (kW)**  
  (Real-time generated power output of the wind turbine)

---

### 🔢 Input Features

The model uses 8 input features:

1. Temperature (°C)
2. Total precipitation (mm/h)
3. Air density (kg/m³)
4. Solar radiation (W/m²)
5. Cloud cover (0–1)
6. Wind speed (m/s)
7. Wind direction (°)
8. Wind direction bin

---

### ⚙️ Preprocessing Steps

- Time synchronization (SCADA + MERRA-2)
- Missing value imputation (moving average + interpolation)
- Z-score normalization
- Feature-target scaling (separate StandardScaler)
- Train/Validation/Test split:
  - 70% / 10% / 20%

---

## 🧠 Proposed Model

The proposed hybrid deep learning architecture consists of:

- **TCN (Temporal Convolutional Network)**  
  → Multi-scale temporal feature extraction  

- **BiGRU (Bidirectional GRU)**  
  → Bidirectional sequence learning  

- **Multi-Head Attention (MHA)**  
  → Dynamic feature weighting  

---

## ⚙️ Optimization Algorithm

Hyperparameters are optimized using:

### Enhanced Dhole Optimization Algorithm (EDOA)

Key improvements:

- Quantum-inspired mutation  
- Differential Evolution (DE) integration  
- Adaptive spiral search  
- Elite archive mechanism  
- Stagnation recovery strategy  

---

---

## 📌 Code Description

### 🔹 1. tcn_bigru_mha.py
- Core model implementation
- Performs:
  - Data loading & preprocessing
  - Model training (TCN-BiGRU-MHA)
  - Evaluation (MAE, RMSE, R², WAPE)
- Uses 70/10/20 data split :contentReference[oaicite:0]{index=0}

---

### 🔹 2. edoa_tcn_gru_mha.py
- Implements Enhanced Dhole Optimization Algorithm (EDOA)
- Optimizes 10 hyperparameters
- Uses validation R² as fitness function :contentReference[oaicite:1]{index=1}

---

### 🔹 3. Seasonal_Generalization_test.py
- Cross-season validation:
  - Train on 3 seasons
  - Test on remaining season
- Evaluates generalization capability :contentReference[oaicite:2]{index=2}

---

### 🔹 4. Missing-Sensor-Robustness-Test.py
- Tests model performance under missing feature scenarios

---

### 🔹 5. Noise-Robustness-Test.py
- Evaluates robustness against noisy input data

---

## 🔁 Reproducibility

Install dependencies:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
Run base model:

python tcn_bigru_mha.py

Run optimization:

python edoa_tcn_gru_mha.py

📈 Evaluation Metrics

MAE (Mean Absolute Error)

RMSE (Root Mean Square Error)

R² Score

WAPE (%)

📬 Author

Timur Lale
Batman University
Department of Electrical and Electronics Engineering

## 📂 Repository Structure
