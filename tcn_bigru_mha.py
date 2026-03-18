"""
TCN-BiGRU-MHA Model Training and Evaluation
Author: Advanced Forecasting System
Date: 2025
"""

#%% ========================================================================
#   USER CONFIGURATION - DOSYA İSİMLERİ
#  ========================================================================

# 🔧 KULLANICI AYARLARI - İSTEDİĞİNİZ GİBİ DEĞİŞTİREBİLİRSİNİZ
OUTPUT_FIGURE_NAME = 'EDOA_Test_Set_Prediction_Analysis_330dpi.tiff'  # Şekil dosya adı
OUTPUT_EXCEL_NAME = 'EDOA_Model_Performance_Metrics.xlsx'              # Excel dosya adı

#%% ========================================================================
#   IMPORTS
#  ========================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib font parameters for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 11

# GPU ayarları (varsa)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU bulundu: {len(gpus)} adet\n")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️  GPU bulunamadı - CPU ile çalışacak\n")

#%% ========================================================================
#   1. VERİ YÜKLEME VE ÖN İŞLEME
#  ========================================================================

print("╔════════════════════════════════════════════════════════════╗")
print("║        TCN-BiGRU-MHA MODEL TRAINING & EVALUATION           ║")
print("╚════════════════════════════════════════════════════════════╝\n")

print("📊 VERİ YÜKLEME...")

# Load data
data = pd.read_csv('Merged_Dataset.csv')

# Features and target
X = data.iloc[:, [1, 2, 3, 4, 5, 7, 9, 10]].values  # 8 features
y = data.iloc[:, 6].values  # LV ActivePower (kW)

print(f"✓ Veri Yüklendi: {X.shape[0]} örnek, {X.shape[1]} özellik")
print(f"  ├─ Min Power: {y.min():.2f} kW")
print(f"  ├─ Max Power: {y.max():.2f} kW")
print(f"  ├─ Mean Power: {y.mean():.2f} kW")
print(f"  └─ Zero/Low values (<10 kW): {np.sum(y < 10)} ({np.sum(y < 10)/len(y)*100:.1f}%)\n")

# Normalization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Train/Val/Test split (70/10/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_normalized, y_normalized, test_size=0.20, random_state=42, shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, shuffle=True
)

print(f"✓ Veri Bölme:")
print(f"  ├─ Train: {len(X_train)} (70%)")
print(f"  ├─ Val:   {len(X_val)} (10%)")
print(f"  └─ Test:  {len(X_test)} (20%)\n")

# Reshape for sequence input
X_train = X_train.reshape(-1, 1, 8)
X_val = X_val.reshape(-1, 1, 8)
X_test = X_test.reshape(-1, 1, 8)

#%% ========================================================================
#   2. MODEL ARCHITECTURE
#  ========================================================================

print("🏗️  MODEL OLUŞTURULUYOR...\n")

# Hyperparameters
TCN_FILTERS = 128
TCN_KERNEL_SIZE = 3
TCN_DILATION_BASE = 4
GRU_UNITS = 64
ATTENTION_HEADS = 4
ATTENTION_KEY_SIZE = 128
FF_UNITS_BASE = 512
DROPOUT_ATTENTION = 0.1306222024280199
LEARNING_RATE = 0.004164094828483213
BATCH_SIZE = 128
EPOCHS = 100

# Auto-calculated parameters
tcn_filters_l2 = TCN_FILTERS * 2
tcn_filters_l3 = TCN_FILTERS * 4
gru_units_l2 = GRU_UNITS // 2
ff_units_l2 = FF_UNITS_BASE // 2
ff_units_l3 = FF_UNITS_BASE // 4

# Input
inputs = keras.Input(shape=(1, 8), name='input')

# TCN BLOCKS
x = layers.Conv1D(TCN_FILTERS, TCN_KERNEL_SIZE, padding='same', 
                  dilation_rate=TCN_DILATION_BASE, name='tcn_conv_1')(inputs)
x = layers.BatchNormalization(name='tcn_bn_1')(x)
x = layers.ReLU(name='tcn_relu_1')(x)

x = layers.Conv1D(tcn_filters_l2, TCN_KERNEL_SIZE, padding='same', 
                  dilation_rate=TCN_DILATION_BASE*2, name='tcn_conv_2')(x)
x = layers.BatchNormalization(name='tcn_bn_2')(x)
x = layers.ReLU(name='tcn_relu_2')(x)

x = layers.Conv1D(tcn_filters_l3, TCN_KERNEL_SIZE, padding='same', 
                  dilation_rate=TCN_DILATION_BASE*4, name='tcn_conv_3')(x)
x = layers.BatchNormalization(name='tcn_bn_3')(x)
x = layers.ReLU(name='tcn_relu_3')(x)

# GRU LAYERS
x = layers.Bidirectional(layers.GRU(GRU_UNITS, return_sequences=True), 
                         name='bigru_1')(x)
x = layers.Bidirectional(layers.GRU(gru_units_l2, return_sequences=True), 
                         name='bigru_2')(x)

# MULTIHEAD ATTENTION
x = layers.MultiHeadAttention(
    num_heads=ATTENTION_HEADS, 
    key_dim=ATTENTION_KEY_SIZE,
    name='multihead_attention'
)(x, x)
x = layers.Dropout(DROPOUT_ATTENTION, name='attention_dropout')(x)
x = layers.LayerNormalization(name='attention_norm')(x)

# GLOBAL POOLING
x = layers.GlobalAveragePooling1D(name='gap')(x)

# FEED-FORWARD NETWORK
x = layers.Dense(FF_UNITS_BASE, name='fc_1')(x)
x = layers.ReLU(name='fc_relu_1')(x)
x = layers.BatchNormalization(name='fc_bn_1')(x)

x = layers.Dense(ff_units_l2, name='fc_2')(x)
x = layers.ReLU(name='fc_relu_2')(x)
x = layers.BatchNormalization(name='fc_bn_2')(x)

x = layers.Dense(ff_units_l3, name='fc_3')(x)
x = layers.ReLU(name='fc_relu_3')(x)

# Output
outputs = layers.Dense(1, name='output')(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='TCN_BiGRU_MHA')

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)

print("✓ Model Oluşturuldu\n")
print(f"📋 Model Parametreleri:")
print(f"  ├─ TCN Filters:        {TCN_FILTERS}")
print(f"  ├─ TCN Kernel Size:    {TCN_KERNEL_SIZE}")
print(f"  ├─ TCN Dilation Base:  {TCN_DILATION_BASE}")
print(f"  ├─ GRU Units:          {GRU_UNITS}")
print(f"  ├─ Attention Heads:    {ATTENTION_HEADS}")
print(f"  ├─ Attention Key Size: {ATTENTION_KEY_SIZE}")
print(f"  ├─ FF Units Base:      {FF_UNITS_BASE}")
print(f"  ├─ Dropout:            {DROPOUT_ATTENTION}")
print(f"  ├─ Learning Rate:      {LEARNING_RATE}")
print(f"  └─ Batch Size:         {BATCH_SIZE}\n")

#%% ========================================================================
#   3. MODEL TRAINING
#  ========================================================================

print("🚀 MODEL EĞİTİMİ BAŞLIYOR...\n")

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

print("\n✓ Eğitim Tamamlandı!\n")

#%% ========================================================================
#   4. MODEL EVALUATION - MAPE KALDIRILDI
#  ========================================================================

print("╔════════════════════════════════════════════════════════════╗")
print("║                  MODEL PERFORMANSI                         ║")
print("╚════════════════════════════════════════════════════════════╝\n")

# Predictions
y_train_pred_norm = model.predict(X_train, verbose=0).ravel()
y_val_pred_norm = model.predict(X_val, verbose=0).ravel()
y_test_pred_norm = model.predict(X_test, verbose=0).ravel()

# Denormalize
y_train_pred = scaler_y.inverse_transform(y_train_pred_norm.reshape(-1, 1)).ravel()
y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()

y_val_pred = scaler_y.inverse_transform(y_val_pred_norm.reshape(-1, 1)).ravel()
y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()

y_test_pred = scaler_y.inverse_transform(y_test_pred_norm.reshape(-1, 1)).ravel()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# ========================================================================
# METRICS FONKSIYONU - SADECE MAE, RMSE, R², WAPE
# ========================================================================
def calculate_metrics(y_true, y_pred):
    """
    Calculate metrics: MAE, RMSE, R², WAPE
    
    Parameters:
    -----------
    y_true : array
        Actual values
    y_pred : array
        Predicted values
    """
    # MAE, RMSE, R² - tüm değerler için
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    # WAPE - tüm değerler için
    wape = np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100
    
    return mae, rmse, r2, wape

# Calculate metrics
mae_train, rmse_train, r2_train, wape_train = calculate_metrics(y_train_actual, y_train_pred)
mae_val, rmse_val, r2_val, wape_val = calculate_metrics(y_val_actual, y_val_pred)
mae_test, rmse_test, r2_test, wape_test = calculate_metrics(y_test_actual, y_test_pred)

# Print results
print("┌─────────────────────────────────────────┐")
print("│         TRAINING SET                    │")
print("├─────────────────────────────────────────┤")
print(f"│  MAE:   {mae_train:8.2f} kW                    │")
print(f"│  RMSE:  {rmse_train:8.2f} kW                    │")
print(f"│  R²:    {r2_train:8.6f}                     │")
print(f"│  WAPE:  {wape_train:8.2f} %                    │")
print("└─────────────────────────────────────────┘\n")

print("┌─────────────────────────────────────────┐")
print("│       VALIDATION SET                    │")
print("├─────────────────────────────────────────┤")
print(f"│  MAE:   {mae_val:8.2f} kW                    │")
print(f"│  RMSE:  {rmse_val:8.2f} kW                    │")
print(f"│  R²:    {r2_val:8.6f}                     │")
print(f"│  WAPE:  {wape_val:8.2f} %                    │")
print("└─────────────────────────────────────────┘\n")

print("┌─────────────────────────────────────────┐")
print("│          TEST SET                       │")
print("├─────────────────────────────────────────┤")
print(f"│  MAE:   {mae_test:8.2f} kW                    │")
print(f"│  RMSE:  {rmse_test:8.2f} kW                    │")
print(f"│  R²:    {r2_test:8.6f}                     │")
print(f"│  WAPE:  {wape_test:8.2f} %                    │")
print("└─────────────────────────────────────────┘\n")

# Overfitting Analysis
overfitting_gap = abs(r2_train - r2_test)
print("┌─────────────────────────────────────────┐")
print("│      OVERFITTING ANALYSIS               │")
print("├─────────────────────────────────────────┤")
print(f"│  Train R²:  {r2_train:8.6f}                │")
print(f"│  Test R²:   {r2_test:8.6f}                │")
print(f"│  Gap:       {overfitting_gap:8.6f}                │")
if overfitting_gap < 0.05:
    status = "✓ EXCELLENT"
elif overfitting_gap < 0.10:
    status = "✓ GOOD"
elif overfitting_gap < 0.15:
    status = "⚠ MODERATE"
else:
    status = "✗ HIGH OVERFITTING"
print(f"│  Status:    {status:24s} │")
print("└─────────────────────────────────────────┘\n")

#%% ========================================================================
#   5. SAVE METRICS TO EXCEL
#  ========================================================================

print("💾 METRİKLER EXCEL'E KAYDEDİLİYOR...\n")

# Create DataFrame
metrics_df = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'MAE (kW)': [mae_train, mae_val, mae_test],
    'RMSE (kW)': [rmse_train, rmse_val, rmse_test],
    'R²': [r2_train, r2_val, r2_test],
    'WAPE (%)': [wape_train, wape_val, wape_test],
    'Samples': [len(y_train_actual), len(y_val_actual), len(y_test_actual)]
})

# Create Excel writer
with pd.ExcelWriter(OUTPUT_EXCEL_NAME, engine='openpyxl') as writer:
    # Write metrics
    metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
    
    # Write overfitting analysis
    overfitting_df = pd.DataFrame({
        'Metric': ['Train R²', 'Test R²', 'Gap', 'Status'],
        'Value': [r2_train, r2_test, overfitting_gap, status]
    })
    overfitting_df.to_excel(writer, sheet_name='Overfitting Analysis', index=False)
    
    # Write hyperparameters
    hyperparams_df = pd.DataFrame({
        'Hyperparameter': [
            'TCN_FILTERS', 'TCN_KERNEL_SIZE', 'TCN_DILATION_BASE',
            'GRU_UNITS', 'ATTENTION_HEADS', 'ATTENTION_KEY_SIZE',
            'FF_UNITS_BASE', 'DROPOUT_ATTENTION', 'LEARNING_RATE',
            'BATCH_SIZE', 'EPOCHS'
        ],
        'Value': [
            TCN_FILTERS, TCN_KERNEL_SIZE, TCN_DILATION_BASE,
            GRU_UNITS, ATTENTION_HEADS, ATTENTION_KEY_SIZE,
            FF_UNITS_BASE, DROPOUT_ATTENTION, LEARNING_RATE,
            BATCH_SIZE, EPOCHS
        ]
    })
    hyperparams_df.to_excel(writer, sheet_name='Hyperparameters', index=False)

print(f"✓ Metrikler Excel'e kaydedildi: {OUTPUT_EXCEL_NAME}")
print(f"  ├─ Sheet 1: Performance Metrics")
print(f"  ├─ Sheet 2: Overfitting Analysis")
print(f"  └─ Sheet 3: Hyperparameters\n")

#%% ========================================================================
#   6. VISUALIZATION - TEST SET PREDICTION ANALYSIS (3 SUBPLOTS)
#  ========================================================================

print("📊 GRAFİKLER OLUŞTURULUYOR...\n")

# Calculate residuals
residuals = y_test_actual - y_test_pred

# Create figure with 3 subplots (vertical layout)
fig, axes = plt.subplots(3, 1, figsize=(4.5, 7.0))

# (a) TIME SERIES COMPARISON
ax1 = axes[0]
n_samples = min(100, len(y_test_actual))
indices = np.arange(n_samples)
ax1.plot(indices, y_test_actual[:n_samples], 'b-', linewidth=1.2, alpha=0.7, label='Actual')
ax1.plot(indices, y_test_pred[:n_samples], 'r-', linewidth=1.2, alpha=0.7, label='Predicted')
ax1.set_xlabel('Sample Index', fontsize=9)
ax1.set_ylabel('Power (kW)', fontsize=9)
ax1.set_title(f'(a) Actual vs Predicted Power (First {n_samples} Samples)', 
              fontsize=10, fontweight='bold')
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3, linewidth=0.5)
ax1.tick_params(labelsize=8)

# (b) SCATTER PLOT
ax2 = axes[1]
ax2.scatter(y_test_actual, y_test_pred, alpha=0.5, s=15, c='blue', 
            edgecolors='black', linewidth=0.3)
min_val = min(y_test_actual.min(), y_test_pred.min())
max_val = max(y_test_actual.max(), y_test_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, 
         label='Perfect Prediction')
ax2.set_xlabel('Actual Power (kW)', fontsize=9)
ax2.set_ylabel('Predicted Power (kW)', fontsize=9)
ax2.set_title('(b) Scatter Plot', fontsize=10, fontweight='bold')
ax2.legend(fontsize=8, loc='best')
ax2.grid(True, alpha=0.3, linewidth=0.5)
ax2.tick_params(labelsize=8)

# (c) RESIDUAL PLOT
ax3 = axes[2]
ax3.scatter(y_test_pred, residuals, alpha=0.5, s=15, c='green', 
            edgecolors='black', linewidth=0.3)
ax3.axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Zero Residual')
ax3.set_xlabel('Predicted Power (kW)', fontsize=9)
ax3.set_ylabel('Residuals (kW)', fontsize=9)
ax3.set_title('(c) Residual Plot', fontsize=10, fontweight='bold')
ax3.legend(fontsize=8, loc='best')
ax3.grid(True, alpha=0.3, linewidth=0.5)
ax3.tick_params(labelsize=8)

# Add residual statistics
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)
residual_text = f'Mean: {residual_mean:.2f} kW\nStd: {residual_std:.2f} kW'
ax3.text(0.05, 0.95, residual_text, transform=ax3.transAxes, 
         fontsize=7.5, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()

# Save as TIFF
plt.savefig(
    OUTPUT_FIGURE_NAME,
    format='tiff',
    dpi=330,
    bbox_inches='tight',
    pil_kwargs={'compression': 'tiff_lzw'}
)

print(f"✓ Şekil kaydedildi: {OUTPUT_FIGURE_NAME}")
print(f"  ├─ Format:     TIFF")
print(f"  ├─ DPI:        330")
print(f"  ├─ Boyut:      4.5 x 7.0 inch")
print(f"  ├─ Sıkıştırma: LZW")
print(f"  └─ Konum:      {OUTPUT_FIGURE_NAME}\n")

plt.show()

print("✓ Grafikler oluşturuldu ve gösterildi\n")

print("╔════════════════════════════════════════════════════════════╗")
print("║                  ✓✓✓ TAMAMLANDI! ✓✓✓                      ║")
print("╚════════════════════════════════════════════════════════════╝")
