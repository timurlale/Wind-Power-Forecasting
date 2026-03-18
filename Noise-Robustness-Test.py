"""
TCN-BiGRU-MHA Model - Noise Robustness Analysis
EDOA-Optimized Model | Gaussian Noise Perturbation
Author: Advanced Forecasting System
Date: 2025
"""

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
import warnings
warnings.filterwarnings('ignore')

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
print("║         TCN-BiGRU-MHA NOISE ROBUSTNESS ANALYSIS            ║")
print("╚════════════════════════════════════════════════════════════╝\n")

data = pd.read_csv('Merged_Dataset.csv')

FEATURE_COLS  = [1, 2, 3, 4, 5, 7, 9, 10]
TARGET_COL    = 6
FEATURE_NAMES = [data.columns[i] for i in FEATURE_COLS]

X = data.iloc[:, FEATURE_COLS].values
y = data.iloc[:, TARGET_COL].values

print(f"✓ Veri Yüklendi: {X.shape[0]} örnek, {X.shape[1]} özellik\n")
print("📋 Özellikler:")
for i, name in enumerate(FEATURE_NAMES):
    print(f"  ├─ F{i}: {name}")
print()

# Normalizasyon
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_normalized = scaler_X.fit_transform(X)
y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Train/Val/Test split — orijinal kodla birebir aynı (70/10/20, shuffle=True)
X_temp, X_test_scaled, y_temp, y_test_scaled = train_test_split(
    X_normalized, y_normalized,
    test_size=0.20, random_state=42, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125, random_state=42, shuffle=True
)

print(f"✓ Veri Bölme:")
print(f"  ├─ Train: {len(X_train)} (70%)")
print(f"  ├─ Val:   {len(X_val)} (10%)")
print(f"  └─ Test:  {len(X_test_scaled)} (20%)\n")

# Test ham (normalize) değerlerini sakla — gürültü bu uzayda uygulanacak
X_test_norm_2d = X_test_scaled.copy()   # shape: (N, 8)

X_train_3d = X_train.reshape(-1, 1, 8)
X_val_3d   = X_val.reshape(-1, 1, 8)
X_test_3d  = X_test_scaled.reshape(-1, 1, 8)

#%% ========================================================================
#   2. MODEL BUILDER
#  ========================================================================

def build_model():
    TCN_FILTERS        = 128
    TCN_KERNEL_SIZE    = 3
    TCN_DILATION_BASE  = 4
    GRU_UNITS          = 64
    ATTENTION_HEADS    = 4
    ATTENTION_KEY_SIZE = 128
    FF_UNITS_BASE      = 512
    DROPOUT_ATTENTION  = 0.130622202
    LEARNING_RATE      = 0.004164095

    tcn_f2 = TCN_FILTERS * 2
    tcn_f3 = TCN_FILTERS * 4
    gru_u2 = GRU_UNITS // 2
    ff_u2  = FF_UNITS_BASE // 2
    ff_u3  = FF_UNITS_BASE // 4

    inputs = keras.Input(shape=(1, 8), name='input')

    x = layers.Conv1D(TCN_FILTERS, TCN_KERNEL_SIZE, padding='same',
                      dilation_rate=TCN_DILATION_BASE,   name='tcn_conv_1')(inputs)
    x = layers.BatchNormalization(name='tcn_bn_1')(x)
    x = layers.ReLU(name='tcn_relu_1')(x)

    x = layers.Conv1D(tcn_f2, TCN_KERNEL_SIZE, padding='same',
                      dilation_rate=TCN_DILATION_BASE*2, name='tcn_conv_2')(x)
    x = layers.BatchNormalization(name='tcn_bn_2')(x)
    x = layers.ReLU(name='tcn_relu_2')(x)

    x = layers.Conv1D(tcn_f3, TCN_KERNEL_SIZE, padding='same',
                      dilation_rate=TCN_DILATION_BASE*4, name='tcn_conv_3')(x)
    x = layers.BatchNormalization(name='tcn_bn_3')(x)
    x = layers.ReLU(name='tcn_relu_3')(x)

    x = layers.Bidirectional(layers.GRU(GRU_UNITS, return_sequences=True),
                             name='bigru_1')(x)
    x = layers.Bidirectional(layers.GRU(gru_u2,    return_sequences=True),
                             name='bigru_2')(x)

    x = layers.MultiHeadAttention(
        num_heads=ATTENTION_HEADS,
        key_dim=ATTENTION_KEY_SIZE,
        name='multihead_attention'
    )(x, x)
    x = layers.Dropout(DROPOUT_ATTENTION, name='attention_dropout')(x)
    x = layers.LayerNormalization(name='attention_norm')(x)

    x = layers.GlobalAveragePooling1D(name='gap')(x)

    x = layers.Dense(FF_UNITS_BASE, name='fc_1')(x)
    x = layers.ReLU(name='fc_relu_1')(x)
    x = layers.BatchNormalization(name='fc_bn_1')(x)

    x = layers.Dense(ff_u2, name='fc_2')(x)
    x = layers.ReLU(name='fc_relu_2')(x)
    x = layers.BatchNormalization(name='fc_bn_2')(x)

    x = layers.Dense(ff_u3, name='fc_3')(x)
    x = layers.ReLU(name='fc_relu_3')(x)

    outputs = layers.Dense(1, name='output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='TCN_BiGRU_MHA')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse', metrics=['mae']
    )
    return model

#%% ========================================================================
#   3. METRİK FONKSİYONU
#  ========================================================================

def calculate_metrics(y_true, y_pred):
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    return mae, rmse, r2, mape, wape

#%% ========================================================================
#   4. BASELINE MODEL EĞİTİMİ
#  ========================================================================

print("╔════════════════════════════════════════════════════════════╗")
print("║  AŞAMA 1: BASELINE MODEL EĞİTİMİ (Temiz Veri)             ║")
print("╚════════════════════════════════════════════════════════════╝\n")

tf.keras.backend.clear_session()
model = build_model()

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15,
    restore_best_weights=True, verbose=1
)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5,
    patience=5, min_lr=1e-6, verbose=1
)

print("🚀 Baseline eğitim başlıyor (max 100 epoch)...")
history = model.fit(
    X_train_3d, y_train,
    validation_data=(X_val_3d, y_val),
    epochs=100,
    batch_size=128,
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

stopped_epoch = (early_stop.stopped_epoch
                 if early_stop.stopped_epoch > 0 else 100)
print(f"\n✓ Baseline eğitim tamamlandı (Epoch: {stopped_epoch})\n")

# Denormalizasyon yardımcı fonksiyonu
def predict_denorm(X_input_3d):
    pred_norm = model.predict(X_input_3d, verbose=0).ravel()
    return scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).ravel()

y_test_true = scaler_y.inverse_transform(
    y_test_scaled.reshape(-1, 1)
).ravel()

baseline_pred    = predict_denorm(X_test_3d)
baseline_metrics = calculate_metrics(y_test_true, baseline_pred)

print(f"┌─────────────────────────────────────────────┐")
print(f"│  BASELINE (σ = 0.00 — Temiz Veri)           │")
print(f"├─────────────────────────────────────────────┤")
print(f"│  MAE : {baseline_metrics[0]:8.2f} kW                          │")
print(f"│  RMSE: {baseline_metrics[1]:8.2f} kW                          │")
print(f"│  R²  : {baseline_metrics[2]:8.6f}                         │")
print(f"│  MAPE: {baseline_metrics[3]:8.2f} %                           │")
print(f"│  WAPE: {baseline_metrics[4]:8.2f} %                           │")
print(f"└─────────────────────────────────────────────┘\n")

#%% ========================================================================
#   5. NOISE ROBUSTNESS TEST
#
#   Yöntem:
#     Model ağırlıkları sabit tutulur. Sadece test girdisine
#     Gaussian gürültü eklenerek model çıktısındaki bozulma ölçülür.
#
#   Gürültü normalize edilmiş uzayda uygulanır:
#     StandardScaler sonrası her özelliğin std ≈ 1 olduğundan
#     σ değerleri doğrudan SNR yorumuna uygundur.
#       σ = 0.1  →  ~%10 pertürbasyon (düşük gürültü)
#       σ = 0.3  →  ~%30 pertürbasyon (orta gürültü)
#       σ = 0.5  →  ~%50 pertürbasyon (yüksek gürültü)
#
#   Her seviye N_REPEAT kez tekrarlanır (stokastik kararlılık için).
#  ========================================================================

print("╔════════════════════════════════════════════════════════════╗")
print("║  AŞAMA 2: NOISE ROBUSTNESS TEST                            ║")
print("╚════════════════════════════════════════════════════════════╝\n")

NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_REPEAT     = 10
np.random.seed(42)

noise_results = {}

print(f"{'σ':>5}  │  {'MAE (kW)':>14}  │  {'RMSE (kW)':>14}  │  "
      f"{'R²':>18}  │  {'WAPE (%)':>12}  │  Status")
print("─" * 95)

for sigma in NOISE_LEVELS:

    rep_mae, rep_rmse, rep_r2, rep_mape, rep_wape = [], [], [], [], []

    for _ in range(N_REPEAT):
        noise        = np.random.normal(0, sigma, X_test_norm_2d.shape)
        X_noisy_3d   = (X_test_norm_2d + noise).reshape(-1, 1, 8)
        y_pred_noisy = predict_denorm(X_noisy_3d)
        m            = calculate_metrics(y_test_true, y_pred_noisy)
        rep_mae.append(m[0]);  rep_rmse.append(m[1])
        rep_r2.append(m[2]);   rep_mape.append(m[3])
        rep_wape.append(m[4])

    r2_drop = baseline_metrics[2] - np.mean(rep_r2)
    status  = ("✓ Robust"   if r2_drop < 0.01 else
               "✓ Good"     if r2_drop < 0.03 else
               "⚠ Moderate" if r2_drop < 0.05 else
               "✗ Sensitive")

    noise_results[sigma] = {
        'mae_mean' : np.mean(rep_mae),  'mae_std' : np.std(rep_mae),
        'rmse_mean': np.mean(rep_rmse), 'rmse_std': np.std(rep_rmse),
        'r2_mean'  : np.mean(rep_r2),   'r2_std'  : np.std(rep_r2),
        'mape_mean': np.mean(rep_mape), 'mape_std': np.std(rep_mape),
        'wape_mean': np.mean(rep_wape), 'wape_std': np.std(rep_wape),
        'r2_drop'  : r2_drop,
        'status'   : status,
    }

    print(f"{sigma:>5.1f}  │  "
          f"{np.mean(rep_mae):7.2f} ± {np.std(rep_mae):5.2f}  │  "
          f"{np.mean(rep_rmse):7.2f} ± {np.std(rep_rmse):5.2f}  │  "
          f"{np.mean(rep_r2):.6f} ± {np.std(rep_r2):.6f}  │  "
          f"{np.mean(rep_wape):6.2f} ± {np.std(rep_wape):4.2f}  │  {status}")

print()

#%% ========================================================================
#   6. EXCEL'E KAYDET
#  ========================================================================

print("💾 Excel'e kaydediliyor...")

EXCEL_PATH = 'noise_robustness_results.xlsx'

with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:

    # ── Sheet 1: Özet ────────────────────────────────────────────────────
    summary_rows = []

    # Baseline satırı
    summary_rows.append({
        'Noise Level (σ)'    : 0.0,
        'Description'        : 'Baseline (Clean Data)',
        'MAE Mean (kW)'      : round(baseline_metrics[0], 4),
        'MAE Std'            : 0.0,
        'RMSE Mean (kW)'     : round(baseline_metrics[1], 4),
        'RMSE Std'           : 0.0,
        'R² Mean'            : round(baseline_metrics[2], 6),
        'R² Std'             : 0.0,
        'MAPE Mean (%)'      : round(baseline_metrics[3], 4),
        'MAPE Std'           : 0.0,
        'WAPE Mean (%)'      : round(baseline_metrics[4], 4),
        'WAPE Std'           : 0.0,
        'R² Drop vs Baseline': 0.0,
        'Robustness Status'  : 'Baseline',
    })

    for sigma in NOISE_LEVELS[1:]:
        r = noise_results[sigma]
        summary_rows.append({
            'Noise Level (σ)'    : sigma,
            'Description'        : f'Gaussian Noise σ={sigma}  (N={N_REPEAT} runs)',
            'MAE Mean (kW)'      : round(r['mae_mean'],  4),
            'MAE Std'            : round(r['mae_std'],   4),
            'RMSE Mean (kW)'     : round(r['rmse_mean'], 4),
            'RMSE Std'           : round(r['rmse_std'],  4),
            'R² Mean'            : round(r['r2_mean'],   6),
            'R² Std'             : round(r['r2_std'],    6),
            'MAPE Mean (%)'      : round(r['mape_mean'], 4),
            'MAPE Std'           : round(r['mape_std'],  4),
            'WAPE Mean (%)'      : round(r['wape_mean'], 4),
            'WAPE Std'           : round(r['wape_std'],  4),
            'R² Drop vs Baseline': round(r['r2_drop'],   6),
            'Robustness Status'  : r['status'],
        })

    pd.DataFrame(summary_rows).to_excel(
        writer, sheet_name='Noise_Summary', index=False
    )

    # ── Sheet 2: Baseline Tahminler ──────────────────────────────────────
    pd.DataFrame({
        'Actual (kW)'   : y_test_true,
        'Predicted (kW)': baseline_pred,
        'Error (kW)'    : y_test_true - baseline_pred,
        'Abs Error (kW)': np.abs(y_test_true - baseline_pred),
    }).to_excel(writer, sheet_name='Baseline_Predictions', index=False)

    # ── Sheet 3: Eğitim Geçmişi ──────────────────────────────────────────
    hist = history.history
    pd.DataFrame({
        'Epoch'   : np.arange(1, len(hist['loss']) + 1),
        'Loss'    : hist['loss'],
        'Val_Loss': hist.get('val_loss', [np.nan] * len(hist['loss'])),
        'MAE'     : hist.get('mae',      [np.nan] * len(hist['loss'])),
        'Val_MAE' : hist.get('val_mae',  [np.nan] * len(hist['loss'])),
    }).to_excel(writer, sheet_name='Training_History', index=False)

print(f"✓ Excel kaydedildi → {EXCEL_PATH}")
print("  ├─ Noise_Summary        → Tüm gürültü seviyeleri (mean ± std)")
print("  ├─ Baseline_Predictions → Temiz veri tahminleri")
print("  └─ Training_History     → Epoch bazlı Loss / MAE")

print("\n╔════════════════════════════════════════════════════════════╗")
print("║         ✓✓✓ NOISE ROBUSTNESS TAMAMLANDI! ✓✓✓              ║")
print("╚════════════════════════════════════════════════════════════╝")
