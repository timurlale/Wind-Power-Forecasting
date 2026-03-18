"""
TCN-BiGRU-MHA Model - Missing Sensor Robustness Analysis
EDOA-Optimized Model | Single & Multi-Sensor Masking
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
from itertools import combinations
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
print("║    TCN-BiGRU-MHA MISSING SENSOR ROBUSTNESS ANALYSIS        ║")
print("╚════════════════════════════════════════════════════════════╝\n")

data = pd.read_csv('Merged_Dataset.csv')

FEATURE_COLS  = [1, 2, 3, 4, 5, 7, 9, 10]
TARGET_COL    = 6
FEATURE_NAMES = [data.columns[i] for i in FEATURE_COLS]
N_FEATURES    = len(FEATURE_COLS)

X = data.iloc[:, FEATURE_COLS].values
y = data.iloc[:, TARGET_COL].values

print(f"✓ Veri Yüklendi: {X.shape[0]} örnek, {X.shape[1]} özellik\n")
print("📋 Sensörler:")
for i, name in enumerate(FEATURE_NAMES):
    print(f"  ├─ Sensör {i}: {name}")
print()

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

# Test normalize verisini 2D olarak sakla (maskeleme bu uzayda yapılacak)
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
print("║  AŞAMA 1: BASELINE MODEL EĞİTİMİ (Tüm Sensörler Mevcut)  ║")
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

def predict_denorm(X_input_3d):
    pred_norm = model.predict(X_input_3d, verbose=0).ravel()
    return scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).ravel()

y_test_true      = scaler_y.inverse_transform(
    y_test_scaled.reshape(-1, 1)
).ravel()
baseline_pred    = predict_denorm(X_test_3d)
baseline_metrics = calculate_metrics(y_test_true, baseline_pred)

print(f"┌─────────────────────────────────────────────┐")
print(f"│  BASELINE (Tüm Sensörler Mevcut)            │")
print(f"├─────────────────────────────────────────────┤")
print(f"│  MAE : {baseline_metrics[0]:8.2f} kW                          │")
print(f"│  RMSE: {baseline_metrics[1]:8.2f} kW                          │")
print(f"│  R²  : {baseline_metrics[2]:8.6f}                         │")
print(f"│  MAPE: {baseline_metrics[3]:8.2f} %                           │")
print(f"│  WAPE: {baseline_metrics[4]:8.2f} %                           │")
print(f"└─────────────────────────────────────────────┘\n")

#%% ========================================================================
#   5A. TEK SENSÖR EKSİKLİĞİ TESTİ
#
#   Yöntem:
#     Her iterasyonda bir sensör sütunu 0.0 yapılır.
#     StandardScaler sonrası ortalama = 0 olduğundan bu işlem
#     "ortalama değer imputation" ile eşdeğerdir — literatürde
#     yaygın kabul gören missing data simülasyon yöntemi.
#
#   Çıktı:
#     Her sensörün modele katkısı (R² düşüşü) ölçülür.
#     En kritik sensör belirlenir.
#  ========================================================================

print("╔════════════════════════════════════════════════════════════╗")
print("║  AŞAMA 2A: TEK SENSÖR EKSİKLİĞİ TESTİ                     ║")
print("╚════════════════════════════════════════════════════════════╝\n")

single_results = {}

print(f"{'Sensör':<6}  {'Özellik Adı':<28}  │  "
      f"{'R²':>10}  │  {'R² Düşüş':>10}  │  "
      f"{'MAE (kW)':>10}  │  Status")
print("─" * 90)

for feat_idx in range(N_FEATURES):
    feat_name = FEATURE_NAMES[feat_idx]

    X_masked          = X_test_norm_2d.copy()
    X_masked[:, feat_idx] = 0.0
    X_masked_3d       = X_masked.reshape(-1, 1, 8)

    y_pred = predict_denorm(X_masked_3d)
    m      = calculate_metrics(y_test_true, y_pred)
    r2_drop = baseline_metrics[2] - m[2]

    status = ("✓ Robust"   if r2_drop < 0.005 else
              "✓ Good"     if r2_drop < 0.02  else
              "⚠ Moderate" if r2_drop < 0.05  else
              "✗ Critical")

    single_results[feat_idx] = {
        'feature_name': feat_name,
        'mae' : m[0], 'rmse': m[1],
        'r2'  : m[2], 'mape': m[3],
        'wape': m[4], 'r2_drop': r2_drop,
        'status': status,
    }

    print(f"  S{feat_idx:<4d}  {feat_name:<28}  │  "
          f"{m[2]:>10.6f}  │  {r2_drop:>+10.6f}  │  "
          f"{m[0]:>10.2f}  │  {status}")

most_critical = max(single_results, key=lambda k: single_results[k]['r2_drop'])
print(f"\n📌 En Kritik Sensör: S{most_critical} "
      f"[{single_results[most_critical]['feature_name']}]  →  "
      f"R² Düşüş = {single_results[most_critical]['r2_drop']:.6f}\n")

#%% ========================================================================
#   5B. ÇOKLU SENSÖR EKSİKLİĞİ TESTİ
#
#   Yöntem:
#     2'li ve 3'lü tüm sensör kombinasyonları sıfırlanır.
#     Gerçek operasyonel arıza senaryolarını simüle eder.
#     Toplam kombinasyon: C(8,2) + C(8,3) = 28 + 56 = 84 test
#  ========================================================================

print("╔════════════════════════════════════════════════════════════╗")
print("║  AŞAMA 2B: ÇOKLU SENSÖR EKSİKLİĞİ TESTİ                   ║")
print("╚════════════════════════════════════════════════════════════╝\n")

combo_2 = list(combinations(range(N_FEATURES), 2))
combo_3 = list(combinations(range(N_FEATURES), 3))
all_combos = combo_2 + combo_3

print(f"  ├─ 2'li kombinasyon: {len(combo_2):3d}")
print(f"  ├─ 3'lü kombinasyon: {len(combo_3):3d}")
print(f"  └─ Toplam test     : {len(all_combos):3d}\n")

multi_results = {}

for combo in all_combos:
    X_masked = X_test_norm_2d.copy()
    for idx in combo:
        X_masked[:, idx] = 0.0
    X_masked_3d = X_masked.reshape(-1, 1, 8)

    y_pred  = predict_denorm(X_masked_3d)
    m       = calculate_metrics(y_test_true, y_pred)
    r2_drop = baseline_metrics[2] - m[2]

    status = ("Robust"   if r2_drop < 0.01 else
              "Good"     if r2_drop < 0.03 else
              "Moderate" if r2_drop < 0.07 else
              "Critical")

    sensor_str = '+'.join([f'S{i}' for i in combo])
    multi_results[str(combo)] = {
        'combo_str'      : sensor_str,
        'sensor_names'   : ', '.join([FEATURE_NAMES[i] for i in combo]),
        'n_missing'      : len(combo),
        'mae' : m[0], 'rmse': m[1],
        'r2'  : m[2], 'mape': m[3],
        'wape': m[4], 'r2_drop': r2_drop,
        'status': status,
    }

    print(f"  [{sensor_str:<16s}]  R²: {m[2]:.6f}  │  "
          f"Düşüş: {r2_drop:+.6f}  │  MAE: {m[0]:7.2f} kW  │  {status}")

print()

#%% ========================================================================
#   6. EXCEL'E KAYDET
#  ========================================================================

print("💾 Excel'e kaydediliyor...")

EXCEL_PATH = 'missing_sensor_robustness_results.xlsx'

with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:

    # ── Sheet 1: Baseline ────────────────────────────────────────────────
    pd.DataFrame([{
        'Description'  : 'Baseline (All Sensors Present)',
        'MAE (kW)'     : round(baseline_metrics[0], 4),
        'RMSE (kW)'    : round(baseline_metrics[1], 4),
        'R²'           : round(baseline_metrics[2], 6),
        'MAPE (%)'     : round(baseline_metrics[3], 4),
        'WAPE (%)'     : round(baseline_metrics[4], 4),
        'R² Drop'      : 0.0,
        'Status'       : 'Baseline',
        'Stopped Epoch': stopped_epoch,
    }]).to_excel(writer, sheet_name='Baseline', index=False)

    # ── Sheet 2: Tek Sensör (R² düşüşüne göre sıralı) ───────────────────
    single_rows = []
    for feat_idx in range(N_FEATURES):
        r = single_results[feat_idx]
        single_rows.append({
            'Sensor Index'  : feat_idx,
            'Sensor Name'   : r['feature_name'],
            'MAE (kW)'      : round(r['mae'],     4),
            'RMSE (kW)'     : round(r['rmse'],    4),
            'R²'            : round(r['r2'],      6),
            'MAPE (%)'      : round(r['mape'],    4),
            'WAPE (%)'      : round(r['wape'],    4),
            'R² Drop'       : round(r['r2_drop'], 6),
            'Impact Status' : r['status'],
        })
    (pd.DataFrame(single_rows)
       .sort_values('R² Drop', ascending=False)
       .reset_index(drop=True)
       .to_excel(writer, sheet_name='Single_Sensor_Missing', index=False))

    # ── Sheet 3: Çoklu Sensör (R² düşüşüne göre sıralı) ─────────────────
    multi_rows = []
    for r in multi_results.values():
        multi_rows.append({
            'Missing Sensors (Code)': r['combo_str'],
            'Missing Sensors (Name)': r['sensor_names'],
            'N Missing'             : r['n_missing'],
            'MAE (kW)'              : round(r['mae'],     4),
            'RMSE (kW)'             : round(r['rmse'],    4),
            'R²'                    : round(r['r2'],      6),
            'MAPE (%)'              : round(r['mape'],    4),
            'WAPE (%)'              : round(r['wape'],    4),
            'R² Drop'               : round(r['r2_drop'], 6),
            'Impact Status'         : r['status'],
        })
    (pd.DataFrame(multi_rows)
       .sort_values('R² Drop', ascending=False)
       .reset_index(drop=True)
       .to_excel(writer, sheet_name='Multi_Sensor_Missing', index=False))

    # ── Sheet 4: Baseline Tahminler ──────────────────────────────────────
    pd.DataFrame({
        'Actual (kW)'   : y_test_true,
        'Predicted (kW)': baseline_pred,
        'Error (kW)'    : y_test_true - baseline_pred,
        'Abs Error (kW)': np.abs(y_test_true - baseline_pred),
    }).to_excel(writer, sheet_name='Baseline_Predictions', index=False)

    # ── Sheet 5: Eğitim Geçmişi ──────────────────────────────────────────
    hist = history.history
    pd.DataFrame({
        'Epoch'   : np.arange(1, len(hist['loss']) + 1),
        'Loss'    : hist['loss'],
        'Val_Loss': hist.get('val_loss', [np.nan] * len(hist['loss'])),
        'MAE'     : hist.get('mae',      [np.nan] * len(hist['loss'])),
        'Val_MAE' : hist.get('val_mae',  [np.nan] * len(hist['loss'])),
    }).to_excel(writer, sheet_name='Training_History', index=False)

print(f"✓ Excel kaydedildi → {EXCEL_PATH}")
print("  ├─ Baseline               → Temiz veri metrikleri")
print("  ├─ Single_Sensor_Missing  → Tek sensör eksikliği (R² düşüşüne göre sıralı)")
print("  ├─ Multi_Sensor_Missing   → 2'li ve 3'lü kombinasyonlar (84 test)")
print("  ├─ Baseline_Predictions   → Gerçek vs Tahmin değerleri")
print("  └─ Training_History       → Epoch bazlı Loss / MAE / Val")

print("\n╔════════════════════════════════════════════════════════════╗")
print("║      ✓✓✓ MISSING SENSOR ROBUSTNESS TAMAMLANDI! ✓✓✓        ║")
print("╚════════════════════════════════════════════════════════════╝")
