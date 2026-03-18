"""
TCN-BiGRU-MHA Model - Seasonal Generalization Analysis
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
import warnings
warnings.filterwarnings('ignore')

# GPU settings
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
#   1. VERİ YÜKLEME
#  ========================================================================

print("╔════════════════════════════════════════════════════════════╗")
print("║     TCN-BiGRU-MHA SEASONAL GENERALIZATION ANALYSIS         ║")
print("╚════════════════════════════════════════════════════════════╝\n")

print("📊 VERİ YÜKLEME...")

data = pd.read_csv('Merged_Dataset.csv')
data['Date/Time'] = pd.to_datetime(data['Date/Time'])

def assign_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

data['Season'] = data['Date/Time'].dt.month.map(assign_season)

print(f"✓ Veri Yüklendi: {len(data)} örnek\n")
print("📅 Mevsimsel Dağılım:")
for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
    count = (data['Season'] == season).sum()
    print(f"  ├─ {season:6s}: {count:4d} örnek ({count/len(data)*100:.1f}%)")
print()

FEATURE_COLS = [1, 2, 3, 4, 5, 7, 9, 10]
TARGET_COL   = 6

X_all       = data.iloc[:, FEATURE_COLS].values
y_all       = data.iloc[:, TARGET_COL].values
seasons_all = data['Season'].values

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

    # TCN BLOCKS
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

    # BiGRU LAYERS
    x = layers.Bidirectional(layers.GRU(GRU_UNITS, return_sequences=True),
                             name='bigru_1')(x)
    x = layers.Bidirectional(layers.GRU(gru_u2,    return_sequences=True),
                             name='bigru_2')(x)

    # MULTI-HEAD ATTENTION
    x = layers.MultiHeadAttention(
        num_heads=ATTENTION_HEADS,
        key_dim=ATTENTION_KEY_SIZE,
        name='multihead_attention'
    )(x, x)
    x = layers.Dropout(DROPOUT_ATTENTION, name='attention_dropout')(x)
    x = layers.LayerNormalization(name='attention_norm')(x)

    # GLOBAL AVERAGE POOLING
    x = layers.GlobalAveragePooling1D(name='gap')(x)

    # FEED-FORWARD NETWORK
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
        loss='mse',
        metrics=['mae']
    )
    return model

#%% ========================================================================
#   3. METRİK FONKSİYONU
#  ========================================================================

def calculate_metrics(y_true, y_pred):
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2   = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    return mae, rmse, r2, wape

#%% ========================================================================
#   4. SEASONAL LOOP
#  ========================================================================

SEASONS    = ['Winter', 'Spring', 'Summer', 'Autumn']
BATCH_SIZE = 128
EPOCHS     = 100

seasonal_results = {}

for test_season in SEASONS:

    print("╔════════════════════════════════════════════════════════════╗")
    print(f"║  TEST MEVSİMİ: {test_season:<10s}  (Diğerleri: Eğitim)          ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # ── Veri Bölme ──────────────────────────────────────────────────────
    test_mask  = (seasons_all == test_season)
    train_mask = ~test_mask

    X_test_raw  = X_all[test_mask]
    y_test_raw  = y_all[test_mask]
    X_train_raw = X_all[train_mask]
    y_train_raw = y_all[train_mask]

    print(f"📊 Veri Dağılımı:")
    train_seasons = ', '.join([s for s in SEASONS if s != test_season])
    print(f"  ├─ Train : {len(X_train_raw):5d} örnek  ({train_seasons})")
    print(f"  └─ Test  : {len(X_test_raw):5d} örnek  ({test_season})\n")

    # ── Normalizasyon (sadece train'e fit) ──────────────────────────────
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train_raw).reshape(-1, 1, 8)
    X_test  = scaler_X.transform(X_test_raw).reshape(-1, 1, 8)

    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_test  = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

    # ── Model ───────────────────────────────────────────────────────────
    tf.keras.backend.clear_session()
    model = build_model()

    # ── Callbacks ───────────────────────────────────────────────────────
    early_stop = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # ── Eğitim ──────────────────────────────────────────────────────────
    print(f"🚀 Eğitim başlıyor (max {EPOCHS} epoch)...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    stopped_epoch = (early_stop.stopped_epoch
                     if early_stop.stopped_epoch > 0
                     else EPOCHS)
    print(f"\n✓ Eğitim tamamlandı  (Durdurulan epoch: {stopped_epoch})\n")

    # ── Tahmin & Denormalizasyon ─────────────────────────────────────────
    def predict_denorm(X_norm):
        pred_norm = model.predict(X_norm, verbose=0).ravel()
        return scaler_y.inverse_transform(pred_norm.reshape(-1, 1)).ravel()

    y_train_pred = predict_denorm(X_train)
    y_test_pred  = predict_denorm(X_test)

    y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_test_true  = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # ── Metrikler ───────────────────────────────────────────────────────
    m_train = calculate_metrics(y_train_true, y_train_pred)
    m_test  = calculate_metrics(y_test_true,  y_test_pred)

    seasonal_results[test_season] = {
        'train'        : m_train,
        'test'         : m_test,
        'y_test_true'  : y_test_true,
        'y_test_pred'  : y_test_pred,
        'history'      : history,
        'stopped_epoch': stopped_epoch,
    }

    # ── Sonuç Yazdır ────────────────────────────────────────────────────
    for label, m in [('Train', m_train), (f'Test  ({test_season})', m_test)]:
        print(f"┌─────────────────────────────────────────────┐")
        print(f"│  {label:<43}│")
        print(f"├─────────────────────────────────────────────┤")
        print(f"│  MAE : {m[0]:8.2f} kW                          │")
        print(f"│  RMSE: {m[1]:8.2f} kW                          │")
        print(f"│  R²  : {m[2]:8.6f}                         │")
        print(f"│  WAPE: {m[3]:8.2f} %                           │")
        print(f"└─────────────────────────────────────────────┘\n")

    gap    = abs(m_train[2] - m_test[2])
    status = ("✓ Excellent"      if gap < 0.05 else
              "✓ Good"           if gap < 0.10 else
              "⚠ Moderate"       if gap < 0.15 else
              "✗ High Overfitting")
    print(f"  Overfitting Gap (R²): {gap:.4f}  →  {status}\n")
    print("─" * 62 + "\n")

#%% ========================================================================
#   5. ÖZET TABLO
#  ========================================================================

print("\n╔══════════════════════════════════════════════════════════════════════╗")
print("║               SEASONAL GENERALIZATION - ÖZET SONUÇLAR               ║")
print("╚══════════════════════════════════════════════════════════════════════╝\n")

header = (f"{'Test Season':<10} │ {'MAE (kW)':>10} │ {'RMSE (kW)':>10} │"
          f" {'R²':>10} │ {'WAPE (%)':>10} │ {'Train R²':>10} │"
          f" {'Gap':>8} │ {'Epoch':>6}")
print(header)
print("─" * len(header))

mae_list  = []
rmse_list = []
r2_list   = []
wape_list = []
gap_list  = []

for season in SEASONS:
    r               = seasonal_results[season]
    mae, rmse, r2, wape = r['test']
    tr_r2           = r['train'][2]
    gap             = abs(tr_r2 - r2)
    ep              = r['stopped_epoch']

    mae_list.append(mae);   rmse_list.append(rmse)
    r2_list.append(r2);     wape_list.append(wape)
    gap_list.append(gap)

    status = ("✓ Exc" if gap < 0.05 else
              "✓ Good" if gap < 0.10 else
              "⚠ Mod"  if gap < 0.15 else
              "✗ High")

    print(f"{season:<10} │ {mae:>10.2f} │ {rmse:>10.2f} │"
          f" {r2:>10.6f} │ {wape:>10.2f} │"
          f" {tr_r2:>10.6f} │ {gap:>8.4f} │ {ep:>6}  {status}")

print("─" * len(header))
print(f"{'Mean':<10} │ {np.mean(mae_list):>10.2f} │ {np.mean(rmse_list):>10.2f} │"
      f" {np.mean(r2_list):>10.6f} │ {np.mean(wape_list):>10.2f} │"
      f" {'':>10} │ {np.mean(gap_list):>8.4f} │")
print(f"{'Std':<10} │ {np.std(mae_list):>10.2f} │ {np.std(rmse_list):>10.2f} │"
      f" {np.std(r2_list):>10.6f} │ {np.std(wape_list):>10.2f} │"
      f" {'':>10} │ {np.std(gap_list):>8.4f} │")

print(f"\n📌 En İyi Mevsim (MAE) : {SEASONS[np.argmin(mae_list)]}"
      f"  →  MAE = {min(mae_list):.2f} kW")
print(f"📌 En Zor Mevsim (MAE) : {SEASONS[np.argmax(mae_list)]}"
      f"  →  MAE = {max(mae_list):.2f} kW")

print("\n╔════════════════════════════════════════════════════════════╗")
print("║            ✓✓✓ TÜM MEVSİMLER TAMAMLANDI! ✓✓✓              ║")
print("╚════════════════════════════════════════════════════════════╝")

#%% ========================================================================
#   6. EXCEL'E KAYDET  ← sadece bu bölüm eklendi, başka hiçbir şey değişmedi
#  ========================================================================

print("\n💾 Excel'e kaydediliyor...")

EXCEL_PATH = 'seasonal_generalization_results.xlsx'

with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:

    # ── Sheet 1: Özet Metrikler ──────────────────────────────────────────
    summary_rows = []
    for season in SEASONS:
        r                        = seasonal_results[season]
        mae, rmse, r2, wape      = r['test']
        tr_mae, tr_rmse, tr_r2, tr_wape = r['train']
        gap                      = abs(tr_r2 - r2)

        summary_rows.append({
            'Test Season'       : season,
            'Stopped Epoch'     : r['stopped_epoch'],
            'Test MAE (kW)'     : round(mae,     4),
            'Test RMSE (kW)'    : round(rmse,    4),
            'Test R²'           : round(r2,      6),
            'Test WAPE (%)'     : round(wape,    4),
            'Train MAE (kW)'    : round(tr_mae,  4),
            'Train RMSE (kW)'   : round(tr_rmse, 4),
            'Train R²'          : round(tr_r2,   6),
            'Train WAPE (%)'    : round(tr_wape, 4),
            'R² Gap'            : round(gap,     6),
            'Overfitting Status': ("Excellent"        if gap < 0.05 else
                                   "Good"             if gap < 0.10 else
                                   "Moderate"         if gap < 0.15 else
                                   "High Overfitting"),
        })

    summary_rows.append({
        'Test Season'   : 'Mean',
        'Test MAE (kW)' : round(np.mean(mae_list),  4),
        'Test RMSE (kW)': round(np.mean(rmse_list), 4),
        'Test R²'       : round(np.mean(r2_list),   6),
        'Test WAPE (%)' : round(np.mean(wape_list), 4),
        'R² Gap'        : round(np.mean(gap_list),  6),
    })
    summary_rows.append({
        'Test Season'   : 'Std',
        'Test MAE (kW)' : round(np.std(mae_list),  4),
        'Test RMSE (kW)': round(np.std(rmse_list), 4),
        'Test R²'       : round(np.std(r2_list),   6),
        'Test WAPE (%)' : round(np.std(wape_list), 4),
        'R² Gap'        : round(np.std(gap_list),  6),
    })

    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

    # ── Sheet 2-5: Her mevsim için Gerçek vs Tahmin ──────────────────────
    for season in SEASONS:
        r = seasonal_results[season]
        pd.DataFrame({
            'Actual (kW)'   : r['y_test_true'],
            'Predicted (kW)': r['y_test_pred'],
            'Error (kW)'    : r['y_test_true'] - r['y_test_pred'],
            'Abs Error (kW)': np.abs(r['y_test_true'] - r['y_test_pred']),
        }).to_excel(writer, sheet_name=f'Predictions_{season}', index=False)

    # ── Sheet 6-9: Her mevsim için Eğitim Geçmişi ───────────────────────
    for season in SEASONS:
        hist = seasonal_results[season]['history'].history
        pd.DataFrame({
            'Epoch'   : np.arange(1, len(hist['loss']) + 1),
            'Loss'    : hist['loss'],
            'MAE'     : hist.get('mae', [np.nan] * len(hist['loss'])),
        }).to_excel(writer, sheet_name=f'History_{season}', index=False)

print(f"✓ Excel kaydedildi → {EXCEL_PATH}")
print("  ├─ Summary            → Özet metrikler (Train / Test / Gap)")
for season in SEASONS:
    print(f"  ├─ Predictions_{season:<6s} → Gerçek vs Tahmin")
for season in SEASONS:
    print(f"  ├─ History_{season:<6s}     → Epoch bazlı Loss / MAE")
