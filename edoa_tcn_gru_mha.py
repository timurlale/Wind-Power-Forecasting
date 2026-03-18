"""
EDOA-based Hyperparameter Optimization for TCN-BiGRU-MHA Model
Author: Advanced Forecasting System
Date: 2025
Fitness: R² Score (Maximize)
Algorithm: Enhanced Dhole Optimization Algorithm (EDOA)
Enhancements: Quantum Mutation + DE + Spiral Search + Memory + Stagnation Recovery
"""

#%% ========================================================================
#   ⚙️ GLOBAL CONFIGURATION - TEK NOKTADAN KONTROL
#  ========================================================================

CONFIG = {
    # EDOA Optimization Parameters
    'POPULATION_SIZE': 10,       # 🔧 Population size (dhole sayısı)
    'MAX_ITERATIONS': 30,        # 🔧 Maksimum iterasyon sayısı
    'RANDOM_SEED': 42,           # 🔧 Tekrarlanabilirlik için seed
    
    # EDOA-Specific Parameters
    'ELITE_SIZE': 5,             # 🔧 Elite archive size
    'STAGNATION_THRESHOLD': 18,  # 🔧 Stagnation detection threshold
    
    # Training Parameters
    'OPTIMIZATION_EPOCHS': 20,   # 🔧 Optimizasyon sırasında epoch sayısı
    'FINAL_EPOCHS': 100,         # 🔧 Final model eğitimi için epoch
    
    # Early Stopping
    'OPTIMIZATION_PATIENCE': 10, # 🔧 Optimizasyon için early stopping patience
    'FINAL_PATIENCE': 15,        # 🔧 Final model için early stopping patience
}

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
import json
import warnings
import time  # ✅ EKLE
from datetime import datetime
warnings.filterwarnings('ignore')

# GPU ayarları
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
#   EDOA ALGORITHM (Enhanced Dhole Optimization Algorithm)
#  ========================================================================

def dhole_optimization_enhanced2(obj_func, lb, ub, population_size=30, max_iter=100, 
                                elite_size=5, stagnation_threshold=18, seed=None):
    """
    EDOA2 (Enhanced Dhole Optimization Algorithm v2)
    
    Three Strategic Enhancements:
    1. Enhanced Exploration Mechanism (Quantum-inspired mutation)
    2. Adaptive Exploitation Strategy (DE + Spiral search)
    3. Intelligent Memory and Recovery System (Elite + Stagnation recovery)
    
    Parameters
    ----------
    obj_func : callable
        Objective function to minimize: f(x) -> scalar
    lb, ub : array-like
        Lower and upper bounds per dimension
    population_size : int
        Number of search agents (dholes)
    max_iter : int
        Maximum number of iterations
    elite_size : int
        Size of elite archive
    stagnation_threshold : int
        Iterations before stagnation recovery
    seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    prey_global : np.ndarray
        Best position found
    Best_fitness : float
        Best fitness value
    iteration_numbers : list
        Iteration history
    best_fitness_values : list
        Best fitness per iteration
    best_r2_values : list
        Best R² per iteration
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    lb = np.array(lb, dtype=float).ravel()
    ub = np.array(ub, dtype=float).ravel()
    
    if lb.size == 1 and ub.size > 1:
        lb = np.full_like(ub, lb.item())
    if ub.size == 1 and lb.size > 1:
        ub = np.full_like(lb, ub.item())
    
    dim = lb.size
    assert ub.shape == lb.shape, "lb and ub must have the same shape"
    
    # Tracking arrays
    iteration_numbers = []
    best_fitness_values = []
    best_r2_values = []
    
    # ═══════════════════════════════════════════════════════════════
    # ENHANCEMENT 1: Enhanced Exploration Mechanism
    # ═══════════════════════════════════════════════════════════════
    
    def quantum_mutation(X_i, prey_global, t, max_iter, alpha=0.5):
        """Quantum-inspired mutation for improved exploration"""
        beta = 1.0 - (t / max_iter)
        quantum_state = np.random.randn(dim) * beta
        scale = 0.01 * (1 - t / max_iter)
        
        X_quantum = (1 - alpha) * X_i + alpha * prey_global + scale * quantum_state * (ub - lb)
        return X_quantum
    
    # ═══════════════════════════════════════════════════════════════
    # ENHANCEMENT 2: Adaptive Exploitation Strategy
    # ═══════════════════════════════════════════════════════════════
    
    def de_mutation(X, i, prey_global, strategy, t, max_iter):
        """Differential Evolution mutation with adaptive parameters"""
        N = X.shape[0]
        
        # Adaptive parameters
        F = 0.5 + 0.3 * (1 - t / max_iter)  # 0.8 → 0.5
        CR = 0.7 + 0.2 * (t / max_iter)     # 0.7 → 0.9
        
        candidates = [idx for idx in range(N) if idx != i]
        r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
        
        if strategy == 0:  # DE/rand/1
            mutant = X[r1] + F * (X[r2] - X[r3])
        elif strategy == 1:  # DE/best/1
            mutant = prey_global + F * (X[r1] - X[r2])
        else:  # DE/current-to-best/1
            mutant = X[i] + F * (prey_global - X[i]) + F * (X[r1] - X[r2])
        
        cross_points = np.random.rand(dim) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dim)] = True
        
        trial = np.where(cross_points, mutant, X[i])
        return trial
    
    def enhanced_spiral_position_v2(X_i, prey, t, max_iter, fitness_i, best_fitness, diversity_factor):
        """Fitness-diversity aware spiral search"""
        fitness_ratio = abs(fitness_i - best_fitness) / (abs(best_fitness) + 1e-10)
        fitness_intensity = np.clip(fitness_ratio, 0.3, 2.5)
        diversity_intensity = 1.0 / (diversity_factor + 0.1)
        intensity = 0.7 * fitness_intensity + 0.3 * diversity_intensity
        
        b = 1.0 * intensity
        l = (2 * np.random.rand() - 1) * (1 - t / max_iter)
        r = np.exp(b * l)
        theta = 2 * np.pi * l
        D = np.abs(prey - X_i)
        
        spiral_pos = D * r * np.cos(theta) + prey
        perturbation = 0.05 * np.random.randn(dim) * (ub - lb) * (1 - t / max_iter)
        
        return spiral_pos + perturbation
    
    # ═══════════════════════════════════════════════════════════════
    # ENHANCEMENT 3: Intelligent Memory and Recovery System
    # ═══════════════════════════════════════════════════════════════
    
    def adaptive_covariance_step(X, fitness, n_elite):
        """CMA-ES inspired covariance adaptation"""
        elite_idx = np.argsort(fitness)[:n_elite]
        elite_pop = X[elite_idx]
        mean_elite = np.mean(elite_pop, axis=0)
        cov_diag = np.var(elite_pop, axis=0) + 1e-8
        step = np.random.randn(dim) * np.sqrt(cov_diag)
        return step, mean_elite
    
    def calculate_diversity(X):
        """Calculate population diversity"""
        mean_pos = np.mean(X, axis=0)
        diversity = np.mean(np.sqrt(np.sum((X - mean_pos)**2, axis=1)))
        return diversity
    
    def opposition_based_learning(X_i, lb, ub):
        """Opposition-based learning for restart"""
        return lb + ub - X_i
    
    def crossover_restart(X_worst, prey_global, elite_archive, CR=0.7):
        """Elite-guided crossover restart"""
        if len(elite_archive) < 2:
            return np.random.uniform(lb, ub, dim)
        
        elite1, elite2 = np.random.choice(len(elite_archive), 2, replace=False)
        elite_pos1, _ = elite_archive[elite1]
        elite_pos2, _ = elite_archive[elite2]
        
        cross_points = np.random.rand(dim) < CR
        result = np.where(cross_points, 
                         0.5 * (elite_pos1 + elite_pos2),
                         prey_global + 0.1 * np.random.randn(dim) * (ub - lb))
        
        return result
    
    def p_obj(x):
        """PMN function from DOA paper"""
        return ((1.0 / (1.0 + np.exp(-0.5 * (x - 25.0))))**2) * np.random.rand()
    
    # ═══════════════════════════════════════════════════════════════
    # Parameters
    # ═══════════════════════════════════════════════════════════════
    
    elite_archive = []
    
    stagnation_counter = 0
    last_best = np.inf
    
    success_history = []
    window_size = 10
    
    de_strategy_weights = [0.33, 0.33, 0.34]
    diversity_history = []
    pending_restart_indices = []
    
    # ═══════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"🚀 EDOA INITIALIZATION")
    print(f"{'='*70}")
    print(f"📊 Population Size: {population_size}")
    print(f"📊 Dimensions: {dim}")
    print(f"📊 Elite Size: {elite_size}")
    print(f"📊 Stagnation Threshold: {stagnation_threshold}")
    print(f"📊 Evaluating {population_size} initial dholes...")
    
    X = np.random.uniform(lb, ub, (population_size, dim))
    fitness_f = np.zeros(population_size, dtype=float)
    
    Best_fitness = np.inf
    for i in range(population_size):
        fitness_f[i] = obj_func(X[i])
        if fitness_f[i] < Best_fitness:
            Best_fitness = fitness_f[i]
            localBest_position = X[i].copy()
    
    prey_global = localBest_position.copy()
    
    sorted_indices = np.argsort(fitness_f)
    for i in range(min(elite_size, population_size)):
        elite_archive.append((X[sorted_indices[i]].copy(), fitness_f[sorted_indices[i]]))
    
    print(f"✅ Initialization complete!")
    print(f"🎯 Initial Best R²: {1.0 - Best_fitness:.6f}")
    
    # ═══════════════════════════════════════════════════════════════
    # MAIN EDOA LOOP
    # ═══════════════════════════════════════════════════════════════
    
    t = 1
    while t <= max_iter:
        
        print(f"\n{'='*70}")
        print(f"🔄 EDOA ITERATION {t}/{max_iter}")
        print(f"{'='*70}")
        
        # Evaluate pending restarts
        if len(pending_restart_indices) > 0:
            for idx in pending_restart_indices:
                fitness_f[idx] = obj_func(X[idx])
                if fitness_f[idx] < Best_fitness:
                    Best_fitness = fitness_f[idx]
                    prey_global = X[idx].copy()
            pending_restart_indices = []
        
        # Calculate diversity
        diversity = calculate_diversity(X)
        diversity_history.append(diversity)
        diversity_factor = np.mean(diversity_history[-5:]) if len(diversity_history) >= 5 else diversity
        
        # DOA parameters
        C = 1.0 - (t / max_iter)
        PWN = int(np.round(np.random.rand() * 15 + 5))
        prey = (prey_global + localBest_position) / 2.0
        prey_local = localBest_position.copy()
        
        # ✅ CRITICAL FIX: Calculate prey_local_fitness ONCE per iteration
        prey_local_fitness = obj_func(prey_local)
        
        Xnew = np.zeros((population_size, dim), dtype=float)
        
        # Adaptive strategy selection
        avg_success = np.mean(success_history[-window_size:]) if len(success_history) >= window_size else 0.2
        
        if avg_success < 0.1:
            de_strategy_weights = [0.5, 0.25, 0.25]
        elif avg_success > 0.3:
            de_strategy_weights = [0.2, 0.4, 0.4]
        else:
            de_strategy_weights = [0.33, 0.33, 0.34]
        
        # Dynamic population partitioning
        explore_ratio = 0.6 - 0.4 * (t / max_iter)
        n_explore = int(population_size * explore_ratio)
        n_exploit = int(population_size * (0.2 + 0.4 * (t / max_iter)))
        
        cov_step, mean_elite = adaptive_covariance_step(X, fitness_f, elite_size)
        
        print(f"📊 Exploration: {n_explore}/{population_size} ({100*n_explore/population_size:.1f}%)")
        print(f"📊 Exploitation: {n_exploit}/{population_size} ({100*n_exploit/population_size:.1f}%)")
        print(f"📊 Diversity: {diversity:.4f}")
        
        # ═══════════════════════════════════════════════════════════════
        # Position update
        # ═══════════════════════════════════════════════════════════════
        
        for i in range(population_size):
            if np.random.rand() < 0.5:
                # EXPLORATION PHASE
                if PWN < 10:
                    # Searching stage
                    Xnew[i] = X[i] + C * np.random.rand(dim) * (prey - X[i])
                    
                    # Apply quantum mutation to exploration group
                    if i < n_explore:
                        alpha = 0.3 + 0.4 * (1 - t / max_iter)
                        Xnew[i] = quantum_mutation(Xnew[i], prey_global, t, max_iter, alpha)
                else:
                    # Encircling stage
                    for j in range(dim):
                        z = int(np.round(np.random.rand() * (population_size - 1)))
                        while z == i:
                            z = int(np.round(np.random.rand() * (population_size - 1)))
                        Xnew[i, j] = X[i, j] - X[z, j] + prey[j]
            else:
                # EXPLOITATION PHASE
                # ✅ Use pre-calculated prey_local_fitness
                Q = 3.0 * np.random.rand() * fitness_f[i] / (prey_local_fitness + 1e-12)
                
                if Q > 2:
                    # Big prey
                    W_prey = np.exp(-1.0 / Q) * prey_local
                    for j in range(dim):
                        p_val1 = p_obj(PWN)
                        p_val2 = p_obj(PWN)
                        Xnew[i, j] = (X[i, j] + 
                                     np.cos(2 * np.pi * np.random.rand()) * W_prey[j] * p_val1 - 
                                     np.sin(2 * np.pi * np.random.rand()) * W_prey[j] * p_val2)
                else:
                    # Small prey
                    p_val1 = p_obj(PWN)
                    p_val2 = p_obj(PWN)
                    Xnew[i] = ((X[i] - prey_global) * p_val1 + 
                              p_val2 * np.random.rand(dim) * X[i])
                    
                    # Apply spiral search to exploitation group
                    if n_explore <= i < n_explore + n_exploit:
                        Xnew[i] = enhanced_spiral_position_v2(Xnew[i], prey_global, t, max_iter,
                                                             fitness_f[i], Best_fitness, diversity_factor)
        
        # DE Mutation for hybrid group
        hybrid_start = n_explore + n_exploit
        if hybrid_start < population_size:
            for i in range(hybrid_start, population_size):
                strategy = np.random.choice([0, 1, 2], p=de_strategy_weights)
                Xnew[i] = de_mutation(X, i, prey_global, strategy, t, max_iter)
        
        # Boundary handling
        for i in range(population_size):
            for j in range(dim):
                Xnew[i, j] = min(ub[j], Xnew[i, j])
                Xnew[i, j] = max(lb[j], Xnew[i, j])
        
        # Update local best
        localBest_position = X[0].copy()
        localBest_fitness = fitness_f[0]
        
        improvements = 0
        
        # Evaluate new population
        for i in range(population_size):
            local_fitness = obj_func(Xnew[i])
            
            if local_fitness < localBest_fitness:
                localBest_fitness = local_fitness
                localBest_position = Xnew[i].copy()
            
            # Greedy selection
            if local_fitness < fitness_f[i]:
                fitness_f[i] = local_fitness
                X[i] = Xnew[i].copy()
                improvements += 1
                
                if fitness_f[i] < Best_fitness:
                    Best_fitness = fitness_f[i]
                    prey_global = X[i].copy()
                    print(f"🎯 NEW GLOBAL BEST! R²: {1.0 - Best_fitness:.6f}")
        
        # Update success history
        success_rate = improvements / population_size
        success_history.append(success_rate)
        if len(success_history) > window_size * 2:
            success_history = success_history[-window_size:]
        
        # Update elite archive
        sorted_indices = np.argsort(fitness_f)
        elite_archive = []
        for i in range(min(elite_size, population_size)):
            elite_archive.append((X[sorted_indices[i]].copy(), fitness_f[sorted_indices[i]]))
        
        # Stagnation detection and recovery
        if abs(Best_fitness - last_best) < 1e-8:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        
        if stagnation_counter > stagnation_threshold and t < max_iter:
            print(f"⚠️  STAGNATION DETECTED! Applying recovery...")
            n_restart = int(0.25 * population_size)
            worst_indices = np.argsort(fitness_f)[-n_restart:]
            
            for idx in worst_indices:
                rand_choice = np.random.rand()
                
                if rand_choice < 0.33:
                    X[idx] = opposition_based_learning(X[idx], lb, ub)
                elif rand_choice < 0.67:
                    X[idx] = crossover_restart(X[idx], prey_global, elite_archive)
                else:
                    X[idx] = np.random.uniform(lb, ub, dim)
                
                X[idx] = np.clip(X[idx], lb, ub)
                pending_restart_indices.append(idx)
            
            stagnation_counter = 0
            print(f"✓ Restarted {n_restart} worst dholes")
        
        last_best = Best_fitness
        
        # Track iteration
        iteration_numbers.append(int(t))
        best_fitness_values.append(float(Best_fitness))
        best_r2_values.append(float(1.0 - Best_fitness))
        
        # Iteration summary
        print(f"\n📊 ITERATION {t} SUMMARY:")
        print(f"  ├─ Global Best Fitness: {Best_fitness:.6f}")
        print(f"  ├─ Global Best R²:      {1.0 - Best_fitness:.6f}")
        print(f"  ├─ Improvements:        {improvements}/{population_size}")
        print(f"  ├─ Success Rate:        {success_rate:.2%}")
        print(f"  └─ Progress:            {t}/{max_iter} ({100*t/max_iter:.1f}%)")
        
        t += 1
    
    # Final evaluation of pending restarts
    if len(pending_restart_indices) > 0:
        for idx in pending_restart_indices:
            fitness_f[idx] = obj_func(X[idx])
            if fitness_f[idx] < Best_fitness:
                Best_fitness = fitness_f[idx]
                prey_global = X[idx].copy()
    
    print(f"\n{'='*70}")
    print(f"✅ EDOA OPTIMIZATION COMPLETE!")
    print(f"{'='*70}")
    print(f"🏆 Final Best R²: {1.0 - Best_fitness:.6f}")
    
    return prey_global.copy(), float(Best_fitness), iteration_numbers, best_fitness_values, best_r2_values

#%% ========================================================================
#   DATA LOADING
#  ========================================================================

print("╔════════════════════════════════════════════════════════════╗")
print("║    EDOA HYPERPARAMETER OPTIMIZATION - TCN-BiGRU-MHA        ║")
print("║              Fitness: R² Score (Maximize)                  ║")
print("║   Algorithm: Enhanced Dhole Optimization Algorithm         ║")
print("╚════════════════════════════════════════════════════════════╝\n")

print("⚙️  CONFIGURATION:")
print(f"  ├─ Population Size:        {CONFIG['POPULATION_SIZE']}")
print(f"  ├─ Max Iterations:         {CONFIG['MAX_ITERATIONS']}")
print(f"  ├─ Optimization Epochs:    {CONFIG['OPTIMIZATION_EPOCHS']}")
print(f"  ├─ Final Epochs:           {CONFIG['FINAL_EPOCHS']}")
print(f"  ├─ Elite Size:             {CONFIG['ELITE_SIZE']}")
print(f"  └─ Stagnation Threshold:   {CONFIG['STAGNATION_THRESHOLD']}\n")

print("📊 VERİ YÜKLEME...")

data = pd.read_csv('Merged_Dataset.csv')
X = data.iloc[:, [1, 2, 3, 4, 5, 7, 9, 10]].values
y = data.iloc[:, 6].values

print(f"✓ Veri Yüklendi: {X.shape[0]} örnek, {X.shape[1]} özellik\n")

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

# Reshape
X_train = X_train.reshape(-1, 1, 8)
X_val = X_val.reshape(-1, 1, 8)
X_test = X_test.reshape(-1, 1, 8)

#%% ========================================================================
#   HYPERPARAMETER SPACE
#  ========================================================================

# Discrete choices
DISCRETE_PARAMS = {
    'TCN_FILTERS': [16, 32, 64, 128],
    'TCN_KERNEL_SIZE': [3, 5, 7, 9, 11],
    'TCN_DILATION_BASE': [1, 2, 4, 8],
    'GRU_UNITS': [256, 128, 64],
    'ATTENTION_HEADS': [2, 4, 8, 16],
    'ATTENTION_KEY_SIZE': [32, 64, 128, 256],
    'FF_UNITS_BASE': [512, 256, 128, 64],
    'BATCH_SIZE': [32, 64, 100, 128, 256]
}

# Continuous ranges
CONTINUOUS_PARAMS = {
    'DROPOUT_ATTENTION': (0.1, 0.5),
    'LEARNING_RATE': (1e-3, 1e-2)
}

# Total dimensions
n_discrete = len(DISCRETE_PARAMS)
n_continuous = len(CONTINUOUS_PARAMS)
total_dim = n_discrete + n_continuous

print(f"🔧 Hiperparametre Uzayı:")
print(f"  ├─ Discrete: {n_discrete} parametre")
print(f"  ├─ Continuous: {n_continuous} parametre")
print(f"  └─ Total Dim: {total_dim}\n")

#%% ========================================================================
#   PARAMETER ENCODING/DECODING
#  ========================================================================

def encode_params_to_vector():
    """Create bounds for EDOA optimization"""
    lb = []
    ub = []
    
    # Discrete parameters (encoded as indices)
    for key in DISCRETE_PARAMS.keys():
        lb.append(0)
        ub.append(len(DISCRETE_PARAMS[key]) - 1)
    
    # Continuous parameters
    for key in CONTINUOUS_PARAMS.keys():
        lb.append(CONTINUOUS_PARAMS[key][0])
        ub.append(CONTINUOUS_PARAMS[key][1])
    
    return np.array(lb), np.array(ub)

def decode_vector_to_params(x):
    """Convert EDOA vector to actual hyperparameters"""
    params = {}
    idx = 0
    
    # Discrete parameters
    for key in DISCRETE_PARAMS.keys():
        choice_idx = int(np.round(x[idx]))
        choice_idx = np.clip(choice_idx, 0, len(DISCRETE_PARAMS[key]) - 1)
        params[key] = DISCRETE_PARAMS[key][choice_idx]
        idx += 1
    
    # Continuous parameters
    for key in CONTINUOUS_PARAMS.keys():
        params[key] = float(x[idx])
        idx += 1
    
    return params

#%% ========================================================================
#   MODEL BUILDER
#  ========================================================================

def build_model(params):
    """Build TCN-BiGRU-MHA model with given hyperparameters"""
    
    # Extract parameters
    TCN_FILTERS = params['TCN_FILTERS']
    TCN_KERNEL_SIZE = params['TCN_KERNEL_SIZE']
    TCN_DILATION_BASE = params['TCN_DILATION_BASE']
    GRU_UNITS = params['GRU_UNITS']
    ATTENTION_HEADS = params['ATTENTION_HEADS']
    ATTENTION_KEY_SIZE = params['ATTENTION_KEY_SIZE']
    FF_UNITS_BASE = params['FF_UNITS_BASE']
    DROPOUT_ATTENTION = params['DROPOUT_ATTENTION']
    LEARNING_RATE = params['LEARNING_RATE']
    
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
    
    return model

#%% ========================================================================
#   FITNESS FUNCTION (R² ONLY)
#  ========================================================================

# Global counter and best tracker
evaluation_counter = 0
best_r2_so_far = -np.inf

def fitness_function(x):
    """
    Fitness function for EDOA optimization
    Metric: R² Score on Validation Set
    Returns: (1 - R²) to convert maximization to minimization
    """
    global evaluation_counter, best_r2_so_far
    evaluation_counter += 1
    
    # Decode parameters
    params = decode_vector_to_params(x)
    
    print(f"\n{'='*60}")
    print(f"🔄 Evaluation #{evaluation_counter}")
    print(f"{'='*60}")
    print(f"Parameters:")
    for key, val in params.items():
        print(f"  {key:20s}: {val}")
    
    try:
        # Build model
        model = build_model(params)
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['OPTIMIZATION_PATIENCE'],
            restore_best_weights=True,
            verbose=0
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG['OPTIMIZATION_EPOCHS'],
            batch_size=params['BATCH_SIZE'],
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predict on validation set
        y_val_pred_norm = model.predict(X_val, verbose=0).ravel()
        
        # Denormalize
        y_val_pred = scaler_y.inverse_transform(y_val_pred_norm.reshape(-1, 1)).ravel()
        y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
        
        # Calculate R² (Coefficient of Determination)
        ss_res = np.sum((y_val_actual - y_val_pred) ** 2)
        ss_tot = np.sum((y_val_actual - np.mean(y_val_actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Also calculate RMSE for reporting
        rmse = np.sqrt(np.mean((y_val_actual - y_val_pred) ** 2))
        
        # Fitness: Convert R² maximization to minimization
        fitness = 1.0 - r2
        
        # Track best R²
        if r2 > best_r2_so_far:
            best_r2_so_far = r2
            print(f"\n🎯 NEW BEST R²: {r2:.6f}")
        
        print(f"\n📊 Results:")
        print(f"  R²:      {r2:.6f}")
        print(f"  RMSE:    {rmse:.4f} kW")
        print(f"  Fitness: {fitness:.6f} (1 - R²)")
        print(f"  Best R² so far: {best_r2_so_far:.6f}")
        
        # Clear memory
        del model
        keras.backend.clear_session()
        
        return fitness
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 2.0  # Penalty

#%% ========================================================================
#   RUN EDOA OPTIMIZATION
#  ========================================================================

print("🚀 EDOA OPTİMİZASYONU BAŞLIYOR...\n")
print("🎯 Fitness Metriği: R² Score (Maximize)\n")
print("🔧 Algorithm: Enhanced Dhole Optimization Algorithm (EDOA)\n")
print("📋 Three Strategic Enhancements:")
print("  1. Enhanced Exploration (Quantum mutation)")
print("  2. Adaptive Exploitation (DE + Spiral search)")
print("  3. Memory & Recovery (Elite + Stagnation)\n")

# Get bounds
lb, ub = encode_params_to_vector()
import time
optimization_start_time = time.time()
# Run EDOA
best_vector, best_fitness, iteration_numbers, best_fitness_values, best_r2_values = dhole_optimization_enhanced2(
    obj_func=fitness_function,
    lb=lb,
    ub=ub,
    population_size=CONFIG['POPULATION_SIZE'],
    max_iter=CONFIG['MAX_ITERATIONS'],
    elite_size=CONFIG['ELITE_SIZE'],
    stagnation_threshold=CONFIG['STAGNATION_THRESHOLD'],
    seed=CONFIG['RANDOM_SEED']
)

# Decode best parameters
best_params = decode_vector_to_params(best_vector)

# Convert fitness back to R²
best_r2 = 1.0 - best_fitness
optimization_end_time = time.time()
optimization_duration = optimization_end_time - optimization_start_time
optimization_duration_minutes = optimization_duration / 60
optimization_duration_hours = optimization_duration / 3600
print("\n" + "="*60)
print("✅ OPTİMİZASYON TAMAMLANDI!")
print("="*60)
print("\n🏆 EN İYİ PARAMETRELER:")
for key, val in best_params.items():
    print(f"  {key:20s}: {val}")
print(f"\n🎯 En İyi R²: {best_r2:.6f}")
print(f"🎯 En İyi Fitness: {best_fitness:.6f} (1 - R²)")
print(f"🎯 Total Evaluations: {evaluation_counter}")
print(f"⏱️  Optimization Duration: {optimization_duration:.2f} seconds ({optimization_duration_minutes:.2f} minutes / {optimization_duration_hours:.4f} hours)")
#%% ========================================================================
#   FINAL MODEL TRAINING WITH BEST PARAMETERS
#  ========================================================================

print("\n" + "="*60)
print("🏗️  FİNAL MODEL EĞİTİMİ (BEST PARAMETERS)")
print("="*60 + "\n")

# Build final model
final_model = build_model(best_params)

# Train with full epochs
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=CONFIG['FINAL_PATIENCE'],
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

history = final_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CONFIG['FINAL_EPOCHS'],
    batch_size=best_params['BATCH_SIZE'],
    callbacks=[early_stop, lr_scheduler],
    verbose=1
)

#%% ========================================================================
#   FINAL EVALUATION
#  ========================================================================

print("\n" + "="*60)
print("📊 FİNAL MODEL PERFORMANSI")
print("="*60 + "\n")

# Predictions
y_train_pred_norm = final_model.predict(X_train, verbose=0).ravel()
y_val_pred_norm = final_model.predict(X_val, verbose=0).ravel()
y_test_pred_norm = final_model.predict(X_test, verbose=0).ravel()

# Denormalize
y_train_pred = scaler_y.inverse_transform(y_train_pred_norm.reshape(-1, 1)).ravel()
y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()

y_val_pred = scaler_y.inverse_transform(y_val_pred_norm.reshape(-1, 1)).ravel()
y_val_actual = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()

y_test_pred = scaler_y.inverse_transform(y_test_pred_norm.reshape(-1, 1)).ravel()
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mae, rmse, r2, mape

mae_train, rmse_train, r2_train, mape_train = calculate_metrics(y_train_actual, y_train_pred)
mae_val, rmse_val, r2_val, mape_val = calculate_metrics(y_val_actual, y_val_pred)
mae_test, rmse_test, r2_test, mape_test = calculate_metrics(y_test_actual, y_test_pred)

# Print results
print("┌─────────────────────────────────────────┐")
print("│         TRAINING SET                    │")
print("├─────────────────────────────────────────┤")
print(f"│  MAE:   {mae_train:8.2f} kW                    │")
print(f"│  RMSE:  {rmse_train:8.2f} kW                    │")
print(f"│  R²:    {r2_train:8.6f}                     │")
print(f"│  MAPE:  {mape_train:8.2f} %                    │")
print("└─────────────────────────────────────────┘\n")

print("┌─────────────────────────────────────────┐")
print("│       VALIDATION SET                    │")
print("├─────────────────────────────────────────┤")
print(f"│  MAE:   {mae_val:8.2f} kW                    │")
print(f"│  RMSE:  {rmse_val:8.2f} kW                    │")
print(f"│  R²:    {r2_val:8.6f}                     │")
print(f"│  MAPE:  {mape_val:8.2f} %                    │")
print("└─────────────────────────────────────────┘\n")

print("┌─────────────────────────────────────────┐")
print("│          TEST SET                       │")
print("├─────────────────────────────────────────┤")
print(f"│  MAE:   {mae_test:8.2f} kW                    │")
print(f"│  RMSE:  {rmse_test:8.2f} kW                    │")
print(f"│  R²:    {r2_test:8.6f}                     │")
print(f"│  MAPE:  {mape_test:8.2f} %                    │")
print("└─────────────────────────────────────────┘\n")

#%% ========================================================================
#   SAVE RESULTS TO JSON
#  ========================================================================

results = {
    'configuration': CONFIG,
    'optimization_info': {
        'algorithm': 'EDOA (Enhanced Dhole Optimization Algorithm)',
        'enhancements': [
            '1. Enhanced Exploration (Quantum-inspired mutation)',
            '2. Adaptive Exploitation (DE + Spiral search)',
            '3. Intelligent Memory & Recovery (Elite + Stagnation)'
        ],
        'fitness_metric': 'R² Score (Coefficient of Determination)',
        'optimization_goal': 'Maximize R² (Minimize 1-R²)',
        'population_size': CONFIG['POPULATION_SIZE'],
        'max_iterations': CONFIG['MAX_ITERATIONS'],
        'optimization_epochs': CONFIG['OPTIMIZATION_EPOCHS'],
        'final_epochs': CONFIG['FINAL_EPOCHS'],
        'total_evaluations': evaluation_counter,
        'optimization_duration_seconds': float(optimization_duration),
        'optimization_duration_minutes': float(optimization_duration_minutes),
        'optimization_duration_hours': float(optimization_duration_hours),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    },
    'best_hyperparameters': best_params,
    'optimization_results': {
        'best_fitness': float(best_fitness),
        'best_r2': float(best_r2),
        'best_r2_during_search': float(best_r2_so_far)
    },
    'iteration_history': {
        'iteration_numbers': iteration_numbers,
        'best_fitness_values': best_fitness_values,
        'best_r2_values': best_r2_values
    },
    'final_metrics': {
        'train': {
            'MAE': float(mae_train),
            'RMSE': float(rmse_train),
            'R2': float(r2_train),
            'MAPE': float(mape_train)
        },
        'validation': {
            'MAE': float(mae_val),
            'RMSE': float(rmse_val),
            'R2': float(r2_val),
            'MAPE': float(mape_val)
        },
        'test': {
            'MAE': float(mae_test),
            'RMSE': float(rmse_test),
            'R2': float(r2_test),
            'MAPE': float(mape_test)
        }
    },
    'overfitting_analysis': {
        'train_r2': float(r2_train),
        'test_r2': float(r2_test),
        'gap': float(abs(r2_train - r2_test)),
        'status': 'EXCELLENT' if abs(r2_train - r2_test) < 0.05 else 
                 'GOOD' if abs(r2_train - r2_test) < 0.10 else
                 'MODERATE' if abs(r2_train - r2_test) < 0.15 else
                 'HIGH OVERFITTING'
    }
}

# Save to JSON
filename = f"EDOA_R2_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n💾 Sonuçlar kaydedildi: {filename}")

print("\n╔════════════════════════════════════════════════════════════╗")
print("║                  ✓✓✓ TAMAMLANDI! ✓✓✓                      ║")
print("╚════════════════════════════════════════════════════════════╝")
