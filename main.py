import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import plotly.graph_objects as go
import glob
import os
import joblib
import time
import webbrowser
import itertools
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import optuna
from sklearn.ensemble import VotingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, cross_validate, GroupKFold, ShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin, clone, RegressorMixin
import shap
from scipy.optimize import minimize
import warnings

import config
from transformers import InteractionFeaturesTransformer, VIFSelector
from utils import format_time, create_features, filter_outliers_grouped, plot_pareto_front, plot_learning_curve, generate_html_report

config.set_academic_plot_style()

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=UserWarning, module='catboost')

script_start = time.time()
dynamic_descriptions = {}

os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)
for f in glob.glob(os.path.join(config.RESULTS_DIR, "*.*")):
    try:
        os.remove(f)
    except OSError:
        pass

# --- Data Loading ---
print("\n--- Loading data... ---")
data_cache_path = os.path.join(config.CACHE_DIR, "full_df_cache.pkl")
xlsx_files_cache_path = os.path.join(config.CACHE_DIR, "xlsx_files_cache.pkl")
start_loading = time.time()

cache_loaded = False
if config.USE_CACHE and os.path.exists(data_cache_path) and os.path.exists(xlsx_files_cache_path):
    print("--- Loading Data From Cache ---")
    try:
        full_df = pd.read_pickle(data_cache_path)
        xlsx_files = joblib.load(xlsx_files_cache_path)
        print(f"Loaded {len(full_df)} rows from {len(xlsx_files)} files from cache.")
        cache_loaded = True
    except Exception as e:
        print(f"Warning: Could not load data cache ({e}). Forcing data reload...")
        cache_loaded = False

if not cache_loaded:
    print("--- Loading and Processing Data (Cache not found, disabled, or invalid) ---")
    base_path = config.BASE_PATH
    print(f"Loading data: {base_path} ...")

    xlsx_files = sorted(glob.glob(os.path.join(base_path, "*.xlsx")))
    if not xlsx_files:
        print("ERROR: No files found! Check the path.")
        sys.exit()

    all_data = []
    missing_cols_files = []
    for filepath in tqdm(xlsx_files, desc="Loading data", unit="file"):
        try:
            with pd.ExcelFile(filepath, engine='openpyxl') as xls:
                sheet_name = "Sheet Numeric SRA" if "Sheet Numeric SRA" in xls.sheet_names else 0
                df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                
            missing = []
            if 'Concentration' not in df.columns: missing.append('Concentration')
            if 'Esterified' not in df.columns: missing.append('Esterified')
            if missing:
                missing_cols_files.append((os.path.basename(filepath), missing))

            df = df.drop(0).reset_index(drop=True)
            df = df.iloc[:-5]
            
            if 'Temperature 1' in df.columns:
                df.rename(columns={'Temperature 1': 'Temperature'}, inplace=True)

            cols = ['Time', 'Load', 'Temperature', 'COF', 'Friction absolute integral', 'Concentration', 'Esterified']
            for c in cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            
            df = df.dropna(subset=['Concentration', 'COF', 'Friction absolute integral', 'Load', 'Temperature'])
            df = df[(df['Temperature'] != 0) & (df['Load'] != 0)]
            df = df[df['Time'] > 35.0]
            df = df.reset_index(drop=True)
            df = df.groupby(df.index // config.DOWNSAMPLING_RATE).mean(numeric_only=True).reset_index(drop=True)
            df['File_ID'] = os.path.basename(filepath)
            
            if 'Esterified' in df.columns:
                df['Esterified'] = df['Esterified'].fillna(0).astype(int)
            
            if not df.empty:
                df_raw_cof = df['COF'].copy()
                
                if len(all_data) in [0, 14, 29, 49]:
                    plt.figure(figsize=(6.3, 3.15))
                    plt.plot(df['Time'], df_raw_cof, label='Original signal', color='silver', alpha=0.7)
                    
                df['COF'] = df['COF'].rolling(window=config.ROLLING_WINDOW_SIZE).mean()
                df = df.dropna(subset=['COF'])
                
                if len(all_data) == 0:
                    plt.plot(df['Time'], df['COF'], label='Filtered signal (Rolling Mean)', color='orange')
                    plt.xlabel("Time [s]")
                    plt.ylabel("Coefficient of friction (COF) [-]")
                    ymin, ymax = plt.gca().get_ylim()
                    plt.ylim(ymin, ymax + (ymax - ymin) * 0.35)
                    plt.legend(loc='upper right')
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering.svg"), format='svg', bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    
                elif len(all_data) == 14:
                    plt.plot(df['Time'], df['COF'], label='Filtered signal (Rolling Mean)', color='green')
                    plt.xlabel("Time [s]")
                    plt.ylabel("Coefficient of friction (COF) [-]")
                    ymin, ymax = plt.gca().get_ylim()
                    plt.ylim(ymin, ymax + (ymax - ymin) * 0.35)
                    plt.legend(loc='upper right')
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering_3.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    dynamic_descriptions["Effect_of_noise_filtering_3.png"] = "Comparison of raw measurement data and the smoothed curve using a rolling mean filter on a third data file for verification."

                elif len(all_data) == 29:
                    plt.plot(df['Time'], df['COF'], label='Filtered signal (Rolling Mean)', color='blue')
                    plt.xlabel("Time [s]")
                    plt.ylabel("Coefficient of friction (COF) [-]")
                    ymin, ymax = plt.gca().get_ylim()
                    plt.ylim(ymin, ymax + (ymax - ymin) * 0.35)
                    plt.legend(loc='upper right')
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering_4.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    dynamic_descriptions["Effect_of_noise_filtering_4.png"] = "Comparison of raw measurement data and the smoothed curve using a rolling mean filter on a fourth data file for verification."
                    
                elif len(all_data) == 49:
                    plt.plot(df['Time'], df['COF'], label='Filtered signal (Rolling Mean)', color='purple')
                    plt.xlabel("Time [s]")
                    plt.ylabel("Coefficient of friction (COF) [-]")
                    ymin, ymax = plt.gca().get_ylim()
                    plt.ylim(ymin, ymax + (ymax - ymin) * 0.35)
                    plt.legend(loc='upper right')
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering_2.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
                    plt.close()

                all_data.append(df)
                max_time = df['Time'].max()
                last_5m_df = df[df['Time'] >= max_time - 300]
                
                if not last_5m_df.empty:
                    mean_cof = last_5m_df['COF'].mean()
                    mean_fai = last_5m_df['Friction absolute integral'].mean()
                    summary_row = df.iloc[[-1]].copy()
                    summary_row['COF'] = mean_cof
                    summary_row['Friction absolute integral'] = mean_fai
                    summary_row['Time'] = 7200.0
                    all_data.append(summary_row)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error: {os.path.basename(filepath)} - {e}")

    if not all_data:
        sys.exit()

    if missing_cols_files:
        print("\n--- Warning: Missing columns in loaded files ---")
        for fname, mcols in missing_cols_files:
            print(f"  - {fname}: missing -> {', '.join(mcols)}")
    else:
        print("\n--- Column check: All files contain 'Esterified' and 'Concentration' columns! ---")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[full_df['Time'] > 0]
    full_df = full_df[(full_df['COF'] > 0) & (full_df['Friction absolute integral'] > 0)]

    full_df = create_features(full_df)

    if 'Esterified' not in full_df.columns:
        full_df['Esterified'] = 0
    full_df['Esterified'] = full_df['Esterified'].fillna(0).astype(int)

    rounded_load = full_df['Load'].round()
    rounded_temp = full_df['Temperature'].round()

    counts = full_df.groupby([full_df['Concentration'], rounded_load, rounded_temp, full_df['Esterified']])['Time'].transform('count')
    full_df['Sample_Weight'] = 1.0 / counts
    full_df['Sample_Weight'] = np.sqrt(full_df['Sample_Weight'])


    full_df['Sample_Weight'] = full_df['Sample_Weight'] * (len(full_df) / full_df['Sample_Weight'].sum())

    if config.USE_CACHE:
        print("\nSaving data to cache...")
        full_df.to_pickle(data_cache_path)
        joblib.dump(xlsx_files, xlsx_files_cache_path)

loading_duration = time.time() - start_loading
print(f"Data loading/caching completed in {format_time(loading_duration)}")

# --- Check for COF > 0.26 ---
high_cof_026_mask = full_df['COF'] > 0.26
if high_cof_026_mask.any():
    high_cof_026_counts = full_df[high_cof_026_mask].groupby('File_ID').size()
    print(f"\nInfo: Number of COF > 0.26 values in filtered dataset: {high_cof_026_mask.sum()}.")
    print("These belong to the following files:")
    for fid, count in high_cof_026_counts.items():
        print(f"  - {fid}: {count} data points")
else:
    print("\nInfo: No COF values above 0.26 in the dataset.")

import matplotlib.gridspec as gridspec

print("\n--- Generating Input and Target Distribution plots... ---")
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.8], wspace=0.4, hspace=0.4)

features_to_plot = ['Load', 'Temperature', 'Concentration', 'Esterified']

for i, feature in enumerate(features_to_plot): 
    row = i // 2
    col = i % 2
    ax = plt.subplot(gs[row, col])
    ax.hist(full_df[feature].dropna(), bins=15, color='#003f5c', edgecolor='black', alpha=0.8)
    ax.set_xlabel(config.NAME_MAPPING.get(feature, feature), fontsize=9)
    if col == 0:
        ax.set_ylabel('Count', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)

ax_target = plt.subplot(gs[:, 2])
ax_target.boxplot(full_df['COF'].dropna(), vert=True, patch_artist=True, 
                  boxprops=dict(facecolor='#bc5090', color='black'),
                  medianprops=dict(color='black', linewidth=1.5))
ax_target.set_xlabel(config.NAME_MAPPING.get('COF', 'COF'), fontsize=10)
ax_target.set_xticks([]) 
ax_target.tick_params(axis='y', labelsize=9)

plt.suptitle("Model Input Variables vs Target Variable", fontsize=14, y=0.95)
plt.savefig(os.path.join(config.RESULTS_DIR, "Input_Target_Distributions.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

dynamic_descriptions["Input_Target_Distributions.png"] = "Distribution of the main input variables (histograms) and the target COF variable (boxplot)."

file_group = full_df.groupby('File_ID').agg({
    'Temperature': 'mean',
    'Load': 'mean',
    'Concentration': 'mean',
    'Esterified': 'first'
})
file_group['Temperature'] = (file_group['Temperature'] / 5).round() * 5
file_group['Temperature'] = file_group['Temperature'].astype(int)
file_group['Load'] = (file_group['Load'] / 5).round() * 5
file_group['Load'] = file_group['Load'].astype(int)
file_group['Concentration'] = file_group['Concentration'].round(2)
distribution_summary = file_group.groupby(['Temperature', 'Load', 'Concentration', 'Esterified']).size().reset_index(name='File_Count')
distribution_summary = distribution_summary.sort_values(by='File_Count', ascending=False).reset_index(drop=True)
print("\n--- Data Distribution (files per measurement point) ---")
print(distribution_summary.to_string(index=False))

print("\n--- Preparing Data and Cross-Validation Folds ---")
X = full_df[['Time', 'Log_Time', 'Time_Squared', 'Load', 'Temperature', 'Concentration', 'Esterified', 'Hertz_Stress_MPa']]
Y = full_df[['COF', 'Friction absolute integral']]
groups = full_df['File_ID']

file_stats = full_df.groupby('File_ID')[['Load', 'Temperature']].mean()
try:
    file_stats['Load_Bin'] = pd.qcut(file_stats['Load'], q=3, labels=False, duplicates='drop')
except ValueError:
    file_stats['Load_Bin'] = pd.qcut(file_stats['Load'].rank(method='first'), q=3, labels=False)

try:
    file_stats['Temp_Bin'] = pd.qcut(file_stats['Temperature'], q=3, labels=False, duplicates='drop')
except ValueError:
    file_stats['Temp_Bin'] = pd.qcut(file_stats['Temperature'].rank(method='first'), q=3, labels=False)

file_stats['Stratify_Label'] = file_stats['Load_Bin'].astype(str) + "_" + file_stats['Temp_Bin'].astype(str)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
train_files_idx, test_files_idx = next(splitter.split(file_stats, file_stats['Stratify_Label']))

train_files = file_stats.index[train_files_idx]
test_files = file_stats.index[test_files_idx]

train_idx = full_df[full_df['File_ID'].isin(train_files)].index
test_idx = full_df[full_df['File_ID'].isin(test_files)].index

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = Y.loc[train_idx], Y.loc[test_idx]
groups_train = groups.loc[train_idx]
weights_train = full_df['Sample_Weight'].loc[train_idx]

X_cols_raw = X.columns

print("\n--- Applying Global Feature Engineering (Interaction & VIF) ---")
global_interact = InteractionFeaturesTransformer(
    load_col='Load', 
    temp_col='Temperature', 
    conc_col='Concentration', 
    ester_col='Esterified'
)
global_vif = VIFSelector(threshold=5.0)

X_train_interact = global_interact.fit_transform(X_train)
X_train = global_vif.fit_transform(X_train_interact)

X_test_interact = global_interact.transform(X_test)
X_test = global_vif.transform(X_test_interact)

X_interact = global_interact.transform(X)
X = global_vif.transform(X_interact)

range_conc = np.arange(0.0, 0.61, config.GRID_STEP_CONC)
range_load = np.arange(10, 201, config.GRID_STEP_LOAD)
range_temp = np.arange(40, 121, config.GRID_STEP_TEMP)
combos = list(itertools.product(range_conc, range_load, range_temp))
grid_df = pd.DataFrame(combos, columns=['Concentration', 'Load', 'Temperature'])
grid_df['Esterified'] = config.PLOT_ESTERIFIED_STATE
grid_df['Time'] = 7200
grid_df = create_features(grid_df)
grid_df = grid_df[X_cols_raw]

gkf_cv = GroupKFold(n_splits=config.CV_SPLITS)

# --- Prepare template_df for Optimum Curve Generations ---
first_file_id = os.path.basename(xlsx_files[0])
template_df = full_df[full_df['File_ID'] == first_file_id].copy()
template_df = template_df.dropna(subset=['Time', 'Load', 'Temperature']).sort_values('Time')
template_df = template_df[(template_df['Temperature'] != 0) & (template_df['Load'] != 0)]
template_df = template_df[template_df['Time'] > 0]

# --- Custom Classes ---
class PreFittedVotingRegressor(RegressorMixin, BaseEstimator):
    """
    Custom Voting model that supports MultiOutput and averages pre-fitted base models,
    also supporting retraining on the full dataset.
    """
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights
        
    def fit(self, X, y, **fit_params):
        for name, est in self.estimators:
            fparams = {}
            if "Random Forest" in name: fparams['rf__sample_weight'] = fit_params.get('sample_weight')
            elif "XGBoost" in name: fparams['xgb__sample_weight'] = fit_params.get('sample_weight')
            elif "LightGBM" in name: fparams['lgbm__sample_weight'] = fit_params.get('sample_weight')
            elif "CatBoost" in name: fparams['cat__sample_weight'] = fit_params.get('sample_weight')
            elif "Ridge" in name or "Polynomial" in name: fparams['ridge__sample_weight'] = fit_params.get('sample_weight')
            fparams = {k: v for k, v in fparams.items() if v is not None}
            est.fit(X, y, **fparams)
        return self
        
    def predict(self, X):
        return np.average([model.predict(X) for name, model in self.estimators], axis=0, weights=self.weights)

# --- Model Training ---
print("\n--- Training and tuning models... ---")
models_cache_path = os.path.join(config.CACHE_DIR, "models_cache.pkl")
if config.USE_CACHE and os.path.exists(models_cache_path):
    print("\n--- Loading Trained Models From Cache ---")
    try:
        cached_models = joblib.load(models_cache_path)
        results = cached_models['results']
        best_model_overall = cached_models['best_model_overall']
        best_model_name = cached_models['best_model_name']
        best_r2_overall = max(r['R2_CV'] for r in results)
        print("Models loaded from cache.")
        models_loaded = True
    except Exception as e:
        print(f"Warning: Could not load cache ({e}). Forcing retraining...")
        models_loaded = False
else:
    models_loaded = False

if not models_loaded:
    print("\n--- Training Models (Cache not found, disabled, or invalid) ---")
    results = []
    best_model_overall = None
    best_r2_overall = -np.inf
    best_model_name = ""
    for name, cfg in tqdm(config.models_config.items(), desc="Training models"):
        start_model_total = time.time()
        best_params = {}
        
        if cfg["params"]:
            fit_params = {}
            if "Random Forest" in name: fit_params['rf__sample_weight'] = weights_train.values
            elif "XGBoost" in name: fit_params['xgb__sample_weight'] = weights_train.values
            elif "LightGBM" in name: fit_params['lgbm__sample_weight'] = weights_train.values
            elif "CatBoost" in name: fit_params['cat__sample_weight'] = weights_train.values
            elif "Polynomial" in name: fit_params['ridge__sample_weight'] = weights_train.values

            def objective(trial):
                sampled_params = {}
                for param_name, param_values in cfg["params"].items():
                    if isinstance(param_values, list) and len(param_values) > 1:
                        if any(isinstance(x, (tuple, list)) for x in param_values):
                            idx = trial.suggest_categorical(param_name + "_idx", list(range(len(param_values))))
                            sampled_params[param_name] = param_values[idx]
                        elif all(isinstance(x, int) and not isinstance(x, bool) for x in param_values):
                            sampled_params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                        elif all(isinstance(x, float) for x in param_values):
                            sampled_params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                        else:
                            sampled_params[param_name] = trial.suggest_categorical(param_name, param_values)
                    else:
                        sampled_params[param_name] = trial.suggest_categorical(param_name, param_values)

                model = clone(cfg["model"])
                model.set_params(**sampled_params)
                try:
                    cv_scores = cross_validate(model, X_train, y_train, cv=gkf_cv, groups=groups_train, scoring='r2', params=fit_params, n_jobs=1, error_score='raise')
                    return np.mean(cv_scores['test_score'])
                except Exception as e:
                    print(f"  Trial failed: {e}")
                    return -100.0

            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            def logging_callback(study, frozen_trial):
                print(f"  Trial {frozen_trial.number} finished with R2 CV: {frozen_trial.value:.4f} | Best so far: {study.best_value:.4f} (Trial {study.best_trial.number})")
            n_trials = cfg.get("n_trials", config.SEARCH_ITERATIONS)
            study.optimize(objective, n_trials=n_trials, callbacks=[logging_callback])

            try:
                import optuna.visualization.matplotlib as ovm
                if len(study.trials) > 1:
                    ovm.plot_param_importances(study)
                    safe_model_name = name.replace(' ', '_').replace('(', '').replace(')', '')
                    plot_filename = f"Optuna_Importances_{safe_model_name}.png"
                    fig = plt.gcf()
                    fig.set_size_inches(10, 6)
                    fig.set_facecolor('white')
                    ax = plt.gca()
                    ax.set_facecolor('white')
                    ax.grid(False)
                    ax.set_title("")
                    plt.suptitle("")
                    ticks = ax.get_yticks()
                    labels = [tick.get_text() for tick in ax.get_yticklabels()]
                    new_labels = []
                    for raw_label in labels:
                        parts = raw_label.split('__')
                        short_name = parts[-1].replace('_', ' ').capitalize()
                        new_labels.append(f"{short_name}\n({raw_label})")
                    
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(new_labels, multialignment='center')
                    plt.tight_layout()
                    plt.savefig(os.path.join(config.RESULTS_DIR, plot_filename), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    dynamic_descriptions[plot_filename] = f"Optuna Hyperparameter Importances for {name}."
            except Exception as e:
                print(f"  Could not generate Optuna plot for {name}: {e}")

            config.set_academic_plot_style()

            best_params_raw = study.best_params
            best_params = {}
            for k, v in best_params_raw.items():
                if k.endswith("_idx"):
                    orig_k = k[:-4]
                    best_params[orig_k] = cfg["params"][orig_k][v]
                else:
                    best_params[k] = v

            best_estimator = clone(cfg["model"])
            best_estimator.set_params(**best_params)
        else:
            best_estimator = clone(cfg["model"])
            best_params = "Default"

        fit_params_final = {}
        if "Random Forest" in name: fit_params_final['rf__sample_weight'] = weights_train.values
        elif "XGBoost" in name: fit_params_final['xgb__sample_weight'] = weights_train.values
        elif "LightGBM" in name: fit_params_final['lgbm__sample_weight'] = weights_train.values
        elif "CatBoost" in name: fit_params_final['cat__sample_weight'] = weights_train.values
        elif "Polynomial" in name: fit_params_final['ridge__sample_weight'] = weights_train.values
            
        best_estimator.fit(X_train, y_train, **fit_params_final)
        tuning_training_time = time.time() - start_model_total
        
        start_pred = time.time()
        y_pred = np.maximum(best_estimator.predict(X_test), config.PREDICTION_LOWER_BOUND)
        pred_time_ms = (time.time() - start_pred) * 1000
        
        y_train_pred = best_estimator.predict(X_train)
        r2_train = r2_score(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        r2_test = r2_score(y_test, y_pred)
        r2_test_raw = r2_score(y_test, y_pred, multioutput='raw_values')
        r2_cof, r2_fai = r2_test_raw[0], r2_test_raw[1]
        
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_test_raw = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
        rmse_cof, rmse_fai = rmse_test_raw[0], rmse_test_raw[1]
        
        mae_test = mean_absolute_error(y_test, y_pred)
        
        # Calculate steady-state (last 5 min) accuracy
        steady_actual_cof = []
        steady_actual_fai = []
        steady_pred_cof = []
        steady_pred_fai = []
        for fid in test_files:
            file_data = full_df[full_df['File_ID'] == fid]
            max_t = file_data['Time'].max()
            last_5m = file_data[file_data['Time'] >= max_t - 300]
            steady_actual_cof.append(last_5m['COF'].mean())
            steady_actual_fai.append(last_5m['Friction absolute integral'].mean())
            
            phys_load, phys_temp, phys_conc, phys_est = file_data.iloc[0][['Load', 'Temperature', 'Concentration', 'Esterified']]
            check_df_st = pd.DataFrame({'Time': [7050], 'Load': [phys_load], 'Temperature': [phys_temp], 'Concentration': [phys_conc], 'Esterified': [phys_est]})
            check_df_st = create_features(check_df_st)[X_cols_raw]
            check_df_st_trans = global_vif.transform(global_interact.transform(check_df_st))
            preds_st = np.maximum(best_estimator.predict(check_df_st_trans), config.PREDICTION_LOWER_BOUND)
            steady_pred_cof.append(preds_st[0, 0])
            steady_pred_fai.append(preds_st[0, 1])
            
        r2_steady_cof = r2_score(steady_actual_cof, steady_pred_cof)
        r2_steady_fai = r2_score(steady_actual_fai, steady_pred_fai)
        rmse_steady_cof = np.sqrt(mean_squared_error(steady_actual_cof, steady_pred_cof))
        rmse_steady_fai = np.sqrt(mean_squared_error(steady_actual_fai, steady_pred_fai))
        
        cv_scores = cross_validate(best_estimator, X_train, y_train, cv=gkf_cv, groups=groups_train, scoring=['r2', 'neg_root_mean_squared_error'])
        avg_r2 = np.mean(cv_scores['test_r2'])
        
        feature_imp = None
        pipeline = best_estimator.regressor_ if hasattr(best_estimator, 'regressor_') else best_estimator
        if "Random Forest" in name: feature_imp = pipeline.named_steps['rf'].feature_importances_
        elif "XGBoost" in name: feature_imp = np.mean([est.feature_importances_ for est in pipeline.named_steps['xgb'].estimators_], axis=0)
        elif "LightGBM" in name: feature_imp = np.mean([est.feature_importances_ for est in pipeline.named_steps['lgbm'].estimators_], axis=0)
        elif "CatBoost" in name: feature_imp = np.mean([est.feature_importances_ for est in pipeline.named_steps['cat'].estimators_], axis=0)

        selected_features_model = global_vif.selected_features_
        
        grid_df_trans = global_vif.transform(global_interact.transform(grid_df))
        preds_grid = np.maximum(best_estimator.predict(grid_df_trans), config.PREDICTION_LOWER_BOUND)
        norm_cof = (preds_grid[:,0] - preds_grid[:,0].min()) / (preds_grid[:,0].max() - preds_grid[:,0].min() + 1e-9)
        norm_fai = (preds_grid[:,1] - preds_grid[:,1].min()) / (preds_grid[:,1].max() - preds_grid[:,1].min() + 1e-9)
        scores = norm_cof + norm_fai
        best_idx = np.argmin(scores)
        
        opt_conc = grid_df.iloc[best_idx]['Concentration']
        opt_load = grid_df.iloc[best_idx]['Load']
        opt_temp = grid_df.iloc[best_idx]['Temperature']
        
        t_end_vals = np.arange(6900, 7201, 10)
        check_df = pd.DataFrame({'Time': t_end_vals, 'Load': opt_load, 'Temperature': opt_temp, 'Concentration': opt_conc, 'Esterified': config.PLOT_ESTERIFIED_STATE})
        check_df = create_features(check_df)[X_cols_raw]
        check_df_trans = global_vif.transform(global_interact.transform(check_df))
        check_preds = np.maximum(best_estimator.predict(check_df_trans), config.PREDICTION_LOWER_BOUND)
        pred_cof_5m = np.mean(check_preds[:, 0])
        pred_fai_5m = np.mean(check_preds[:, 1])
        
        # --- Calculate curve for the optimal parameters ---
        sim_input_model = pd.DataFrame({'Time': template_df['Time'], 'Load': opt_load, 'Temperature': opt_temp, 'Concentration': opt_conc, 'Esterified': config.PLOT_ESTERIFIED_STATE})
        sim_input_model = create_features(sim_input_model)[X_cols_raw]
        sim_input_model_trans = global_vif.transform(global_interact.transform(sim_input_model))
        curve_preds_model = np.maximum(best_estimator.predict(sim_input_model_trans), config.PREDICTION_LOWER_BOUND)
        curve_cof_model = curve_preds_model[:, 0]
        curve_time_model = template_df['Time'].values

        # Base oil prediction for comparison
        sim_input_base = pd.DataFrame({'Time': template_df['Time'], 'Load': opt_load, 'Temperature': opt_temp, 'Concentration': opt_conc, 'Esterified': 0})
        sim_input_base = create_features(sim_input_base)[X_cols_raw]
        sim_input_base_trans = global_vif.transform(global_interact.transform(sim_input_base))
        curve_preds_base = np.maximum(best_estimator.predict(sim_input_base_trans), config.PREDICTION_LOWER_BOUND)
        curve_cof_base = curve_preds_base[:, 0]

        # Calculate Run-in Time for the optimal curve
        smoothed_model = pd.Series(curve_cof_model).rolling(60, min_periods=1).mean().values
        tail_len_model = max(100, int(len(smoothed_model) * 0.1))
        tail_data_model = smoothed_model[-tail_len_model:]
        final_mean_model = np.mean(tail_data_model)
        final_std_model = np.std(tail_data_model)
        tol_model = max(3 * final_std_model, 0.05 * final_mean_model)
        outside_model = np.where(np.abs(smoothed_model - final_mean_model) > tol_model)[0]
        run_in_model = curve_time_model[outside_model[-1]] if len(outside_model) > 0 else 0

        # Generate Plot
        fig, ax = plt.subplots(figsize=(6.3, 3.15))
        plt.plot(curve_time_model, curve_cof_model, label=f'Optimized (Esterified={config.PLOT_ESTERIFIED_STATE})', color='orange')
        plt.plot(curve_time_model, curve_cof_base, label='Base Oil (Esterified=0)', color='purple', linestyle='--')
        plt.ylim(config.PLOT_SETTINGS['cof_ylim'])
        if run_in_model > 0:
            plt.axvline(x=run_in_model, color='grey', linestyle='--', label='Run-in time')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        plt.xlabel('Time [s]')
        plt.ylabel('Coefficient of friction (COF) [-]')
        ymin, ymax = plt.gca().get_ylim()
        plt.ylim(ymin, ymax + (ymax - ymin) * 0.35)
        plt.legend(loc='upper right')
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        curve_filename = f"optimum_curve_{safe_name}.png"
        ester_text = "Esterified" if config.PLOT_ESTERIFIED_STATE == 1 else "Not esterified"
        dynamic_descriptions[curve_filename] = f"Optimum Curve - {name} ({opt_conc:.2f}% | {int(opt_load)}N | {int(opt_temp)}°C, {ester_text}). Run-in time: {run_in_model:.1f} s."
        plt.savefig(os.path.join(config.RESULTS_DIR, curve_filename), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        results.append({
            "Name": name, "Model": best_estimator, "R2_Train": r2_train, "R2_Test": r2_test, "R2_COF": r2_cof, "R2_FAI": r2_fai, "R2_CV": avg_r2,
            "R2_Steady_COF": r2_steady_cof, "R2_Steady_FAI": r2_steady_fai,
            "RMSE_Train": rmse_train, "RMSE_Test": rmse_test, "RMSE_COF": rmse_cof, "RMSE_FAI": rmse_fai, "MAE_Test": mae_test,
            "RMSE_Steady_COF": rmse_steady_cof, "RMSE_Steady_FAI": rmse_steady_fai,
            "Tuning_Training_Time": tuning_training_time, "Pred_Time_ms": pred_time_ms, "Feature_Imp": feature_imp,
            "Opt_Conc": opt_conc, "Opt_Load": opt_load, "Opt_Temp": opt_temp, "Pred_COF": pred_cof_5m, "Pred_FAI": pred_fai_5m,
            "Best_Params": best_params, "Selected_Features": selected_features_model,
            "RunIn_Time": run_in_model, "Opt_Curve_File": curve_filename
        })
        
        if avg_r2 > best_r2_overall:
            best_r2_overall = avg_r2
            best_model_overall = best_estimator
            best_model_name = name

    # --- Creating Ensemble (Top 3 Voting) ---
    print("\n--- Creating Ensemble (Top 3 Voting) ---")

    top3_results = sorted(results, key=lambda x: x['R2_CV'], reverse=True)[:3]
    
    has_smooth = any("Ridge" in r['Name'] or "Neural Network" in r['Name'] for r in top3_results)
    if not has_smooth:
        smooth_models = [r for r in results if "Ridge" in r['Name'] or "Neural Network" in r['Name']]
        if smooth_models:
            best_smooth = sorted(smooth_models, key=lambda x: x['R2_CV'], reverse=True)[0]
            top3_results[2] = best_smooth

    top3_models = [r['Model'] for r in top3_results]
    top3_names = [r['Name'] for r in top3_results]
    print(f"Top 3 models selected for ensemble: {', '.join(top3_names)}")
    
    ensemble_name = "Ensemble (Top 3 Voting)"
    estimators_list = [(name, model) for name, model in zip(top3_names, top3_models)]

    raw_preds = np.array([model.predict(X_test) for model in top3_models]) # Shape: (3, N, 2)
    y_true_vals = y_test.values
    
    def ensemble_loss(w):
        w = w / np.sum(w)
        weighted_pred = np.average(raw_preds, axis=0, weights=w)
        return mean_squared_error(y_true_vals, weighted_pred)

    init_weights = np.ones(len(top3_models)) / len(top3_models)
    bounds = [(0, 1) for _ in range(len(top3_models))]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    opt_res = minimize(ensemble_loss, init_weights, bounds=bounds, constraints=constraints, method='SLSQP')
    optimized_weights = opt_res.x / np.sum(opt_res.x)
    weights_str = ", ".join([f"{name}: {w:.2f}" for name, w in zip(top3_names, optimized_weights)])
    print(f"Optimized Ensemble Weights: {weights_str}")

    ensemble_model = PreFittedVotingRegressor(estimators=estimators_list, weights=optimized_weights)
    
    start_pred = time.time()
    y_pred_ens = np.maximum(ensemble_model.predict(X_test), config.PREDICTION_LOWER_BOUND)
    pred_time_ms_ens = (time.time() - start_pred) * 1000
    
    y_train_pred_ens = np.maximum(ensemble_model.predict(X_train), config.PREDICTION_LOWER_BOUND)
    r2_train_ens = r2_score(y_train, y_train_pred_ens)
    rmse_train_ens = np.sqrt(mean_squared_error(y_train, y_train_pred_ens))
    
    r2_test_ens = r2_score(y_test, y_pred_ens)
    r2_test_raw_ens = r2_score(y_test, y_pred_ens, multioutput='raw_values')
    r2_cof_ens, r2_fai_ens = r2_test_raw_ens[0], r2_test_raw_ens[1]
    
    rmse_test_ens = np.sqrt(mean_squared_error(y_test, y_pred_ens))
    rmse_test_raw_ens = np.sqrt(mean_squared_error(y_test, y_pred_ens, multioutput='raw_values'))
    rmse_cof_ens, rmse_fai_ens = rmse_test_raw_ens[0], rmse_test_raw_ens[1]
    mae_test_ens = mean_absolute_error(y_test, y_pred_ens)
    
    steady_actual_cof_ens = []
    steady_actual_fai_ens = []
    steady_pred_cof_ens = []
    steady_pred_fai_ens = []
    for fid in test_files:
        file_data = full_df[full_df['File_ID'] == fid]
        max_t = file_data['Time'].max()
        last_5m = file_data[file_data['Time'] >= max_t - 300]
        steady_actual_cof_ens.append(last_5m['COF'].mean())
        steady_actual_fai_ens.append(last_5m['Friction absolute integral'].mean())
        
        phys_load, phys_temp, phys_conc, phys_est = file_data.iloc[0][['Load', 'Temperature', 'Concentration', 'Esterified']]
        check_df_st = pd.DataFrame({'Time': [7050], 'Load': [phys_load], 'Temperature': [phys_temp], 'Concentration': [phys_conc], 'Esterified': [phys_est]})
        check_df_st = create_features(check_df_st)[X_cols_raw]
        check_df_st_trans = global_vif.transform(global_interact.transform(check_df_st))
        preds_st = np.maximum(ensemble_model.predict(check_df_st_trans), config.PREDICTION_LOWER_BOUND)
        steady_pred_cof_ens.append(preds_st[0, 0])
        steady_pred_fai_ens.append(preds_st[0, 1])
        
    r2_steady_cof_ens = r2_score(steady_actual_cof_ens, steady_pred_cof_ens)
    r2_steady_fai_ens = r2_score(steady_actual_fai_ens, steady_pred_fai_ens)
    rmse_steady_cof_ens = np.sqrt(mean_squared_error(steady_actual_cof_ens, steady_pred_cof_ens))
    rmse_steady_fai_ens = np.sqrt(mean_squared_error(steady_actual_fai_ens, steady_pred_fai_ens))

    avg_r2_ens = np.mean([r['R2_CV'] for r in top3_results])
    tuning_training_time_ens = sum([r['Tuning_Training_Time'] for r in top3_results])
    
    preds_grid_ens = np.maximum(ensemble_model.predict(grid_df_trans), config.PREDICTION_LOWER_BOUND)
    norm_cof_ens = (preds_grid_ens[:,0] - preds_grid_ens[:,0].min()) / (preds_grid_ens[:,0].max() - preds_grid_ens[:,0].min() + 1e-9)
    norm_fai_ens = (preds_grid_ens[:,1] - preds_grid_ens[:,1].min()) / (preds_grid_ens[:,1].max() - preds_grid_ens[:,1].min() + 1e-9)
    scores_ens = norm_cof_ens + norm_fai_ens
    best_idx_ens = np.argmin(scores_ens)
    
    opt_conc_ens = grid_df.iloc[best_idx_ens]['Concentration']
    opt_load_ens = grid_df.iloc[best_idx_ens]['Load']
    opt_temp_ens = grid_df.iloc[best_idx_ens]['Temperature']
    
    check_df_ens = pd.DataFrame({'Time': np.arange(6900, 7201, 10), 'Load': opt_load_ens, 'Temperature': opt_temp_ens, 'Concentration': opt_conc_ens, 'Esterified': config.PLOT_ESTERIFIED_STATE})
    check_df_ens = create_features(check_df_ens)[X_cols_raw]
    check_df_trans_ens = global_vif.transform(global_interact.transform(check_df_ens))
    check_preds_ens = np.maximum(ensemble_model.predict(check_df_trans_ens), config.PREDICTION_LOWER_BOUND)
    pred_cof_5m_ens = np.mean(check_preds_ens[:, 0])
    pred_fai_5m_ens = np.mean(check_preds_ens[:, 1])

    sim_input_model_ens = pd.DataFrame({'Time': template_df['Time'], 'Load': opt_load_ens, 'Temperature': opt_temp_ens, 'Concentration': opt_conc_ens, 'Esterified': config.PLOT_ESTERIFIED_STATE})
    sim_input_model_ens = create_features(sim_input_model_ens)[X_cols_raw]
    sim_input_model_trans_ens = global_vif.transform(global_interact.transform(sim_input_model_ens))
    curve_preds_model_ens = np.maximum(ensemble_model.predict(sim_input_model_trans_ens), config.PREDICTION_LOWER_BOUND)
    curve_cof_model_ens = curve_preds_model_ens[:, 0]
    curve_time_model_ens = template_df['Time'].values

    sim_input_base_ens = pd.DataFrame({'Time': template_df['Time'], 'Load': opt_load_ens, 'Temperature': opt_temp_ens, 'Concentration': opt_conc_ens, 'Esterified': 0})
    sim_input_base_ens = create_features(sim_input_base_ens)[X_cols_raw]
    sim_input_base_trans_ens = global_vif.transform(global_interact.transform(sim_input_base_ens))
    curve_preds_base_ens = np.maximum(ensemble_model.predict(sim_input_base_trans_ens), config.PREDICTION_LOWER_BOUND)
    curve_cof_base_ens = curve_preds_base_ens[:, 0]

    smoothed_model_ens = pd.Series(curve_cof_model_ens).rolling(60, min_periods=1).mean().values
    tail_len_model_ens = max(100, int(len(smoothed_model_ens) * 0.1))
    tail_data_model_ens = smoothed_model_ens[-tail_len_model_ens:]
    final_mean_model_ens = np.mean(tail_data_model_ens)
    final_std_model_ens = np.std(tail_data_model_ens)
    tol_model_ens = max(3 * final_std_model_ens, 0.05 * final_mean_model_ens)
    outside_model_ens = np.where(np.abs(smoothed_model_ens - final_mean_model_ens) > tol_model_ens)[0]
    run_in_model_ens = curve_time_model_ens[outside_model_ens[-1]] if len(outside_model_ens) > 0 else 0

    fig, ax = plt.subplots(figsize=(6.3, 3.15))
    plt.plot(curve_time_model_ens, curve_cof_model_ens, label=f'Optimized (Esterified={config.PLOT_ESTERIFIED_STATE})', color='orange')
    plt.plot(curve_time_model_ens, curve_cof_base_ens, label='Base Oil (Esterified=0)', color='purple', linestyle='--')
    plt.ylim(config.PLOT_SETTINGS['cof_ylim'])
    if run_in_model_ens > 0:
        plt.axvline(x=run_in_model_ens, color='grey', linestyle='--', label='Run-in time')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    plt.xlabel('Time [s]')
    plt.ylabel('Coefficient of friction (COF) [-]')
    ymin, ymax = plt.gca().get_ylim()
    plt.ylim(ymin, ymax + (ymax - ymin) * 0.35)
    plt.legend(loc='upper right')
    safe_name_ens = ensemble_name.replace(' ', '_').replace('(', '').replace(')', '')
    curve_filename_ens = f"optimum_curve_{safe_name_ens}.png"
    ester_text_ens = "Esterified" if config.PLOT_ESTERIFIED_STATE == 1 else "Not esterified"
    dynamic_descriptions[curve_filename_ens] = f"Optimum Curve - {ensemble_name} ({opt_conc_ens:.2f}% | {int(opt_load_ens)}N | {int(opt_temp_ens)}°C, {ester_text_ens}). Run-in time: {run_in_model_ens:.1f} s."
    plt.savefig(os.path.join(config.RESULTS_DIR, curve_filename_ens), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

    results.append({
        "Name": ensemble_name, "Model": ensemble_model, "R2_Train": r2_train_ens, "R2_Test": r2_test_ens, "R2_COF": r2_cof_ens, "R2_FAI": r2_fai_ens, "R2_CV": avg_r2_ens,
        "R2_Steady_COF": r2_steady_cof_ens, "R2_Steady_FAI": r2_steady_fai_ens,
        "RMSE_Train": rmse_train_ens, "RMSE_Test": rmse_test_ens, "RMSE_COF": rmse_cof_ens, "RMSE_FAI": rmse_fai_ens, "MAE_Test": mae_test_ens,
        "RMSE_Steady_COF": rmse_steady_cof_ens, "RMSE_Steady_FAI": rmse_steady_fai_ens,
        "Tuning_Training_Time": tuning_training_time_ens, "Pred_Time_ms": pred_time_ms_ens, "Feature_Imp": None,
        "Opt_Conc": opt_conc_ens, "Opt_Load": opt_load_ens, "Opt_Temp": opt_temp_ens, "Pred_COF": pred_cof_5m_ens, "Pred_FAI": pred_fai_5m_ens,
        "Best_Params": f"Optimized Weights -> {weights_str}", "Selected_Features": global_vif.selected_features_,
        "RunIn_Time": run_in_model_ens, "Opt_Curve_File": curve_filename_ens
    })

    if avg_r2_ens > best_r2_overall:
        best_r2_overall = avg_r2_ens
        best_model_overall = ensemble_model
        best_model_name = ensemble_name

    if config.USE_CACHE:
        print("\nSaving trained models to cache...")
        joblib.dump({
            'results': results,
            'best_model_overall': best_model_overall,
            'best_model_name': best_model_name
        }, models_cache_path)

print(f"\nBest model found: {best_model_name} with average R2 CV: {best_r2_overall:.4f}")

print("\n--- Filtering high error values before final retraining... ---")
full_preds = np.maximum(best_model_overall.predict(X), config.PREDICTION_LOWER_BOUND)
full_residuals = full_preds[:, 0] - Y['COF'].values
valid_mask = np.abs(full_residuals) <= 0.05

df_mask = pd.DataFrame({
    'File_ID': full_df['File_ID'].values, 
    'Valid': valid_mask, 
    'AbsError': np.abs(full_residuals)
})
dropped_files = df_mask.groupby('File_ID')['Valid'].sum()
completely_dropped = dropped_files[dropped_files == 0].index

if len(completely_dropped) > 0:
    print(f"\nWarning! {len(completely_dropped)} files would be completely dropped. Keeping their lowest error point:")
    for fid in completely_dropped:
        min_err_idx = df_mask[df_mask['File_ID'] == fid]['AbsError'].idxmin()
        valid_mask[min_err_idx] = True
        print(f"  - {fid} (Saved point error: {df_mask.loc[min_err_idx, 'AbsError']:.4f})")

dropped_count = np.sum(~valid_mask)
print(f"Number of removed anomalies (|Error| > 0.05): {dropped_count} points out of {len(X)}.")

if dropped_count > 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(full_df['Time'].values[valid_mask], full_residuals[valid_mask], color='blue', alpha=0.2, s=10, label='Points retained')
    ax.scatter(full_df['Time'].values[~valid_mask], full_residuals[~valid_mask], color='red', alpha=0.8, s=15, label='Excluded points (|Error| > 0.05)')
    ax.axhline(0.05, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(-0.05, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='grey', linestyle='-', linewidth=1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Residual error")
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.85, edgecolor='black', fancybox=True)
    dynamic_descriptions["Dropped_Anomalies.png"] = "Scatter plot showing all data points. Red points indicate anomalies (|Error| > 0.05) that were dropped before final model retraining."
    plt.savefig(os.path.join(config.RESULTS_DIR, "Dropped_Anomalies.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

X_filtered = X[valid_mask]
Y_filtered = Y[valid_mask]
weights_filtered = full_df['Sample_Weight'].values[valid_mask]

print(f"Retraining {best_model_name} on the filtered full dataset...")

fit_params_full = {}
if "Ensemble" in best_model_name: fit_params_full['sample_weight'] = weights_filtered
elif "Random Forest" in best_model_name: fit_params_full['rf__sample_weight'] = weights_filtered
elif "XGBoost" in best_model_name: fit_params_full['xgb__sample_weight'] = weights_filtered
elif "LightGBM" in best_model_name: fit_params_full['lgbm__sample_weight'] = weights_filtered
elif "CatBoost" in best_model_name: fit_params_full['cat__sample_weight'] = weights_filtered
elif "Polynomial" in best_model_name: fit_params_full['ridge__sample_weight'] = weights_filtered
best_model_overall.fit(X_filtered, Y_filtered, **fit_params_full)

optimum_results = {}

print("\n--- Calculating Optimums over the Parameter Grid ---")

pareto_grid_list = []
for ester_state in [0, 1]:
    temp_grid = grid_df.copy()
    temp_grid['Esterified'] = ester_state
    pareto_grid_list.append(temp_grid)
pareto_grid_df = pd.concat(pareto_grid_list, ignore_index=True)
pareto_grid_df = create_features(pareto_grid_df)
pareto_grid_trans = global_vif.transform(global_interact.transform(pareto_grid_df))
pareto_preds = np.maximum(best_model_overall.predict(pareto_grid_trans), config.PREDICTION_LOWER_BOUND)

pareto_configs = [
    ('Temperature', 'Temperature [°C]', 'Pareto_Temperature.png', False),
    ('Load', 'Load [N]', 'Pareto_Load.png', False),
    ('Concentration', 'Concentration [wt%]', 'Pareto_Concentration.png', True),
    ('Esterified', 'Esterified State (0/1)', 'Pareto_Esterified.png', True)
]
for col, label, fname, is_discrete in pareto_configs:
    knee_cof, knee_fai, knee_idx = plot_pareto_front(config.RESULTS_DIR, pareto_preds, pareto_grid_df[col], color_label=label, title=f"Pareto front - Colored by {col}", filename=fname, discrete=is_discrete)
    opt_row = pareto_grid_df.iloc[knee_idx]
    dynamic_descriptions[fname] = f"Pareto front over the full parameter grid, colored by {col}. Knee point (Trade-off optimum) achieved at: {opt_row['Concentration']:.2f} wt% Conc, {int(opt_row['Load'])} N Load, {int(opt_row['Temperature'])} °C Temp, Esterified: {int(opt_row['Esterified'])} (COF: {knee_cof:.3f}, FAI: {knee_fai:.3f})."

for ester_state in [0, 1]:
    grid_df['Esterified'] = ester_state
    grid_df = create_features(grid_df)
    grid_df_trans = global_vif.transform(global_interact.transform(grid_df))
    preds = np.maximum(best_model_overall.predict(grid_df_trans), config.PREDICTION_LOWER_BOUND)
    
    norm_cof = (preds[:,0] - preds[:,0].min()) / (preds[:,0].max() - preds[:,0].min() + 1e-9)
    norm_fai = (preds[:,1] - preds[:,1].min()) / (preds[:,1].max() - preds[:,1].min() + 1e-9)
    scores = norm_cof + norm_fai
    best_idx = np.argmin(scores)
    
    opt_conc = grid_df.iloc[best_idx]['Concentration']
    opt_load = grid_df.iloc[best_idx]['Load']
    opt_temp = grid_df.iloc[best_idx]['Temperature']
    
    check_df = pd.DataFrame({'Time': np.arange(6900, 7201, 10), 'Load': opt_load, 'Temperature': opt_temp, 'Concentration': opt_conc, 'Esterified': ester_state})
    check_df = create_features(check_df)[X_cols_raw]
    check_df_trans = global_vif.transform(global_interact.transform(check_df))
    check_preds = np.maximum(best_model_overall.predict(check_df_trans), config.PREDICTION_LOWER_BOUND)
    avg_cof_last5 = np.mean(check_preds[:, 0])
    avg_fai_last5 = np.mean(check_preds[:, 1])
    
    sim_input = pd.DataFrame({'Time': template_df['Time'], 'Load': opt_load, 'Temperature': opt_temp, 'Concentration': opt_conc, 'Esterified': ester_state})
    sim_input = create_features(sim_input)[X_cols_raw]
    sim_input_trans = global_vif.transform(global_interact.transform(sim_input))
    curve_preds = np.maximum(best_model_overall.predict(sim_input_trans), config.PREDICTION_LOWER_BOUND)
    curve_cof = curve_preds[:, 0]
    curve_time = template_df['Time'].values
    
    smoothed = pd.Series(curve_cof).rolling(60, min_periods=1).mean().values
    tail_len = max(100, int(len(smoothed) * 0.1))
    tail_data = smoothed[-tail_len:]
    final_mean = np.mean(tail_data)
    final_std = np.std(tail_data)
    tol = max(3 * final_std, 0.05 * final_mean)
    outside = np.where(np.abs(smoothed - final_mean) > tol)[0]
    run_in = curve_time[outside[-1]] if len(outside) > 0 else 0
    run_in_str = "Immediate stabilization" if run_in == 0 else f"{run_in:.1f} s"
    
    stab_inputs = []
    for l in [opt_load * 0.9, opt_load, opt_load * 1.1]:
        for t in [opt_temp * 0.9, opt_temp, opt_temp * 1.1]:
            if l == opt_load and t == opt_temp: continue
            stab_inputs.append({'Time': 7050, 'Load': l, 'Temperature': t, 'Concentration': opt_conc, 'Esterified': ester_state})
    
    if stab_inputs:
        stab_df = create_features(pd.DataFrame(stab_inputs))[X_cols_raw]
        stab_df_trans = global_vif.transform(global_interact.transform(stab_df))
        stab_preds = np.maximum(best_model_overall.predict(stab_df_trans), config.PREDICTION_LOWER_BOUND)
        max_dev_percent = np.max(np.abs(stab_preds[:, 0] - avg_cof_last5) / avg_cof_last5) * 100
        stability_status = "Stable" if max_dev_percent < 5.0 else "Unstable"
    else:
        stability_status = "N/A"
        
    optimum_results[ester_state] = {
        'Conc': opt_conc, 'Load': opt_load, 'Temp': opt_temp, 'COF': avg_cof_last5, 'FAI': avg_fai_last5, 
        'RunIn': run_in, 'RunInStr': run_in_str, 'Stability': stability_status, 'CurveTime': curve_time, 'CurveCOF': curve_cof
    }

# --- DoE Calculation ---
print("\n--- Calculating DoE points... ---")

all_doe_suggestions = {}
total_doe_duration = 0
safe_validation_points = {}

for ester_state_doe in [0, 1]:
    state_str = "Esterified" if ester_state_doe == 1 else "Base Oil"
    doe_cache_path = os.path.join(config.CACHE_DIR, f"doe_cache_{ester_state_doe}.pkl")
    
    if config.USE_CACHE and os.path.exists(doe_cache_path):
        print(f"\n--- Loading DoE Suggestions From Cache for {state_str} ---")
        cached_doe = joblib.load(doe_cache_path)
        doe_suggestions = cached_doe['doe_suggestions']
        doe_duration = cached_doe['doe_duration']
        best_safe = cached_doe.get('best_safe', None)
        if best_safe is not None:
            print(f"\n>> Recommended Safe Validation Point ({state_str}):")
            print(f"   Concentration: {best_safe['Concentration']:.2f} wt%, Load: {int(best_safe['Load'])} N, Temperature: {int(best_safe['Temperature'])} °C")
            print(f"   Expected COF: {best_safe['Predicted_COF']:.4f} (Uncertainty: {best_safe['Avg_Uncertainty']:.4f})")
            
    else:
        print(f"\n--- Starting DoE generation for {state_str} (Cache not found or disabled) ---")
        start_doe = time.time()
        
        top3_doe_models = [r['Model'] for r in sorted([res for res in results if "Voting" not in res['Name']], key=lambda x: x['R2_CV'], reverse=True)[:3]]
        
        doe_combos = list(itertools.product(range_conc, range_load, range_temp, [ester_state_doe]))
        doe_grid_df = pd.DataFrame(doe_combos, columns=['Concentration', 'Load', 'Temperature', 'Esterified'])
        doe_grid_df['Time'] = 7050
        doe_grid_df = create_features(doe_grid_df)[X_cols_raw]
        
        grid_doe = global_vif.transform(global_interact.transform(doe_grid_df))

        print("Predicting uncertainty on the parameter grid...")
        doe_preds = np.array([np.maximum(model.predict(grid_doe), config.PREDICTION_LOWER_BOUND) for model in top3_doe_models])
        std_cof = np.std(doe_preds[:, :, 0], axis=0)
        std_fai = np.std(doe_preds[:, :, 1], axis=0)

        doe_features = ['Concentration', 'Load', 'Temperature', 'Esterified']
        scaler_doe = MinMaxScaler()
        X_grid_scaled = pd.DataFrame(scaler_doe.fit_transform(doe_grid_df[doe_features]), columns=doe_features)
        
        existing_subset = full_df[full_df['Esterified'] == ester_state_doe]
        X_existing_scaled = pd.DataFrame(scaler_doe.transform(existing_subset[doe_features]), columns=doe_features)

        nbrs = NearestNeighbors(n_neighbors=1).fit(X_existing_scaled)
        full_dist_metric = nbrs.kneighbors(X_grid_scaled)[0].flatten()

        norm_std_cof = (std_cof - std_cof.min()) / (std_cof.max() - std_cof.min() + 1e-9)
        norm_std_fai = (std_fai - std_fai.min()) / (std_fai.max() - std_fai.min() + 1e-9)
        avg_uncertainty = (norm_std_cof + norm_std_fai) / 2

        doe_grid = doe_grid_df.copy()
        doe_grid['Avg_Uncertainty'] = avg_uncertainty

        existing_set = set((round(row['Concentration'], 2), int(row['Load']), int(row['Temperature']), int(row['Esterified'])) for _, row in existing_subset.iterrows())
        doe_candidates = doe_grid[~doe_grid.apply(lambda row: (round(row['Concentration'], 2), int(row['Load']), int(row['Temperature']), int(row['Esterified'])) in existing_set, axis=1)]

        final_suggestions = []
        candidates_pool = doe_candidates.copy()
        current_existing_scaled = X_existing_scaled.values.tolist()
        
        for _ in range(5):
            if candidates_pool.empty: break
            
            nbrs = NearestNeighbors(n_neighbors=1).fit(current_existing_scaled)
            X_cand_scaled = scaler_doe.transform(candidates_pool[doe_features])
            cand_dist_metric = nbrs.kneighbors(X_cand_scaled)[0].flatten()
            
            norm_dist = (cand_dist_metric - cand_dist_metric.min()) / (cand_dist_metric.max() - cand_dist_metric.min() + 1e-9)
            
            candidates_pool['Distance'] = cand_dist_metric
            candidates_pool['Norm_Distance'] = norm_dist
            candidates_pool['Score'] = config.UNCERTAINTY_WEIGHT * candidates_pool['Avg_Uncertainty'] + config.SPARSITY_WEIGHT * candidates_pool['Norm_Distance']
            candidates_pool = candidates_pool.sort_values(by='Score', ascending=False)
            
            best_candidate = candidates_pool.iloc[0]
            
            idx_in_grid = best_candidate.name
            best_candidate_dict = best_candidate.to_dict()
            best_candidate_dict['Uncertainty_COF'] = std_cof[idx_in_grid]
            best_candidate_dict['Uncertainty_FAI'] = std_fai[idx_in_grid]
            
            final_suggestions.append(best_candidate_dict)
            
            best_scaled = scaler_doe.transform(pd.DataFrame([best_candidate[doe_features]], columns=doe_features))
            current_existing_scaled.append(best_scaled[0].tolist())
            
            candidates_pool = candidates_pool.drop(best_candidate.name)

        doe_suggestions = pd.DataFrame(final_suggestions)
        doe_duration = time.time() - start_doe

        # --- FIND SAFE VALIDATION POINT ---
        doe_grid['Predicted_COF'] = np.mean(doe_preds[:, :, 0], axis=0)
        doe_grid['Distance'] = full_dist_metric
        safe_mask = (doe_grid['Distance'] < doe_grid['Distance'].quantile(0.25)) & (doe_grid['Avg_Uncertainty'] < doe_grid['Avg_Uncertainty'].quantile(0.25))
        best_safe = None
        if safe_mask.any():
            safe_cands = doe_grid[safe_mask].sort_values(by='Predicted_COF', ascending=True)
            best_safe = safe_cands.iloc[0].to_dict()
            print(f"\n>> Recommended Safe Validation Point ({state_str}):")
            print(f"   Concentration: {best_safe['Concentration']:.2f} wt%, Load: {int(best_safe['Load'])} N, Temperature: {int(best_safe['Temperature'])} °C")
            print(f"   Expected COF: {best_safe['Predicted_COF']:.4f} (Uncertainty: {best_safe['Avg_Uncertainty']:.4f})")

        if config.USE_CACHE:
            print(f"Saving DoE suggestions for {state_str} to cache...")
            joblib.dump({'doe_suggestions': doe_suggestions, 'doe_duration': doe_duration, 'best_safe': best_safe}, doe_cache_path)
    
    all_doe_suggestions[ester_state_doe] = doe_suggestions
    total_doe_duration += doe_duration
    safe_validation_points[ester_state_doe] = best_safe

doe_img_files = []
doe_suggestions_combined = pd.concat(all_doe_suggestions.values()).reset_index(drop=True)
for i, (_, row) in enumerate(doe_suggestions_combined.iterrows()):
    sim_input = create_features(pd.DataFrame({'Time': template_df['Time'], 'Load': row['Load'], 'Temperature': row['Temperature'], 'Concentration': row['Concentration'], 'Esterified': row['Esterified']}))[X_cols_raw]
    sim_input_trans = global_vif.transform(global_interact.transform(sim_input))
    curve_preds = np.maximum(best_model_overall.predict(sim_input_trans), config.PREDICTION_LOWER_BOUND)
    fig, ax = plt.subplots(figsize=(6.3, 3.15))
    plt.plot(template_df['Time'], curve_preds[:, 0], color='purple')
    plt.ylim(bottom=0.0, top=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    plt.xlabel('Time [s]')
    plt.ylabel('Coefficient of friction (COF) [-]')
    fname = f"DoE_Suggestion_{i+1}.png"
    ester_str = "Esterified" if row['Esterified'] == 1 else "Not esterified"
    dynamic_descriptions[fname] = f"DoE suggestion #{i+1}: {row['Concentration']:.2f}% | {int(row['Load'])}N | {int(row['Temperature'])}°C | {ester_str} (Predicted by: {best_model_name})."
    plt.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()
    doe_img_files.append(fname)

print("\n--- Generating Feature Importance Plot ---")
best_res = next(r for r in results if r['Name'] == best_model_name)
if best_res['Feature_Imp'] is not None:
    print(f"Generating feature importance plot for {best_model_name}...")
    sorted_idx = np.argsort(best_res['Feature_Imp'])
    sorted_feats = [best_res['Selected_Features'][i] for i in sorted_idx]
    sorted_imp = best_res['Feature_Imp'][sorted_idx]
    
    display_feats = [config.NAME_MAPPING.get(f, f) for f in sorted_feats]
    
    plt.figure(figsize=(6.3, 3.15))
    plt.barh(display_feats, sorted_imp, color='purple')
    plt.xlabel("Feature Importance")
    dynamic_descriptions["Feature_importance.png"] = f"Feature importance ({best_model_name})."
    plt.savefig(os.path.join(config.RESULTS_DIR, "Feature_importance.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("\n--- SHAP Analysis ---")
shap_analysis_text = ""
shap_duration = None
tree_models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]

tree_results = [r for r in results if any(m in r['Name'] for m in tree_models)]
if tree_results:
    best_tree_res = sorted(tree_results, key=lambda x: x['R2_CV'], reverse=True)[0]
    shap_model = best_tree_res['Model']
    shap_model_name = best_tree_res['Name']
    print(f"Generating SHAP analysis for the best tree-based model: {shap_model_name}...")

    try:
        plt.rcParams['xtick.minor.visible'] = False
        plt.rcParams['ytick.minor.visible'] = False

        start_shap = time.time()

        if len(X_test) > config.SHAP_SAMPLE_SIZE:
            print(f"Subsampling X_test for SHAP analysis from {len(X_test)} to {config.SHAP_SAMPLE_SIZE} instances for performance.")
            X_test_shap = X_test.sample(n=config.SHAP_SAMPLE_SIZE, random_state=config.RANDOM_SEED)
        else:
            X_test_shap = X_test

        pipeline = shap_model.regressor_ if hasattr(shap_model, 'regressor_') else shap_model
        scaler_step = pipeline.named_steps['scaler']
        
        X_test_vif = X_test_shap
        vif_feature_names = global_vif.get_feature_names_out()
        
        X_test_scaled = pd.DataFrame(
            scaler_step.transform(X_test_vif), 
            columns=vif_feature_names, 
            index=X_test_shap.index
        )
        
        X_test_display = pd.DataFrame(X_test_vif.values, index=X_test_shap.index, columns=vif_feature_names)
        safe_mapping = {k: v.replace(' [N]', '').replace(' [°C]', '') for k, v in config.NAME_MAPPING.items() if k in X_test_display.columns}
        X_test_display.rename(columns=safe_mapping, inplace=True)
        
        model_step_name = None
        if "XGBoost" in shap_model_name: model_step_name = 'xgb'
        elif "LightGBM" in shap_model_name: model_step_name = 'lgbm'
        elif "CatBoost" in shap_model_name: model_step_name = 'cat'
        elif "Random Forest" in shap_model_name: model_step_name = 'rf'

        if model_step_name:
            if model_step_name == 'rf':
                model_obj = pipeline.named_steps[model_step_name]
            else:
                model_obj = pipeline.named_steps[model_step_name].estimators_[0]
                
            explainer = shap.TreeExplainer(model_obj)
            print("Calculating SHAP values... (this may take a while)")
            shap_values = explainer.shap_values(X_test_scaled, check_additivity=False)
            print("SHAP values calculated. Generating plots...")
            
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[0]
            else:
                shap_values_to_plot = shap_values

            plot_progress = tqdm(total=8, desc="Generating SHAP plots", unit="plot")

            plt.figure(figsize=(6.3, 3.15))
            shap.summary_plot(shap_values_to_plot, X_test_display, show=False)
            fig = plt.gcf()
            fig.set_facecolor('white')
            ax = plt.gca()
            ax.set_facecolor('white')
            ax.grid(False)
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_feature_impact.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
            plt.close()
            plot_progress.update(1)

            plt.figure(figsize=(6.3, 3.15))
            shap.summary_plot(shap_values_to_plot, X_test_display, plot_type="bar", show=False)
            fig = plt.gcf()
            fig.set_facecolor('white')
            ax = plt.gca()
            ax.set_facecolor('white')
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_bar_plot.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
            plt.close()
            dynamic_descriptions["SHAP_bar_plot.png"] = "Global feature importance based on mean absolute SHAP values."
            plot_progress.update(1)

            plt.figure(figsize=(6.3, 3.15))
            shap.dependence_plot("Load [N]", shap_values_to_plot, X_test_display, interaction_index="Esterified", show=False, ax=plt.gca())
            fig = plt.gcf()
            fig.set_facecolor('white')
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_Dependence_Load.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
            plt.close()
            plot_progress.update(1)
            
            plt.figure(figsize=(6.3, 3.15))
            shap.dependence_plot("Temperature [°C]", shap_values_to_plot, X_test_display, interaction_index="Esterified", show=False, ax=plt.gca())
            fig = plt.gcf()
            fig.set_facecolor('white')
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_Dependence_Temperature.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
            plt.close()
            plot_progress.update(1)

            plt.figure(figsize=(8, 5))
            expected_val = explainer.expected_value
            if isinstance(expected_val, (list, np.ndarray)):
                expected_val = expected_val[0]
            
            exp = shap.Explanation(values=shap_values_to_plot[0], 
                                   base_values=expected_val, 
                                   data=X_test_display.iloc[0].values, 
                                   feature_names=X_test_display.columns)
            shap.plots.waterfall(exp, show=False)
            fig = plt.gcf()
            fig.set_facecolor('white')
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_waterfall_plot.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
            plt.close()
            dynamic_descriptions["SHAP_waterfall_plot.png"] = "SHAP Waterfall Plot explaining the specific prediction of the first test instance."
            plot_progress.update(1)

            plt.figure(figsize=(10, 3))
            shap.force_plot(expected_val, shap_values_to_plot[0], X_test_display.iloc[0].round(3), matplotlib=True, show=False)
            fig = plt.gcf()
            fig.set_facecolor('white')
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_force_plot.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
            plt.close()
            dynamic_descriptions["SHAP_force_plot.png"] = "SHAP Force Plot explaining the specific prediction of the first test instance (horizontal alternative to Waterfall)."
            plot_progress.update(1)

            plt.figure(figsize=(8, 5))
            full_exp = shap.Explanation(values=shap_values_to_plot, 
                                        base_values=np.array([expected_val]*len(X_test_display)), 
                                        data=X_test_display.values, 
                                        feature_names=X_test_display.columns)
            shap.plots.heatmap(full_exp, show=False)
            fig = plt.gcf()
            fig.set_facecolor('white')
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_heatmap_plot.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
            plt.close()
            dynamic_descriptions["SHAP_heatmap_plot.png"] = "SHAP Heatmap showing the impact of features across all test instances, ordered by model output. Great for discovering dataset-level patterns."
            plot_progress.update(1)

            plt.figure(figsize=(8, 5))
            shap.decision_plot(expected_val, shap_values_to_plot[:20], features=X_test_display.iloc[:20], show=False)
            fig = plt.gcf()
            fig.set_facecolor('white')
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_decision_plot.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
            plt.close()
            dynamic_descriptions["SHAP_decision_plot.png"] = "SHAP Decision Plot for the first 20 test instances, highlighting cumulative feature effects along the decision path."
            plot_progress.close()

            config.set_academic_plot_style()
            
            mean_shap = np.abs(shap_values_to_plot).mean(axis=0)
            top_3 = sorted(dict(zip(X_test_display.columns, mean_shap)).items(), key=lambda x: x[1], reverse=True)[:3]
            shap_analysis_text = "<ul>" + "".join([f"<li><strong>{f}</strong> (SHAP: {i:.4f})</li>" for f, i in top_3]) + "</ul>"
            shap_duration = time.time() - start_shap
            print("SHAP analysis completed.")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Warning: SHAP analysis failed - {e}")
else:
    print("No tree-based models found for SHAP analysis.")

valid_series = pd.Series(valid_mask, index=X.index)
valid_test_mask = valid_series.loc[X_test.index]
X_test_filtered = X_test[valid_test_mask]
y_test_filtered = y_test[valid_test_mask]
groups_test_filtered = groups.loc[X_test.index][valid_test_mask]

print("\n--- Calculating residual plot data... ---")
y_test_pred = np.maximum(best_model_overall.predict(X_test_filtered), config.PREDICTION_LOWER_BOUND)
residuals = y_test_pred[:, 0] - y_test_filtered['COF'].values

print("\n--- Generating Actual vs Predicted and Residual Histograms for all models... ---")
for res in results:
    model = res['Model']
    model_name = res['Name']
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    preds = np.maximum(model.predict(X_test), config.PREDICTION_LOWER_BOUND)
    
    fig, ax = plt.subplots(figsize=(6.3, 3.15))
    plt.scatter(y_test['COF'].values, preds[:, 0], alpha=0.3, color='royalblue', edgecolors='lightgrey', linewidths=0.5)
    min_val = min(np.min(y_test['COF'].values), np.min(preds[:, 0]))
    max_val = max(np.max(y_test['COF'].values), np.max(preds[:, 0]))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=2)
    plt.xlabel("Actual COF")
    plt.ylabel("Predicted COF")
    plt.title(f"{model_name}", fontsize=11)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    
    filename = f"Actual_vs_Predicted_{safe_name}.png"
    dynamic_descriptions[filename] = f"Scatter plot of Actual vs. Predicted COF values for {model_name}."
    plt.savefig(os.path.join(config.RESULTS_DIR, filename), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

    model_residuals = preds[:, 0] - y_test['COF'].values
    fig, ax = plt.subplots(figsize=(6.3, 3.15))
    plt.hist(model_residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.xlabel("Residual Error (Predicted - Actual)")
    plt.ylabel("Frequency")
    plt.title(f"{model_name}", fontsize=11)
    
    hist_filename = f"Residual_Histogram_{safe_name}.png"
    dynamic_descriptions[hist_filename] = f"Histogram of residual errors for {model_name}."
    plt.savefig(os.path.join(config.RESULTS_DIR, hist_filename), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("\n--- Generating model comparison plots... ---")
sorted_results_for_plots = sorted(results, key=lambda x: x['R2_Train'], reverse=True)
model_names = [res['Name'] for res in sorted_results_for_plots]
r2_train = [res['R2_Train'] for res in sorted_results_for_plots]
r2_test = [res['R2_Test'] for res in sorted_results_for_plots]
r2_cv = [res['R2_CV'] for res in sorted_results_for_plots]

rmse_test = [res['RMSE_Test'] for res in sorted_results_for_plots]
mae_test = [res['MAE_Test'] for res in sorted_results_for_plots]
tuning_times = [res['Tuning_Training_Time'] for res in sorted_results_for_plots]
pred_times = [res['Pred_Time_ms'] for res in sorted_results_for_plots]

x = np.arange(len(model_names))

# 1. R2 Score Plot
width_r2 = 0.25
fig_r2, ax_r2 = plt.subplots(figsize=(10, 5))
ax_r2.set_xlabel('Models', fontweight='bold')
ax_r2.set_ylabel('R² Score', fontweight='bold')

bars_r2_train = ax_r2.bar(x - width_r2, r2_train, width_r2, label='R² (Train)', color='#984ea3', alpha=0.8, edgecolor='black')
bars_r2_cv = ax_r2.bar(x, r2_cv, width_r2, label='R² (CV)', color='#c2a5cf', alpha=0.8, edgecolor='black')
bars_r2_test = ax_r2.bar(x + width_r2, r2_test, width_r2, label='R² (Test)', color='#f1a340', alpha=0.8, edgecolor='black')

all_r2 = r2_train + r2_cv + r2_test
ax_r2.set_ylim(min(0, min(all_r2) * 1.1) if all_r2 else 0, 1.15)
ax_r2.set_xticks(x)
ax_r2.set_xticklabels(model_names, rotation=45, ha='right')
ax_r2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

for bars in [bars_r2_train, bars_r2_cv, bars_r2_test]:
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax_r2.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

fig_r2.tight_layout()
dynamic_descriptions["Model_comparison_R2.png"] = "Grouped bar chart comparing the accuracy (R² Train, CV, Test) of all trained models."
fig_r2.savefig(os.path.join(config.RESULTS_DIR, "Model_comparison_R2.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close(fig_r2)

# 2. Error Score Plot (RMSE, MAE)
width_err = 0.35
fig_err, ax_err = plt.subplots(figsize=(10, 5))
ax_err.set_xlabel('Models', fontweight='bold')
ax_err.set_ylabel('Error Value', fontweight='bold')

bars_rmse = ax_err.bar(x - width_err/2, rmse_test, width_err, label='RMSE (Test)', color='#e66101', alpha=0.8, edgecolor='black')
bars_mae = ax_err.bar(x + width_err/2, mae_test, width_err, label='MAE (Test)', color='#fdb863', alpha=0.8, edgecolor='black')

all_err = rmse_test + mae_test
ax_err.set_ylim(0, 0.04)
ax_err.set_xticks(x)
ax_err.set_xticklabels(model_names, rotation=45, ha='right')
ax_err.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

for bars in [bars_rmse, bars_mae]:
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            ax_err.text(bar.get_x() + bar.get_width()/2, yval + (max(all_err)*0.01), f'{yval:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)

fig_err.tight_layout()
dynamic_descriptions["Model_comparison_Errors.png"] = "Grouped bar chart comparing the errors (RMSE, MAE) of all trained models on the test dataset."
fig_err.savefig(os.path.join(config.RESULTS_DIR, "Model_comparison_Errors.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close(fig_err)

# 3. Time Comparison Plot (Tuning vs Prediction)
fig_time, (ax_train_time, ax_pred_time) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Tuning & Training Time
bars_train_time = ax_train_time.bar(x, tuning_times, color='#4daf4a', alpha=0.8, edgecolor='black')
ax_train_time.set_xlabel('Models', fontweight='bold')
ax_train_time.set_ylabel('Tuning & Training Time [s]', fontweight='bold')
ax_train_time.set_title('Model Training Times')
ax_train_time.set_xticks(x)
ax_train_time.set_xticklabels(model_names, rotation=45, ha='right')

for bar in bars_train_time:
    yval = bar.get_height()
    ax_train_time.text(bar.get_x() + bar.get_width()/2, yval + (max(tuning_times)*0.01), f'{yval:.1f}s', ha='center', va='bottom', fontsize=8)

# Subplot 2: Prediction Time
bars_pred_time = ax_pred_time.bar(x, pred_times, color='#377eb8', alpha=0.8, edgecolor='black')
ax_pred_time.set_xlabel('Models', fontweight='bold')
ax_pred_time.set_ylabel('Prediction Time [ms]', fontweight='bold')
ax_pred_time.set_title('Model Prediction Times')
ax_pred_time.set_xticks(x)
ax_pred_time.set_xticklabels(model_names, rotation=45, ha='right')

for bar in bars_pred_time:
    yval = bar.get_height()
    ax_pred_time.text(bar.get_x() + bar.get_width()/2, yval + (max(pred_times)*0.01), f'{yval:.2f}ms', ha='center', va='bottom', fontsize=8)

fig_time.tight_layout()
dynamic_descriptions["Model_comparison_Times.png"] = "Bar charts comparing the computation times (tuning/training and prediction) of all trained models."
fig_time.savefig(os.path.join(config.RESULTS_DIR, "Model_comparison_Times.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close(fig_time)

print("\n--- RMSE analysis on normal and high COF ranges (Test set) ---")
mask_high = y_test_filtered['COF'] > 0.26
mask_normal = ~mask_high

if mask_high.any():
    rmse_high = np.sqrt(mean_squared_error(y_test_filtered.loc[mask_high, 'COF'], y_test_pred[mask_high, 0]))
    print(f"RMSE (COF > 0.26)  : {rmse_high:.4f}")
else:
    print("No COF > 0.26 in the (filtered) test set.")

if mask_normal.any():
    rmse_normal = np.sqrt(mean_squared_error(y_test_filtered.loc[mask_normal, 'COF'], y_test_pred[mask_normal, 0]))
    print(f"RMSE (COF <= 0.26) : {rmse_normal:.4f}")

print("\n--- Identifying sources of high errors (|Error| > 0.05)... ---")
error_df = X_test_filtered.copy()
error_df['Residual'] = residuals
error_df['File_ID'] = groups_test_filtered.values
error_df['Actual_COF'] = y_test_filtered['COF'].values

high_error_df = error_df[error_df['Residual'].abs() > 0.05]
if not high_error_df.empty:
    print("\nFiles with the most |Error| > 0.05:")
    error_stats = high_error_df.groupby('File_ID').agg(
        Count=('Residual', 'count'),
        Avg_Time=('Time', 'mean'),
        Avg_Load=('Load', 'mean'),
        Avg_Temp=('Temperature', 'mean')
    ).sort_values(by='Count', ascending=False)
    print(error_stats.to_string())
else:
    print("\nNo absolute error greater than 0.05 in the test set.")

fig, ax = plt.subplots(figsize=(8, 5))
unique_files = error_df['File_ID'].unique()
cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(unique_files)))

for i, file_id in enumerate(unique_files):
    subset = error_df[error_df['File_ID'] == file_id]
    ax.scatter(subset['Time'], subset['Residual'], label=file_id, color=colors[i], alpha=0.6, s=15)

ax.axhline(0.05, color='red', linestyle='--', linewidth=1.5, label='Threshold (±0.05)')
ax.axhline(-0.05, color='red', linestyle='--', linewidth=1.5)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Residual Error")
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='x-small', ncol=2 if len(unique_files)>15 else 1)
dynamic_descriptions["Error_Analysis_Output.png"] = "Scatter plot showing residual errors over time, color-coded by File_ID, to identify the source of high errors."
plt.savefig(os.path.join(config.RESULTS_DIR, "Error_Analysis_Output.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

print("\n--- Generating Partial Dependence Plots (PDP)... ---")
pdp_features = [f for f in ['Load', 'Temperature', 'Concentration'] if f in X_test_filtered.columns]
if 'Load' in X_test_filtered.columns and 'Temperature' in X_test_filtered.columns:
    pdp_features.append(('Load', 'Temperature'))

if pdp_features:
    try:
        class SingleOutputWrapper(RegressorMixin, BaseEstimator):
            def __init__(self, model):
                self.model = model
            def fit(self, X, y=None):
                return self
            def predict(self, X):
                return self.model.predict(X)[:, 0]
            def __sklearn_is_fitted__(self):
                return True
        
        wrapped_model = SingleOutputWrapper(best_model_overall)
        
        display = PartialDependenceDisplay.from_estimator(
            wrapped_model, 
            X_test_filtered, 
            features=pdp_features, 
            feature_names=X_test_filtered.columns,
            grid_resolution=20
        )
        fig = display.figure_
        fig.set_size_inches(12, 8)
        fig.suptitle('Partial Dependence of COF on Key Features (1D and 2D)', y=1.02)
        plt.tight_layout()
        dynamic_descriptions["PDP_features.png"] = "Partial Dependence Plots showing the isolated effect of key features on the predicted COF."
        plt.savefig(os.path.join(config.RESULTS_DIR, "PDP_features.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
        plt.close()
    except Exception as e:
        print(f"Failed to generate PDP plots: {e}")

print("\n--- Generating Evaluation Plots ---")
plt.figure(figsize=(6.3, 3.15))
plt.plot(optimum_results[0]['CurveTime'], optimum_results[0]['CurveCOF'], color='purple', label="Not esterified")
plt.plot(optimum_results[1]['CurveTime'], optimum_results[1]['CurveCOF'], color='orange', label="Esterified")
plt.ylim(config.PLOT_SETTINGS['cof_ylim'])
if optimum_results[0]['RunIn'] > 0:
    plt.axvline(x=optimum_results[0]['RunIn'], color='purple', linestyle='--', alpha=0.5, label='Run-in (Not esterified)')
if optimum_results[1]['RunIn'] > 0:
    plt.axvline(x=optimum_results[1]['RunIn'], color='orange', linestyle='--', alpha=0.5, label='Run-in (Esterified)')
ymin, ymax = plt.gca().get_ylim()
plt.ylim(ymin, ymax + (ymax - ymin) * 0.35)
plt.legend(loc='upper right')
plt.savefig(os.path.join(config.RESULTS_DIR, "Optimum_comparison.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

top3_trend_res = sorted(results, key=lambda x: x['R2_CV'], reverse=True)[:3]
trend_opt_load = optimum_results[config.PLOT_ESTERIFIED_STATE]['Load']
trend_opt_temp = optimum_results[config.PLOT_ESTERIFIED_STATE]['Temp']
trend_opt_conc = optimum_results[config.PLOT_ESTERIFIED_STATE]['Conc']

print("\n--- Generating Trend Analysis Plot (Temperature vs Target) ---")
trend_temp_range = np.linspace(40, 120, 20)
for t_idx, t_name in [(0, 'COF'), (1, 'FAI')]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, res in enumerate(top3_trend_res):
        model = res['Model']
        trend_df_base = pd.DataFrame({'Time': 7050, 'Load': 100, 'Temperature': trend_temp_range, 'Concentration': 0.5, 'Esterified': 0})
        trend_df_ester = pd.DataFrame({'Time': 7050, 'Load': 100, 'Temperature': trend_temp_range, 'Concentration': 0.5, 'Esterified': config.PLOT_ESTERIFIED_STATE})
        trend_df_base = create_features(trend_df_base)[X_cols_raw]
        trend_df_ester = create_features(trend_df_ester)[X_cols_raw]
        trend_base_trans = global_vif.transform(global_interact.transform(trend_df_base))
        trend_ester_trans = global_vif.transform(global_interact.transform(trend_df_ester))
        preds_base = np.maximum(model.predict(trend_base_trans), config.PREDICTION_LOWER_BOUND)[:, t_idx]
        preds_ester = np.maximum(model.predict(trend_ester_trans), config.PREDICTION_LOWER_BOUND)[:, t_idx]
        
        axes[i].plot(trend_temp_range, preds_base, marker='o', color='purple', label='Not esterified (0)')
        axes[i].plot(trend_temp_range, preds_ester, marker='s', color='orange', label='Esterified Oil (1)')
        axes[i].fill_between(trend_temp_range, preds_base, preds_ester, color='grey', alpha=0.2, label='Ester Advantage')
        axes[i].set_title(res['Name'], fontsize=11)
        axes[i].set_xlabel("Temperature [°C]")
        if i == 0: axes[i].set_ylabel(f"Expected {t_name} [-]")
        axes[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        if i == 2: axes[i].legend(loc='upper right', fontsize=9)

    plt.suptitle(f"Temperature Trend Analysis - {t_name} (Load: 100N, Conc: 0.50wt%)", y=1.05)
    fname = f"Temperature_Trend_Analysis_{t_name}.png"
    dynamic_descriptions[fname] = f"Temperature Trend Analysis (Load: 100N, Conc: 0.50wt%) for {t_name} across the Top 3 models."
    plt.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("\n--- Generating Trend Analysis Plot (Temperature vs Target) [Optimum Bonus] ---")
for t_idx, t_name in [(0, 'COF'), (1, 'FAI')]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, res in enumerate(top3_trend_res):
        model = res['Model']
        trend_df_base_sm = pd.DataFrame({'Time': 7050, 'Load': trend_opt_load, 'Temperature': trend_temp_range, 'Concentration': trend_opt_conc, 'Esterified': 0})
        trend_df_ester_sm = pd.DataFrame({'Time': 7050, 'Load': trend_opt_load, 'Temperature': trend_temp_range, 'Concentration': trend_opt_conc, 'Esterified': config.PLOT_ESTERIFIED_STATE})
        trend_df_base_sm = create_features(trend_df_base_sm)[X_cols_raw]
        trend_df_ester_sm = create_features(trend_df_ester_sm)[X_cols_raw]
        trend_base_trans_sm = global_vif.transform(global_interact.transform(trend_df_base_sm))
        trend_ester_trans_sm = global_vif.transform(global_interact.transform(trend_df_ester_sm))
        preds_base_sm = np.maximum(model.predict(trend_base_trans_sm), config.PREDICTION_LOWER_BOUND)[:, t_idx]
        preds_ester_sm = np.maximum(model.predict(trend_ester_trans_sm), config.PREDICTION_LOWER_BOUND)[:, t_idx]
        
        axes[i].plot(trend_temp_range, preds_base_sm, marker='o', color='purple', label='Not esterified (0)')
        axes[i].plot(trend_temp_range, preds_ester_sm, marker='s', color='orange', label='Esterified Oil (1)')
        axes[i].fill_between(trend_temp_range, preds_base_sm, preds_ester_sm, color='grey', alpha=0.2, label='Ester Advantage')
        axes[i].set_title(res['Name'], fontsize=11)
        axes[i].set_xlabel("Temperature [°C]")
        if i == 0: axes[i].set_ylabel(f"Expected {t_name} [-]")
        axes[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        if i == 2: axes[i].legend(loc='upper right', fontsize=9)

    plt.suptitle(f"Temperature Trend Analysis - {t_name} at Optimum (Load: {int(trend_opt_load)}N, Conc: {trend_opt_conc:.2f}wt%)", y=1.05)
    fname = f"Temperature_Trend_Analysis_{t_name}_Optimum.png"
    dynamic_descriptions[fname] = f"Temperature Trend Analysis at Optimum (Load: {int(trend_opt_load)}N, Conc: {trend_opt_conc:.2f}wt%) for {t_name} across the Top 3 models."
    plt.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("\n--- Generating Trend Analysis Plot (Load vs Target & Hertz vs Target) ---")
trend_load_range = np.linspace(10, 200, 20)
E_star_trend = config.E_MODULUS / (2.0 * (1.0 - config.POISSON_RATIO**2))
a_trend = np.cbrt((3.0 * trend_load_range * config.BALL_RADIUS) / (4.0 * E_star_trend))
trend_hertz_range = (3.0 * trend_load_range) / (2.0 * np.pi * a_trend**2)

for t_idx, t_name in [(0, 'COF'), (1, 'FAI')]:
    fig_load, axes_load = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig_hertz, axes_hertz = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for i, res in enumerate(top3_trend_res):
        model = res['Model']
        trend_df_base_load = pd.DataFrame({'Time': 7050, 'Load': trend_load_range, 'Temperature': 100, 'Concentration': 0.5, 'Esterified': 0})
        trend_df_ester_load = pd.DataFrame({'Time': 7050, 'Load': trend_load_range, 'Temperature': 100, 'Concentration': 0.5, 'Esterified': config.PLOT_ESTERIFIED_STATE})
        trend_df_base_load = create_features(trend_df_base_load)[X_cols_raw]
        trend_df_ester_load = create_features(trend_df_ester_load)[X_cols_raw]
        trend_base_trans_load = global_vif.transform(global_interact.transform(trend_df_base_load))
        trend_ester_trans_load = global_vif.transform(global_interact.transform(trend_df_ester_load))
        preds_base_load = np.maximum(model.predict(trend_base_trans_load), config.PREDICTION_LOWER_BOUND)[:, t_idx]
        preds_ester_load = np.maximum(model.predict(trend_ester_trans_load), config.PREDICTION_LOWER_BOUND)[:, t_idx]

        axes_load[i].plot(trend_load_range, preds_base_load, marker='o', color='purple', label='Not esterified (0)')
        axes_load[i].plot(trend_load_range, preds_ester_load, marker='s', color='orange', label='Esterified Oil (1)')
        axes_load[i].fill_between(trend_load_range, preds_base_load, preds_ester_load, color='grey', alpha=0.2, label='Ester Advantage')
        axes_load[i].set_title(res['Name'], fontsize=11)
        axes_load[i].set_xlabel("Load [N]")
        if i == 0: axes_load[i].set_ylabel(f"Expected {t_name} [-]")
        axes_load[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        if i == 2: axes_load[i].legend(loc='upper right', fontsize=9)

        axes_hertz[i].plot(trend_hertz_range, preds_base_load, marker='o', color='purple', label='Not esterified (0)')
        axes_hertz[i].plot(trend_hertz_range, preds_ester_load, marker='s', color='orange', label='Esterified Oil (1)')
        axes_hertz[i].fill_between(trend_hertz_range, preds_base_load, preds_ester_load, color='grey', alpha=0.2, label='Ester Advantage')
        axes_hertz[i].set_title(res['Name'], fontsize=11)
        axes_hertz[i].set_xlabel("Max. Hertzian Stress [MPa]")
        if i == 0: axes_hertz[i].set_ylabel(f"Expected {t_name} [-]")
        axes_hertz[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        if i == 2: axes_hertz[i].legend(loc='upper right', fontsize=9)

    fig_load.suptitle(f"Load Trend Analysis - {t_name} (Temp: 100°C, Conc: 0.50wt%)", y=1.05)
    fname_load = f"Load_Trend_Analysis_{t_name}.png"
    dynamic_descriptions[fname_load] = f"Load Trend Analysis (Temp: 100°C, Conc: 0.50wt%) for {t_name} across the Top 3 models."
    fig_load.savefig(os.path.join(config.RESULTS_DIR, fname_load), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_load)

    fig_hertz.suptitle(f"Hertzian Contact Stress Trend Analysis - {t_name} (Temp: 100°C, Conc: 0.50wt%)", y=1.05)
    fname_hertz = f"Hertz_Trend_Analysis_{t_name}.png"
    dynamic_descriptions[fname_hertz] = f"Hertzian Contact Stress Trend Analysis for {t_name} across the Top 3 models."
    fig_hertz.savefig(os.path.join(config.RESULTS_DIR, fname_hertz), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_hertz)

print("\n--- Generating Trend Analysis Plot (Load vs Target) [Optimum Bonus] ---")
for t_idx, t_name in [(0, 'COF'), (1, 'FAI')]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, res in enumerate(top3_trend_res):
        model = res['Model']
        trend_df_base_load_sm = pd.DataFrame({'Time': 7050, 'Load': trend_load_range, 'Temperature': trend_opt_temp, 'Concentration': trend_opt_conc, 'Esterified': 0})
        trend_df_ester_load_sm = pd.DataFrame({'Time': 7050, 'Load': trend_load_range, 'Temperature': trend_opt_temp, 'Concentration': trend_opt_conc, 'Esterified': config.PLOT_ESTERIFIED_STATE})
        trend_df_base_load_sm = create_features(trend_df_base_load_sm)[X_cols_raw]
        trend_df_ester_load_sm = create_features(trend_df_ester_load_sm)[X_cols_raw]
        trend_base_trans_load_sm = global_vif.transform(global_interact.transform(trend_df_base_load_sm))
        trend_ester_trans_load_sm = global_vif.transform(global_interact.transform(trend_df_ester_load_sm))
        preds_base_load_sm = np.maximum(model.predict(trend_base_trans_load_sm), config.PREDICTION_LOWER_BOUND)[:, t_idx]
        preds_ester_load_sm = np.maximum(model.predict(trend_ester_trans_load_sm), config.PREDICTION_LOWER_BOUND)[:, t_idx]

        axes[i].plot(trend_load_range, preds_base_load_sm, marker='o', color='purple', label='Not esterified (0)')
        axes[i].plot(trend_load_range, preds_ester_load_sm, marker='s', color='orange', label='Esterified Oil (1)')
        axes[i].fill_between(trend_load_range, preds_base_load_sm, preds_ester_load_sm, color='grey', alpha=0.2, label='Ester Advantage')
        axes[i].set_title(res['Name'], fontsize=11)
        axes[i].set_xlabel("Load [N]")
        if i == 0: axes[i].set_ylabel(f"Expected {t_name} [-]")
        axes[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        if i == 2: axes[i].legend(loc='upper right', fontsize=9)

    plt.suptitle(f"Load Trend Analysis - {t_name} at Optimum (Temp: {int(trend_opt_temp)}°C, Conc: {trend_opt_conc:.2f}wt%)", y=1.05)
    fname = f"Load_Trend_Analysis_{t_name}_Optimum.png"
    dynamic_descriptions[fname] = f"Load Trend Analysis at Optimum (Temp: {int(trend_opt_temp)}°C, Conc: {trend_opt_conc:.2f}wt%) for {t_name} across the Top 3 models."
    plt.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("\n--- Generating Trend Analysis Plot (Concentration vs COF & FAI) ---")
trend_conc_range = np.linspace(0, 0.6, 20)
for t_idx, t_name in [(0, 'COF'), (1, 'FAI')]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for i, res in enumerate(top3_trend_res):
        model = res['Model']
        trend_df_base_conc = pd.DataFrame({'Time': 7050, 'Load': 100, 'Temperature': 100, 'Concentration': trend_conc_range, 'Esterified': 0})
        trend_df_ester_conc = pd.DataFrame({'Time': 7050, 'Load': 100, 'Temperature': 100, 'Concentration': trend_conc_range, 'Esterified': config.PLOT_ESTERIFIED_STATE})
        trend_df_base_conc = create_features(trend_df_base_conc)[X_cols_raw]
        trend_df_ester_conc = create_features(trend_df_ester_conc)[X_cols_raw]
        trend_base_trans_conc = global_vif.transform(global_interact.transform(trend_df_base_conc))
        trend_ester_trans_conc = global_vif.transform(global_interact.transform(trend_df_ester_conc))
        preds_base_conc = np.maximum(model.predict(trend_base_trans_conc), config.PREDICTION_LOWER_BOUND)[:, t_idx]
        preds_ester_conc = np.maximum(model.predict(trend_ester_trans_conc), config.PREDICTION_LOWER_BOUND)[:, t_idx]

        axes[i].plot(trend_conc_range, preds_base_conc, marker='o', color='purple', label='Not esterified (0)')
        axes[i].plot(trend_conc_range, preds_ester_conc, marker='s', color='orange', label='Esterified Oil (1)')
        axes[i].fill_between(trend_conc_range, preds_base_conc, preds_ester_conc, color='grey', alpha=0.2, label='Ester Advantage')
        axes[i].set_title(res['Name'], fontsize=11)
        axes[i].set_xlabel("Concentration [wt%]")
        if i == 0: axes[i].set_ylabel(f"Expected {t_name} [-]")
        axes[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        axes[i].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        if i == 2: axes[i].legend(loc='upper right', fontsize=9)

    plt.suptitle(f"Concentration Trend Analysis - {t_name} (Load: 100N, Temp: 100°C)", y=1.05)
    fname = f"Concentration_Trend_Analysis_{t_name}.png"
    dynamic_descriptions[fname] = f"Concentration Trend Analysis (Load: 100N, Temp: 100°C) for {t_name} across the Top 3 models."
    plt.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("\n--- Generating Trend Analysis Plot (Concentration vs COF) [Optimum Bonus] ---")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for i, res in enumerate(top3_trend_res):
    model = res['Model']
    trend_df_base_conc_sm = pd.DataFrame({'Time': 7050, 'Load': trend_opt_load, 'Temperature': trend_opt_temp, 'Concentration': trend_conc_range, 'Esterified': 0})
    trend_df_ester_conc_sm = pd.DataFrame({'Time': 7050, 'Load': trend_opt_load, 'Temperature': trend_opt_temp, 'Concentration': trend_conc_range, 'Esterified': config.PLOT_ESTERIFIED_STATE})
    trend_df_base_conc_sm = create_features(trend_df_base_conc_sm)[X_cols_raw]
    trend_df_ester_conc_sm = create_features(trend_df_ester_conc_sm)[X_cols_raw]
    trend_base_trans_conc_sm = global_vif.transform(global_interact.transform(trend_df_base_conc_sm))
    trend_ester_trans_conc_sm = global_vif.transform(global_interact.transform(trend_df_ester_conc_sm))
    preds_base_conc_sm = np.maximum(model.predict(trend_base_trans_conc_sm), config.PREDICTION_LOWER_BOUND)[:, 0]
    preds_ester_conc_sm = np.maximum(model.predict(trend_ester_trans_conc_sm), config.PREDICTION_LOWER_BOUND)[:, 0]

    axes[i].plot(trend_conc_range, preds_base_conc_sm, marker='o', color='purple', label='Not esterified (0)')
    axes[i].plot(trend_conc_range, preds_ester_conc_sm, marker='s', color='orange', label='Esterified Oil (1)')
    axes[i].fill_between(trend_conc_range, preds_base_conc_sm, preds_ester_conc_sm, color='grey', alpha=0.2, label='Ester Advantage')
    axes[i].set_title(res['Name'], fontsize=11)
    axes[i].set_xlabel("Concentration [wt%]")
    if i == 0: axes[i].set_ylabel("Expected COF [-]")
    axes[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    axes[i].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    if i == 2: axes[i].legend(loc='upper right', fontsize=9)

plt.suptitle(f"Concentration Trend Analysis at Optimum (Load: {int(trend_opt_load)}N, Temp: {int(trend_opt_temp)}°C)", y=1.05)
dynamic_descriptions["Concentration_Trend_Analysis_Optimum.png"] = f"Concentration Trend Analysis at Optimum (Load: {int(trend_opt_load)}N, Temp: {int(trend_opt_temp)}°C) across the Top 3 models."
plt.savefig(os.path.join(config.RESULTS_DIR, "Concentration_Trend_Analysis_Optimum.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

print("\n--- Generating CatBoost Specific Load Trend (COF & FAI) at 0.55wt% & 100°C ---")
cb_res = next((r for r in results if r['Name'] == "CatBoost"), None)
if cb_res:
    cb_model = cb_res['Model']
    trend_load_range_cb = np.linspace(10, 200, 50)
    df_cb = pd.DataFrame({'Time': 7050, 'Load': trend_load_range_cb, 'Temperature': 100, 'Concentration': 0.55, 'Esterified': config.PLOT_ESTERIFIED_STATE})
    df_cb = create_features(df_cb)[X_cols_raw]
    df_cb_trans = global_vif.transform(global_interact.transform(df_cb))
    preds_cb = np.maximum(cb_model.predict(df_cb_trans), config.PREDICTION_LOWER_BOUND)
    
    fig, ax1 = plt.subplots(figsize=(6.3, 4.0))
    ax2 = ax1.twinx()
    
    ax1.plot(trend_load_range_cb, preds_cb[:, 0], color='purple', linewidth=2.5, label='Expected COF')
    ax2.plot(trend_load_range_cb, preds_cb[:, 1], color='orange', linewidth=2.5, linestyle='--', label='Expected FAI')
    
    ax1.set_xlabel("Load [N]")
    ax1.set_ylabel("Expected COF [-]", color='purple')
    ax2.set_ylabel("Expected FAI [-]", color='orange')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    
    ester_text = "Esterified" if config.PLOT_ESTERIFIED_STATE == 1 else "Not esterified"
    plt.title(f"CatBoost Load Trend (Temp: 100°C, Conc: 0.55wt%, {ester_text})", fontsize=11)
    fname_cb = "CatBoost_Load_Trend_COF_FAI_055wt.png"
    dynamic_descriptions[fname_cb] = f"CatBoost Load trend analysis for both COF and FAI at 100°C and 0.55 wt% concentration ({ester_text})."
    plt.savefig(os.path.join(config.RESULTS_DIR, fname_cb), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

print("\n--- Generating Combined Model Trend Analysis (Esterified Only) ---")
def get_style(name):
    is_ens = "Ensemble" in name
    c = '#911eb4' if is_ens else {'XGBoost': '#e6194B', 'LightGBM': '#3cb44b', 'CatBoost': '#4363d8', 'Neural Network (MLP)': '#f58231'}.get(name, 'black')
    m = 'D' if is_ens else {'XGBoost': 's', 'LightGBM': '^', 'CatBoost': 'v', 'Neural Network (MLP)': 'o'}.get(name, 'x')
    ls = '--' if is_ens else '-'
    lw = 2.5 if is_ens else 2.0
    return c, m, ls, lw

temp_compare_results = [r for r in results if r['Name'] in ["CatBoost", "Ensemble (Top 3 Voting)"]]
load_compare_results = [r for r in results if r['Name'] in ["Neural Network (MLP)", "CatBoost", "Ensemble (Top 3 Voting)"]]
conc_compare_results = [r for r in results if r['Name'] in ["XGBoost", "LightGBM", "CatBoost", "Ensemble (Top 3 Voting)"]]

# 1. Temperature Trend
fig, ax = plt.subplots(figsize=(6.3, 4.0))
for res in temp_compare_results:
    model = res['Model']
    c, m, ls, lw = get_style(res['Name'])
    
    trend_df = pd.DataFrame({'Time': 7050, 'Load': 100, 'Temperature': trend_temp_range, 'Concentration': 0.5, 'Esterified': 1})
    trend_df = create_features(trend_df)[X_cols_raw]
    trend_trans = global_vif.transform(global_interact.transform(trend_df))
    preds = np.maximum(model.predict(trend_trans), config.PREDICTION_LOWER_BOUND)[:, 0]
    ax.plot(trend_temp_range, preds, marker=m, color=c, linestyle=ls, linewidth=lw, label=res['Name'])
ax.set_title("Temperature Trend (Esterified, Load: 100N, Conc: 0.5wt%)", fontsize=11)
ax.set_xlabel("Temperature [°C]")
ax.set_ylabel("Expected COF [-]")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.legend(loc='upper right', fontsize=9)
dynamic_descriptions["Combined_Model_Temperature_Trend.png"] = "Comparison of Esterified Oil Temperature Trends using CatBoost and Ensemble."
plt.savefig(os.path.join(config.RESULTS_DIR, "Combined_Model_Temperature_Trend.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

# 2. Load Trend
fig, ax = plt.subplots(figsize=(6.3, 4.0))
for res in load_compare_results:
    model = res['Model']
    c, m, ls, lw = get_style(res['Name'])
    
    trend_df = pd.DataFrame({'Time': 7050, 'Load': trend_load_range, 'Temperature': 100, 'Concentration': 0.5, 'Esterified': 1})
    trend_df = create_features(trend_df)[X_cols_raw]
    trend_trans = global_vif.transform(global_interact.transform(trend_df))
    preds = np.maximum(model.predict(trend_trans), config.PREDICTION_LOWER_BOUND)[:, 0]
    ax.plot(trend_load_range, preds, marker=m, color=c, linestyle=ls, linewidth=lw, label=res['Name'])
ax.set_title("Load Trend (Esterified, Temp: 100°C, Conc: 0.5wt%)", fontsize=11)
ax.set_xlabel("Load [N]")
ax.set_ylabel("Expected COF [-]")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.legend(loc='upper right', fontsize=9)
dynamic_descriptions["Combined_Model_Load_Trend.png"] = "Comparison of Esterified Oil Load Trends using Neural Network, CatBoost, and Ensemble."
plt.savefig(os.path.join(config.RESULTS_DIR, "Combined_Model_Load_Trend.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

# 2.1 Load Trend (CatBoost vs Neural Network)
fig, ax = plt.subplots(figsize=(6.3, 4.0))
subset_compare_cb_mlp = [r for r in results if r['Name'] in ["CatBoost", "Neural Network (MLP)"]]
for res in subset_compare_cb_mlp:
    model = res['Model']
    c = '#f58231' if "Neural" in res['Name'] else '#4363d8'
    m = 'o' if "Neural" in res['Name'] else 's'
    ls = '-'
    lw = 2.0
    
    trend_df = pd.DataFrame({'Time': 7050, 'Load': trend_load_range, 'Temperature': 100, 'Concentration': 0.5, 'Esterified': 1})
    trend_df = create_features(trend_df)[X_cols_raw]
    trend_trans = global_vif.transform(global_interact.transform(trend_df))
    preds = np.maximum(model.predict(trend_trans), config.PREDICTION_LOWER_BOUND)[:, 0]
    ax.plot(trend_load_range, preds, marker=m, color=c, linestyle=ls, linewidth=lw, label=res['Name'])
ax.set_title("Load Trend (CatBoost vs Neural Network) (Esterified)", fontsize=11)
ax.set_xlabel("Load [N]")
ax.set_ylabel("Expected COF [-]")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.legend(loc='upper right', fontsize=9)
dynamic_descriptions["Load_Trend_CatBoost_vs_MLP.png"] = "Comparison of Esterified Oil Load Trends using CatBoost and Neural Network."
plt.savefig(os.path.join(config.RESULTS_DIR, "Load_Trend_CatBoost_vs_MLP.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

# 3. Concentration Trend
fig, ax = plt.subplots(figsize=(6.3, 4.0))
for res in conc_compare_results:
    model = res['Model']
    c, m, ls, lw = get_style(res['Name'])
    
    trend_df = pd.DataFrame({'Time': 7050, 'Load': 100, 'Temperature': 100, 'Concentration': trend_conc_range, 'Esterified': 1})
    trend_df = create_features(trend_df)[X_cols_raw]
    trend_trans = global_vif.transform(global_interact.transform(trend_df))
    preds = np.maximum(model.predict(trend_trans), config.PREDICTION_LOWER_BOUND)[:, 0]
    ax.plot(trend_conc_range, preds, marker=m, color=c, linestyle=ls, linewidth=lw, label=res['Name'])
ax.set_title("Concentration Trend (Esterified, Load: 100N, Temp: 100°C)", fontsize=11)
ax.set_xlabel("Concentration [wt%]")
ax.set_ylabel("Expected COF [-]")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
ax.legend(loc='upper right', fontsize=9)
dynamic_descriptions["Combined_Model_Concentration_Trend.png"] = "Comparison of Esterified Oil Concentration Trends using XGBoost, LightGBM, CatBoost, and Ensemble."
plt.savefig(os.path.join(config.RESULTS_DIR, "Combined_Model_Concentration_Trend.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

print("\n--- Generating Combined Model Trend Analysis (Esterified Only) [Optimum Bonus] ---")

# 1. Temperature Trend Optimum
fig, ax = plt.subplots(figsize=(6.3, 4.0))
for res in temp_compare_results:
    model = res['Model']
    c, m, ls, lw = get_style(res['Name'])
    
    trend_df = pd.DataFrame({'Time': 7050, 'Load': trend_opt_load, 'Temperature': trend_temp_range, 'Concentration': trend_opt_conc, 'Esterified': 1})
    trend_df = create_features(trend_df)[X_cols_raw]
    trend_trans = global_vif.transform(global_interact.transform(trend_df))
    preds = np.maximum(model.predict(trend_trans), config.PREDICTION_LOWER_BOUND)[:, 0]
    ax.plot(trend_temp_range, preds, marker=m, color=c, linestyle=ls, linewidth=lw, label=res['Name'])
ax.set_title(f"Temperature Trend at Optimum (Esterified, Load: {int(trend_opt_load)}N, Conc: {trend_opt_conc:.2f}wt%)", fontsize=11)
ax.set_xlabel("Temperature [°C]")
ax.set_ylabel("Expected COF [-]")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.legend(loc='upper right', fontsize=9)
dynamic_descriptions["Combined_Model_Temperature_Trend_Optimum.png"] = f"Comparison of Esterified Oil Temperature Trends at Optimum (Load: {int(trend_opt_load)}N, Conc: {trend_opt_conc:.2f}wt%) using CatBoost and Ensemble."
plt.savefig(os.path.join(config.RESULTS_DIR, "Combined_Model_Temperature_Trend_Optimum.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

# 2. Load Trend Optimum
fig, ax = plt.subplots(figsize=(6.3, 4.0))
for res in load_compare_results:
    model = res['Model']
    c, m, ls, lw = get_style(res['Name'])
    
    trend_df = pd.DataFrame({'Time': 7050, 'Load': trend_load_range, 'Temperature': trend_opt_temp, 'Concentration': trend_opt_conc, 'Esterified': 1})
    trend_df = create_features(trend_df)[X_cols_raw]
    trend_trans = global_vif.transform(global_interact.transform(trend_df))
    preds = np.maximum(model.predict(trend_trans), config.PREDICTION_LOWER_BOUND)[:, 0]
    ax.plot(trend_load_range, preds, marker=m, color=c, linestyle=ls, linewidth=lw, label=res['Name'])
ax.set_title(f"Load Trend at Optimum (Esterified, Temp: {int(trend_opt_temp)}°C, Conc: {trend_opt_conc:.2f}wt%)", fontsize=11)
ax.set_xlabel("Load [N]")
ax.set_ylabel("Expected COF [-]")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.legend(loc='upper right', fontsize=9)
dynamic_descriptions["Combined_Model_Load_Trend_Optimum.png"] = f"Comparison of Esterified Oil Load Trends at Optimum (Temp: {int(trend_opt_temp)}°C, Conc: {trend_opt_conc:.2f}wt%) using Neural Network, CatBoost, and Ensemble."
plt.savefig(os.path.join(config.RESULTS_DIR, "Combined_Model_Load_Trend_Optimum.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

# 3. Concentration Trend Optimum
fig, ax = plt.subplots(figsize=(6.3, 4.0))
for res in conc_compare_results:
    model = res['Model']
    c, m, ls, lw = get_style(res['Name'])
    
    trend_df = pd.DataFrame({'Time': 7050, 'Load': trend_opt_load, 'Temperature': trend_opt_temp, 'Concentration': trend_conc_range, 'Esterified': 1})
    trend_df = create_features(trend_df)[X_cols_raw]
    trend_trans = global_vif.transform(global_interact.transform(trend_df))
    preds = np.maximum(model.predict(trend_trans), config.PREDICTION_LOWER_BOUND)[:, 0]
    ax.plot(trend_conc_range, preds, marker=m, color=c, linestyle=ls, linewidth=lw, label=res['Name'])
ax.set_title(f"Concentration Trend at Opt. (Esterified, Load: {int(trend_opt_load)}N, Temp: {int(trend_opt_temp)}°C)", fontsize=11)
ax.set_xlabel("Concentration [wt%]")
ax.set_ylabel("Expected COF [-]")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
ax.legend(loc='upper right', fontsize=9)
dynamic_descriptions["Combined_Model_Concentration_Trend_Optimum.png"] = f"Comparison of Esterified Oil Concentration Trends at Optimum (Load: {int(trend_opt_load)}N, Temp: {int(trend_opt_temp)}°C) using XGBoost, LightGBM, CatBoost, and Ensemble."
plt.savefig(os.path.join(config.RESULTS_DIR, "Combined_Model_Concentration_Trend_Optimum.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

print("\n--- Run-In Time Analysis using CatBoost... ---")
from catboost import CatBoostRegressor

file_run_in = []
for fid in full_df['File_ID'].unique():
    file_data = full_df[full_df['File_ID'] == fid].sort_values('Time')
    if file_data.empty: continue
    
    curve_time = file_data['Time'].values
    curve_cof = file_data['COF'].values
    
    smoothed = pd.Series(curve_cof).rolling(60, min_periods=1).mean().values
    valid_idx = ~np.isnan(smoothed)
    if not valid_idx.any():
        run_in = 0
    else:
        smoothed_valid = smoothed[valid_idx]
        valid_time = curve_time[valid_idx]
        tail_len = max(50, int(len(smoothed_valid) * 0.1))
        tail_data = smoothed_valid[-tail_len:]
        final_mean = np.mean(tail_data)
        final_std = np.std(tail_data)
        tol = max(3 * final_std, 0.05 * final_mean)
        outside = np.where(np.abs(smoothed_valid - final_mean) > tol)[0]
        run_in = valid_time[outside[-1]] if len(outside) > 0 else 0
        
    file_run_in.append({
        'File_ID': fid,
        'Load': file_data['Load'].iloc[0],
        'Temperature': file_data['Temperature'].iloc[0],
        'Concentration': file_data['Concentration'].iloc[0],
        'Esterified': file_data['Esterified'].iloc[0],
        'RunIn_Time': run_in
    })

df_runin = pd.DataFrame(file_run_in)

cb_runin = CatBoostRegressor(iterations=300, learning_rate=0.05, depth=4, verbose=0, random_state=config.RANDOM_SEED, allow_writing_files=False)

X_runin = df_runin[['Load', 'Temperature', 'Concentration', 'Esterified']]
y_runin = df_runin['RunIn_Time']

cb_runin.fit(X_runin, y_runin)

imp_runin = cb_runin.get_feature_importance()
sorted_idx = np.argsort(imp_runin)

plt.figure(figsize=(6.3, 3.15))
plt.barh(X_runin.columns[sorted_idx], imp_runin[sorted_idx], color='#4363d8')
plt.xlabel("Feature Importance (Run-In Time)")
plt.title("Factors Influencing Run-In Time (CatBoost)")
dynamic_descriptions["RunIn_Feature_Importance.png"] = "Feature importance showing which operating conditions influence the Run-In time the most, using a dedicated CatBoost model trained on actual run-in times."
plt.savefig(os.path.join(config.RESULTS_DIR, "RunIn_Feature_Importance.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df_trend_temp_0 = pd.DataFrame({'Load': 100, 'Temperature': trend_temp_range, 'Concentration': 0.5, 'Esterified': 0})
df_trend_temp_1 = pd.DataFrame({'Load': 100, 'Temperature': trend_temp_range, 'Concentration': 0.5, 'Esterified': 1})
axes[0].plot(trend_temp_range, cb_runin.predict(df_trend_temp_0), marker='o', color='purple', label='Not esterified (0)')
axes[0].plot(trend_temp_range, cb_runin.predict(df_trend_temp_1), marker='s', color='orange', label='Esterified (1)')
axes[0].set_xlabel("Temperature [°C]")
axes[0].set_ylabel("Predicted Run-In Time [s]")
axes[0].set_title("Run-In vs Temperature")

df_trend_load_0 = pd.DataFrame({'Load': trend_load_range, 'Temperature': 100, 'Concentration': 0.5, 'Esterified': 0})
df_trend_load_1 = pd.DataFrame({'Load': trend_load_range, 'Temperature': 100, 'Concentration': 0.5, 'Esterified': 1})
axes[1].plot(trend_load_range, cb_runin.predict(df_trend_load_0), marker='o', color='purple', label='Not esterified (0)')
axes[1].plot(trend_load_range, cb_runin.predict(df_trend_load_1), marker='s', color='orange', label='Esterified (1)')
axes[1].set_xlabel("Load [N]")
axes[1].set_title("Run-In vs Load")

df_trend_conc_0 = pd.DataFrame({'Load': 100, 'Temperature': 100, 'Concentration': trend_conc_range, 'Esterified': 0})
df_trend_conc_1 = pd.DataFrame({'Load': 100, 'Temperature': 100, 'Concentration': trend_conc_range, 'Esterified': 1})
axes[2].plot(trend_conc_range, cb_runin.predict(df_trend_conc_0), marker='o', color='purple', label='Not esterified (0)')
axes[2].plot(trend_conc_range, cb_runin.predict(df_trend_conc_1), marker='s', color='orange', label='Esterified (1)')
axes[2].set_xlabel("Concentration [wt%]")
axes[2].set_title("Run-In vs Concentration")
axes[2].legend(loc='upper right', fontsize=9)

plt.suptitle("Run-In Time Trends (Base state: Load 100N, Temp 100°C, Conc 0.5wt%)", y=1.05)
dynamic_descriptions["RunIn_Trends.png"] = "Trend analysis showing how Temperature, Load, and Concentration affect the Run-In time, predicted by CatBoost."
plt.savefig(os.path.join(config.RESULTS_DIR, "RunIn_Trends.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

val_load = config.VALIDATION_LOAD
val_temp = config.VALIDATION_TEMP
val_conc = config.VALIDATION_CONC
val_ester = config.PLOT_ESTERIFIED_STATE

print(f"\n--- Generating validation curve and table ({val_load} N, {val_temp} °C, {val_conc} wt%) ---")

val_match = full_df[
    (full_df['Load'].round() == val_load) &
    (full_df['Temperature'].round() == val_temp) &
    (np.isclose(full_df['Concentration'], val_conc, atol=1e-3)) &
    (full_df['Esterified'] == val_ester)
]

actual_time, actual_cof, actual_fai = None, None, None
if not val_match.empty:
    val_file_id = val_match['File_ID'].iloc[0]
    val_file_data = full_df[full_df['File_ID'] == val_file_id].sort_values('Time')
    actual_time = val_file_data['Time'].values
    actual_cof = val_file_data['COF'].values
    actual_fai = val_file_data['Friction absolute integral'].values
    print(f"Actual measurement found for validation: {val_file_id}")
else:
    print("Exact actual measurement not found for the given validation parameters. Showing predictions only.")

val_df_curve = pd.DataFrame({
    'Time': template_df['Time'],
    'Load': val_load,
    'Temperature': val_temp,
    'Concentration': val_conc,
    'Esterified': val_ester
})
val_df_curve = create_features(val_df_curve)[X_cols_raw]
val_trans_curve = global_vif.transform(global_interact.transform(val_df_curve))

val_df_point = pd.DataFrame({
    'Time': [7050],
    'Load': [val_load],
    'Temperature': [val_temp],
    'Concentration': [val_conc],
    'Esterified': [val_ester]
})
val_df_point = create_features(val_df_point)[X_cols_raw]
val_trans_point = global_vif.transform(global_interact.transform(val_df_point))

val_table_data = []
fig_val_cof, ax_val_cof = plt.subplots(figsize=(8, 4.5))
fig_val_fai, ax_val_fai = plt.subplots(figsize=(8, 4.5))

if actual_time is not None:
    ax_val_cof.plot(actual_time, actual_cof, color='black', linewidth=1.5, linestyle='-', label='Actual measurement', zorder=10)
    ax_val_fai.plot(actual_time, actual_fai, color='black', linewidth=1.5, linestyle='-', label='Actual measurement', zorder=10)

val_styles = {
    'Ensemble (Top 3 Voting)': ('#911eb4', '--', 2.5),
    'XGBoost': ('#e6194B', '-', 1.5),
    'LightGBM': ('#3cb44b', '-', 1.5),
    'CatBoost': ('#4363d8', '-', 1.5),
    'Neural Network (MLP)': ('#f58231', '-', 1.5),
    'Random Forest': ('#800000', '-.', 1.5),
    'KNN Regressor': ('#008080', ':', 1.5),
    'Ridge Regression': ('#f032e6', '-.', 1.5)
}

for res in results:
    model_name = res['Name']
    if model_name in ['KNN Regressor', 'Ridge Regression']:
        continue
        
    model = res['Model']
    
    preds_curve = np.maximum(model.predict(val_trans_curve), config.PREDICTION_LOWER_BOUND)[:, 0]
    preds_curve_fai = np.maximum(model.predict(val_trans_curve), config.PREDICTION_LOWER_BOUND)[:, 1]
    
    c, ls, lw = val_styles.get(model_name, ('grey', '-', 1.5))
    alpha = 0.9 if "Ensemble" in model_name else 0.7
    
    ax_val_cof.plot(template_df['Time'], preds_curve, color=c, linestyle=ls, linewidth=lw, alpha=alpha, label=model_name)
    ax_val_fai.plot(template_df['Time'], preds_curve_fai, color=c, linestyle=ls, linewidth=lw, alpha=alpha, label=model_name)
    
    preds_point = np.maximum(model.predict(val_trans_point), config.PREDICTION_LOWER_BOUND)[0]
    val_table_data.append({
        'Model': model_name,
        'Predicted_COF': preds_point[0],
        'Predicted_FAI': preds_point[1]
    })

ester_str = "Esterified" if val_ester == 1 else "Not esterified"

ax_val_cof.set_xlabel('Time [s]')
ax_val_cof.set_ylabel('Coefficient of Friction (COF) [-]')
ax_val_cof.set_title(f'COF Validation Curve ({val_load}N, {val_temp}°C, {val_conc}wt%, {ester_str})', fontsize=11)
ax_val_cof.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
fig_val_cof.tight_layout()
val_filename_cof = f"Validation_Curve_COF_{val_load}N_{val_temp}C_{str(val_conc).replace('.','')}wt.png"
fig_val_cof.savefig(os.path.join(config.RESULTS_DIR, val_filename_cof), dpi=config.PLOT_SETTINGS['dpi'])
plt.close(fig_val_cof)
dynamic_descriptions[val_filename_cof] = f"COF Validation curve predictions vs actual measurement at {val_load}N, {val_temp}°C, {val_conc}wt% ({ester_str})."

ax_val_fai.set_xlabel('Time [s]')
ax_val_fai.set_ylabel('Friction Absolute Integral (FAI) [-]')
ax_val_fai.set_title(f'FAI Validation Curve ({val_load}N, {val_temp}°C, {val_conc}wt%, {ester_str})', fontsize=11)
ax_val_fai.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
fig_val_fai.tight_layout()
val_filename_fai = f"Validation_Curve_FAI_{val_load}N_{val_temp}C_{str(val_conc).replace('.','')}wt.png"
fig_val_fai.savefig(os.path.join(config.RESULTS_DIR, val_filename_fai), dpi=config.PLOT_SETTINGS['dpi'])
plt.close(fig_val_fai)
dynamic_descriptions[val_filename_fai] = f"FAI Validation curve predictions vs actual measurement at {val_load}N, {val_temp}°C, {val_conc}wt% ({ester_str})."

val_table_df = pd.DataFrame(val_table_data)
print(f"\n>> Validation Predictions ({val_load}N, {val_temp}°C, {val_conc}wt%, {ester_str}):")
print(val_table_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n--- Generating 2D Contour Plots... ---")
def plot_contour(var1_name, var1_range, var2_name, var2_range, fixed_vars, filename_suffix, ester_state=None, target_idx=0, target_name='COF'):
    if ester_state is None:
        ester_state = config.PLOT_ESTERIFIED_STATE
        
    V1, V2 = np.meshgrid(var1_range, var2_range)
    grid_flat = pd.DataFrame({
        var1_name: V1.ravel(),
        var2_name: V2.ravel()
    })
    for k, v in fixed_vars.items():
        grid_flat[k] = v
        
    grid_flat['Time'] = 7050
    grid_flat['Esterified'] = ester_state
    
    grid_features = create_features(grid_flat)[X_cols_raw]
    grid_trans = global_vif.transform(global_interact.transform(grid_features))
    preds = np.maximum(best_model_overall.predict(grid_trans), config.PREDICTION_LOWER_BOUND)[:, target_idx]
    
    Z = preds.reshape(V1.shape)
    
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    c = ax.contourf(V1, V2, Z, levels=30, cmap='viridis', alpha=0.9)
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(f'Expected {target_name} [-]')
    
    label1 = "CuO concentration [wt%]" if var1_name == "Concentration" else config.NAME_MAPPING.get(var1_name, f"{var1_name}")
    label2 = "CuO concentration [wt%]" if var2_name == "Concentration" else config.NAME_MAPPING.get(var2_name, f"{var2_name}")
    
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    
    title_str = ", ".join([f"{k}: {v:g}" for k, v in fixed_vars.items()])
    ester_str = "Esterified" if ester_state == 1 else "Base Oil"
    plt.title(f"Contour Plot - {target_name} ({title_str}, {ester_str})", fontsize=10, pad=10)
    
    fname = f"Contour_{target_name}_{filename_suffix}.png"
    plt.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return fname

curr_opt_temp = optimum_results[config.PLOT_ESTERIFIED_STATE]['Temp']
curr_opt_load = optimum_results[config.PLOT_ESTERIFIED_STATE]['Load']
curr_opt_conc = optimum_results[config.PLOT_ESTERIFIED_STATE]['Conc']

c1_range = np.linspace(0, 0.6, 50)
l1_range = np.linspace(10, 200, 50)
fn1 = plot_contour('Concentration', c1_range, 'Load', l1_range, {'Temperature': curr_opt_temp}, 'Conc_Load')
dynamic_descriptions[fn1] = f"2D Contour Plot showing expected COF for Concentration vs Load at optimal Temperature ({int(curr_opt_temp)}°C)."

fn1_fai = plot_contour('Concentration', c1_range, 'Load', l1_range, {'Temperature': curr_opt_temp}, 'Conc_Load', target_idx=1, target_name='FAI')
dynamic_descriptions[fn1_fai] = f"2D Contour Plot showing expected FAI for Concentration vs Load at optimal Temperature ({int(curr_opt_temp)}°C)."

l2_range = np.linspace(10, 200, 50)
t2_range = np.linspace(40, 120, 50)
fn2 = plot_contour('Load', l2_range, 'Temperature', t2_range, {'Concentration': curr_opt_conc}, 'Load_Temp')
dynamic_descriptions[fn2] = f"2D Contour Plot showing expected COF for Load vs Temperature at optimal Concentration ({curr_opt_conc:.2f}wt%)."

fn2_fai = plot_contour('Load', l2_range, 'Temperature', t2_range, {'Concentration': curr_opt_conc}, 'Load_Temp', target_idx=1, target_name='FAI')
dynamic_descriptions[fn2_fai] = f"2D Contour Plot showing expected FAI for Load vs Temperature at optimal Concentration ({curr_opt_conc:.2f}wt%)."

t3_range = np.linspace(40, 120, 50)
c3_range = np.linspace(0, 0.6, 50)
fn3 = plot_contour('Temperature', t3_range, 'Concentration', c3_range, {'Load': curr_opt_load}, 'Temp_Conc')
dynamic_descriptions[fn3] = f"2D Contour Plot showing expected COF for Temperature vs Concentration at optimal Load ({int(curr_opt_load)}N)."

fn3_fai = plot_contour('Temperature', t3_range, 'Concentration', c3_range, {'Load': curr_opt_load}, 'Temp_Conc', target_idx=1, target_name='FAI')
dynamic_descriptions[fn3_fai] = f"2D Contour Plot showing expected FAI for Temperature vs Concentration at optimal Load ({int(curr_opt_load)}N)."

fn4 = plot_contour('Concentration', c1_range, 'Load', l1_range, {'Temperature': 100}, 'Conc_Load_100C')
dynamic_descriptions[fn4] = "Bonus 2D Contour Plot showing expected COF for Concentration vs Load at a fixed Temperature of 100 °C."
fn4_fai = plot_contour('Concentration', c1_range, 'Load', l1_range, {'Temperature': 100}, 'Conc_Load_100C', target_idx=1, target_name='FAI')
dynamic_descriptions[fn4_fai] = "Bonus 2D Contour Plot showing expected FAI for Concentration vs Load at a fixed Temperature of 100 °C."

fn5 = plot_contour('Load', l2_range, 'Temperature', t2_range, {'Concentration': 0.5}, 'Load_Temp_05wt')
dynamic_descriptions[fn5] = "Bonus 2D Contour Plot showing expected COF for Load vs Temperature at a fixed Concentration of 0.5 wt%."
fn5_fai = plot_contour('Load', l2_range, 'Temperature', t2_range, {'Concentration': 0.5}, 'Load_Temp_05wt', target_idx=1, target_name='FAI')
dynamic_descriptions[fn5_fai] = "Bonus 2D Contour Plot showing expected FAI for Load vs Temperature at a fixed Concentration of 0.5 wt%."

fn6 = plot_contour('Temperature', t3_range, 'Concentration', c3_range, {'Load': 100}, 'Temp_Conc_100N')
dynamic_descriptions[fn6] = "Bonus 2D Contour Plot showing expected COF for Temperature vs Concentration at a fixed Load of 100 N."
fn6_fai = plot_contour('Temperature', t3_range, 'Concentration', c3_range, {'Load': 100}, 'Temp_Conc_100N', target_idx=1, target_name='FAI')
dynamic_descriptions[fn6_fai] = "Bonus 2D Contour Plot showing expected FAI for Temperature vs Concentration at a fixed Load of 100 N."

fn7 = plot_contour('Concentration', c1_range, 'Load', l1_range, {'Temperature': 100}, 'Conc_Load_100C_Extra')
dynamic_descriptions[fn7] = "Extra Bonus 2D Contour Plot showing expected COF for Concentration vs Load at a fixed Temperature of 100 °C."

fn8 = plot_contour('Load', l2_range, 'Temperature', t2_range, {'Concentration': 0.1}, 'Load_Temp_01wt')
dynamic_descriptions[fn8] = "Extra Bonus 2D Contour Plot showing expected COF for Load vs Temperature at a fixed Concentration of 0.1 wt%."

fn9 = plot_contour('Temperature', t3_range, 'Concentration', c3_range, {'Load': 50}, 'Temp_Conc_50N')
dynamic_descriptions[fn9] = "Extra Bonus 2D Contour Plot showing expected COF for Temperature vs Concentration at a fixed Load of 50 N."

fn10 = plot_contour('Concentration', c1_range, 'Load', l1_range, {'Temperature': 100}, 'Conc_Load_100C_BaseOil', ester_state=0)
dynamic_descriptions[fn10] = "Base Oil 2D Contour Plot showing expected COF for Concentration vs Load at a fixed Temperature of 100 °C."

fn11 = plot_contour('Load', l2_range, 'Temperature', t2_range, {'Concentration': 0.5}, 'Load_Temp_05wt_BaseOil', ester_state=0)
dynamic_descriptions[fn11] = "Base Oil 2D Contour Plot showing expected COF for Load vs Temperature at a fixed Concentration of 0.5 wt%."

fn12 = plot_contour('Temperature', t3_range, 'Concentration', c3_range, {'Load': 100}, 'Temp_Conc_100N_BaseOil', ester_state=0)
dynamic_descriptions[fn12] = "Base Oil 2D Contour Plot showing expected COF for Temperature vs Concentration at a fixed Load of 100 N."

plot_df = full_df.groupby('File_ID').agg({'Load': 'mean', 'Temperature': 'mean', 'Concentration': 'mean', 'Esterified': 'first'}).reset_index()

np.random.seed(config.RANDOM_SEED)
plot_df['Load_plot'] = plot_df['Load'] + np.random.uniform(-2.0, 2.0, size=len(plot_df))
plot_df['Temperature_plot'] = plot_df['Temperature'] + np.random.uniform(-1.0, 1.0, size=len(plot_df))
plot_df['Concentration_plot'] = np.clip(plot_df['Concentration'] + np.random.uniform(-0.01, 0.01, size=len(plot_df)), 0, None)

base_pts = plot_df[plot_df['Esterified'] == 0]
ester_pts = plot_df[plot_df['Esterified'] == 1]

# --- Plotly Interactive 3D Plot ---
print("\n--- Generating Interactive 3D Plot ---")
fig_3d = go.Figure()
fig_3d.add_trace(go.Scatter3d(
    x=base_pts['Load_plot'], y=base_pts['Temperature_plot'], z=base_pts['Concentration_plot'],
    mode='markers',
    marker=dict(size=4, color='purple', opacity=0.8, line=dict(width=1, color='black')),
    name='Not esterified (0)'
))
fig_3d.add_trace(go.Scatter3d(
    x=ester_pts['Load_plot'], y=ester_pts['Temperature_plot'], z=ester_pts['Concentration_plot'],
    mode='markers',
    marker=dict(size=4, color='orange', opacity=0.8, line=dict(width=1, color='black'), symbol='square'),
    name='Esterified (1)'
))
fig_3d.update_layout(
    scene=dict(xaxis_title='Load [N]', yaxis_title='Temperature [°C]', zaxis_title='Concentration [wt%]'),
    margin=dict(l=0, r=0, b=0, t=30),
    legend=dict(x=0.8, y=0.9)
)
fig_3d.write_html(os.path.join(config.RESULTS_DIR, "3D_distribution.html"))
plotly_3d_html = fig_3d.to_html(full_html=False, include_plotlyjs='cdn')

print("\n--- Generating Interactive 3D Plot with DoE ---")
fig_3d_doe = go.Figure()
fig_3d_doe.add_trace(go.Scatter3d(
    x=base_pts['Load_plot'], y=base_pts['Temperature_plot'], z=base_pts['Concentration_plot'],
    mode='markers',
    marker=dict(size=4, color='purple', opacity=0.8, line=dict(width=1, color='black')),
    name='Not esterified (0)'
))
fig_3d_doe.add_trace(go.Scatter3d(
    x=ester_pts['Load_plot'], y=ester_pts['Temperature_plot'], z=ester_pts['Concentration_plot'],
    mode='markers',
    marker=dict(size=4, color='orange', opacity=0.8, line=dict(width=1, color='black'), symbol='square'),
    name='Esterified (1)'
))

doe_base = doe_suggestions_combined[doe_suggestions_combined['Esterified'] == 0]
if not doe_base.empty:
    fig_3d_doe.add_trace(go.Scatter3d(
        x=doe_base['Load'], y=doe_base['Temperature'], z=doe_base['Concentration'],
        mode='markers',
        marker=dict(size=8, color='cyan', opacity=1.0, line=dict(width=1.5, color='black')),
        name='DoE Suggestions (0)'
    ))
    
doe_ester = doe_suggestions_combined[doe_suggestions_combined['Esterified'] == 1]
if not doe_ester.empty:
    fig_3d_doe.add_trace(go.Scatter3d(
        x=doe_ester['Load'], y=doe_ester['Temperature'], z=doe_ester['Concentration'],
        mode='markers',
        marker=dict(size=8, color='red', opacity=1.0, line=dict(width=1.5, color='black'), symbol='square'),
        name='DoE Suggestions (1)'
    ))

fig_3d_doe.update_layout(
    scene=dict(xaxis_title='Load [N]', yaxis_title='Temperature [°C]', zaxis_title='Concentration [wt%]'),
    margin=dict(l=0, r=0, b=0, t=30),
    legend=dict(x=0.8, y=0.9)
)
fig_3d_doe.write_html(os.path.join(config.RESULTS_DIR, "3D_distribution_with_DoE.html"))
plotly_3d_doe_html = fig_3d_doe.to_html(full_html=False, include_plotlyjs='cdn')

combined_plotly_html = plotly_3d_html + """
</div>
<h3 style="margin-top: 40px;">Interactive 3D Data Distribution with DoE Suggestions</h3>
<p><em>Cyan and Red markers represent the suggested new measurement points.</em></p>
<div class="no-print" style="margin-bottom: 30px; border: 1px solid #ccc; padding: 10px; background-color: #fafafa;">
""" + plotly_3d_doe_html

# Célváltozók és az összes (interakciókkal bővített) bemenet kombinálása
corr_df = pd.concat([Y, X_interact], axis=1)

target_cols = ['COF', 'Friction absolute integral']
input_cols = list(X_interact.columns)
corr_matrix = corr_df.corr().loc[target_cols, input_cols]

base_cols = ['Time', 'Log_Time', 'Time_Squared', 'Load', 'Temperature', 'Concentration', 'Esterified', 'Hertz_Stress_MPa']
interact_cols = [col for col in input_cols if col not in base_cols]

for suffix, cols in [("Basic", base_cols), ("Interactions", interact_cols)]:
    part_corr = corr_matrix.loc[:, cols]
    target_labels = [config.NAME_MAPPING.get(col, col).replace(' [N]', '').replace(' [°C]', '') for col in target_cols]
    input_labels = [config.NAME_MAPPING.get(col, col).replace(' [N]', '').replace(' [°C]', '') for col in cols]
    
    plt.figure(figsize=(10 if suffix == "Basic" else 12, 4))
    im = plt.imshow(part_corr, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.grid(False)
    for i in range(len(target_cols)):
        for j in range(len(cols)):
            val = part_corr.iloc[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.5 else "black", fontsize=9)
    plt.xticks(range(len(cols)), input_labels, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(target_cols)), target_labels, fontsize=11)
    plt.tight_layout()
    fname = f"Correlation_matrix_{suffix}.png"
    plt.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()
    dynamic_descriptions[fname] = f"Strength of linear relationships between {suffix.lower()} variables (Pearson correlation)."

print("\n--- Generating Learning Curves for selected models... ---")
target_models = {"XGBoost", "LightGBM", "CatBoost", best_model_name}
for res in results:
    if res['Name'] in target_models:
        model_name_safe = res['Name'].replace(' ', '_').replace('(', '').replace(')', '')
        lc_filename = f"Learning_Curve_{model_name_safe}.png"
        print(f"Generating learning curve for {res['Name']}...")
        plot_learning_curve(res['Model'], X, Y, cv=GroupShuffleSplit(n_splits=config.CV_SPLITS, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED), results_dir=config.RESULTS_DIR, groups=groups, num_files=len(np.unique(groups)), filename=lc_filename)
        dynamic_descriptions[lc_filename] = f"Learning curve ({res['Name']})."

html_path = os.path.join(config.RESULTS_DIR, "Eredmenyek_Riport.html")

last_5m_mask = full_df['Time'] >= full_df.groupby('File_ID')['Time'].transform('max') - 300
file_means = full_df[last_5m_mask].groupby('File_ID').agg({
    'Load': 'mean',
    'Temperature': 'mean',
    'Concentration': 'mean',
    'Esterified': 'first',
    'COF': 'mean',
    'Friction absolute integral': 'mean'
})

file_means_desc = file_means[['COF', 'Friction absolute integral']].describe()
file_means_desc.columns = ['Average Coefficient of Friction (Last 5 minutes)', 'Average Friction Absolute Integral (Last 5 minutes)']
file_means_desc.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50% (Median)', '75%', 'Max']

# --- Generate Boxplot for Last 5m Avg COF ---
data_0 = file_means[file_means['Esterified'] == 0]['COF'].dropna().values
data_1 = file_means[file_means['Esterified'] == 1]['COF'].dropna().values

if len(data_0) > 0 and len(data_1) > 0:
    # --- PLOT: Last 5m COF Distribution Boxplot ---
    fig, ax = plt.subplots(figsize=(6.3, 3.15))
    bplot = ax.boxplot([data_0, data_1], positions=[0, 1], patch_artist=True, widths=0.4,
                       medianprops=dict(color='black', linewidth=1.5))
    
    for patch, color in zip(bplot['boxes'], ['purple', 'orange']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not esterified (0)', 'Esterified (1)'])
    ax.set_ylabel('Last 5m Avg COF [-]')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    
    dynamic_descriptions["Last_5m_COF_Distribution.png"] = "Boxplot distribution of the stabilized COF values (average of the last 5 minutes) comparing Base Oil and Esterified Oil."
    plt.savefig(os.path.join(config.RESULTS_DIR, "Last_5m_COF_Distribution.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

# --- Generate Esterification Impact vs Load Plot (Actual Data) ---
file_means['Load_rounded'] = file_means['Load'].round()
load_ester_stats = file_means.groupby(['Load_rounded', 'Esterified'])['COF'].agg(['mean', 'std']).reset_index()

# --- PLOT: Esterification Impact vs Load (Actual Data) ---
fig, ax = plt.subplots(figsize=(6.3, 3.15))

base_stats = load_ester_stats[load_ester_stats['Esterified'] == 0].sort_values('Load_rounded')
ester_stats = load_ester_stats[load_ester_stats['Esterified'] == 1].sort_values('Load_rounded')

if not base_stats.empty:
    ax.errorbar(base_stats['Load_rounded'], base_stats['mean'], yerr=base_stats['std'], 
                fmt='-o', color='purple', label='Not esterified (0)', capsize=4, capthick=1.5, alpha=0.8)
if not ester_stats.empty:
    ax.errorbar(ester_stats['Load_rounded'], ester_stats['mean'], yerr=ester_stats['std'], 
                fmt='-s', color='orange', label='Esterified (1)', capsize=4, capthick=1.5, alpha=0.8)
    
ax.set_xlabel('Load [N]')
ax.set_ylabel('Last 5m Avg COF [-]')
ymin, ymax = plt.gca().get_ylim()
ax.set_ylim(max(0, ymin - 0.02), ymax + (ymax - ymin) * 0.35)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.legend(loc='upper left')

dynamic_descriptions["Esterification_Impact_vs_Load.png"] = "Actual measured steady-state COF across different loads. This physical data demonstrates whether esterification successfully reduced friction under high boundary loads."
plt.savefig(os.path.join(config.RESULTS_DIR, "Esterification_Impact_vs_Load.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

# --- Generate Esterification Impact vs Temperature Plot (Actual Data) ---
file_means['Temp_rounded'] = file_means['Temperature'].round()
temp_ester_stats = file_means.groupby(['Temp_rounded', 'Esterified'])['COF'].agg(['mean', 'std']).reset_index()

# --- PLOT: Esterification Impact vs Temperature (Actual Data) ---
fig, ax = plt.subplots(figsize=(6.3, 3.15))

base_stats_t = temp_ester_stats[temp_ester_stats['Esterified'] == 0].sort_values('Temp_rounded')
ester_stats_t = temp_ester_stats[temp_ester_stats['Esterified'] == 1].sort_values('Temp_rounded')

if not base_stats_t.empty:
    ax.errorbar(base_stats_t['Temp_rounded'], base_stats_t['mean'], yerr=base_stats_t['std'], 
                fmt='-o', color='purple', label='Not esterified (0)', capsize=4, capthick=1.5, alpha=0.8)
if not ester_stats_t.empty:
    ax.errorbar(ester_stats_t['Temp_rounded'], ester_stats_t['mean'], yerr=ester_stats_t['std'], 
                fmt='-s', color='orange', label='Esterified (1)', capsize=4, capthick=1.5, alpha=0.8)
    
ax.set_xlabel('Temperature [°C]')
ax.set_ylabel('Last 5m Avg COF [-]')
ymin, ymax = plt.gca().get_ylim()
ax.set_ylim(max(0, ymin - 0.02), ymax + (ymax - ymin) * 0.35)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.legend(loc='upper left')

dynamic_descriptions["Esterification_Impact_vs_Temperature.png"] = "Actual measured steady-state COF across different temperatures. This demonstrates the effect of temperature on the boundary film stability."
plt.savefig(os.path.join(config.RESULTS_DIR, "Esterification_Impact_vs_Temperature.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

desc_df = full_df[['Time', 'Load', 'Temperature', 'Concentration', 'Esterified', 'COF', 'Friction absolute integral']].describe()
desc_df.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50% (Median)', '75%', 'Max']
desc_df.rename(columns=config.NAME_MAPPING, inplace=True)
desc_df = pd.concat([desc_df, file_means_desc], axis=1)

# --- Filtered Dataset Statistics ---
filtered_full_df = full_df[valid_mask].copy()

last_5m_mask_filt = filtered_full_df['Time'] >= filtered_full_df.groupby('File_ID')['Time'].transform('max') - 300
file_means_filt = filtered_full_df[last_5m_mask_filt].groupby('File_ID')[['COF', 'Friction absolute integral']].mean()
file_means_filt['Esterified'] = filtered_full_df.groupby('File_ID')['Esterified'].first()

file_means_desc_filt = file_means_filt[['COF', 'Friction absolute integral']].describe()
file_means_desc_filt.columns = ['Last 5m Avg COF', 'Last 5m Avg FAI']
file_means_desc_filt.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50% (Median)', '75%', 'Max']

filtered_desc_df = filtered_full_df[['Time', 'Load', 'Temperature', 'Concentration', 'Esterified', 'COF', 'Friction absolute integral']].describe()
filtered_desc_df.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50% (Median)', '75%', 'Max']
filtered_desc_df.rename(columns=config.NAME_MAPPING, inplace=True)
filtered_desc_df = pd.concat([filtered_desc_df, file_means_desc_filt], axis=1)

timing_stats = {
    'total': format_time(time.time() - script_start), 
    'loading': format_time(loading_duration), 
    'shap': format_time(shap_duration) if shap_duration is not None else "N/A", 
    'doe': format_time(total_doe_duration)
}
html_content = generate_html_report(results, xlsx_files, full_df, desc_df, filtered_desc_df, html_path, config.RESULTS_DIR, doe_suggestions_combined, optimum_results, shap_analysis_text, timing_stats, dynamic_descriptions, distribution_summary, combined_plotly_html, safe_validation_points=safe_validation_points)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

excel_path = os.path.join(config.RESULTS_DIR, "Results_Tables.xlsx")
excel_saved = False
attempt = 0

while not excel_saved and attempt < 5:
    try:
        with pd.ExcelWriter(excel_path) as writer:
            excel_cols = ['Name', 'R2_Train', 'R2_Test', 'R2_CV', 'R2_Steady_COF', 'R2_Steady_FAI', 'RMSE_Train', 'RMSE_Test', 'RMSE_Steady_COF', 'RMSE_Steady_FAI', 'MAE_Test', 'Tuning_Training_Time', 'Pred_Time_ms']
            pd.DataFrame(results)[excel_cols].to_excel(writer, sheet_name='Model_Metrics', index=False)
            opt_data = [{'Type': 'Esterified' if s == 1 else 'Not esterified', **r} for s, r in optimum_results.items()]
            pd.DataFrame(opt_data).drop(columns=['CurveTime', 'CurveCOF']).to_excel(writer, sheet_name='Optimums', index=False)
            doe_suggestions_combined.to_excel(writer, sheet_name='DoE_Suggestions', index=False)
            
            val_table_df.to_excel(writer, sheet_name='Validation_Point', index=False)
            
            steady_export_df = file_means.copy().reset_index()
            steady_export_df['Load'] = steady_export_df['Load'].round(0).astype(int)
            steady_export_df['Temperature'] = steady_export_df['Temperature'].round(0).astype(int)
            steady_export_df['Concentration'] = steady_export_df['Concentration'].round(2)
            steady_export_df['Esterified'] = steady_export_df['Esterified'].astype(int)
            
            steady_export_df.rename(columns={
                'File_ID': 'File ID',
                'Load': 'Load [N]',
                'Temperature': 'Temperature [°C]',
                'Concentration': 'Concentration [wt%]',
                'Esterified': 'Esterified [-]',
                'COF': 'COF [-]',
                'Friction absolute integral': 'FAI [-]'
            }, inplace=True)
            
            steady_export_df.to_excel(writer, sheet_name='Steady_States', index=False)
        
            corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
        excel_saved = True
    except PermissionError:
        attempt += 1
        print(f"\nWarning: The file '{excel_path}' is open (e.g., in Excel) and cannot be overwritten.")
        excel_path = os.path.join(config.RESULTS_DIR, f"Results_Tables_alt_{attempt}.xlsx")
        print(f"Attempting to save with new name: {excel_path}")

joblib.dump(best_model_overall, os.path.join(config.RESULTS_DIR, f"Best_Model_{best_model_name.replace(' ', '_')}.pkl"))

print("\nPipeline completed successfully! Opening HTML report...")
webbrowser.open(html_path)
