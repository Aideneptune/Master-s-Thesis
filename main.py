import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import glob
import os
import joblib
import time
import webbrowser
import itertools
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, cross_validate, GroupKFold, ShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin
import shap
import warnings

import config
from transformers import InteractionFeaturesTransformer, VIFSelector
from utils import format_time, create_features, filter_outliers_grouped, plot_pareto_front, plot_learning_curve, generate_html_report

warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=UserWarning, module='catboost')

class PandasStandardScaler(BaseEstimator, TransformerMixin):
    """Egyedi StandardScaler, amely DataFrame-et ad vissza az oszlopnevek megtartásához."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X, y)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        cols = X.columns if isinstance(X, pd.DataFrame) else self.feature_names_in_
        return pd.DataFrame(X_scaled, columns=cols, index=X.index if isinstance(X, pd.DataFrame) else None)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_) if self.feature_names_in_ is not None else None

script_start = time.time()

os.makedirs(config.RESULTS_DIR, exist_ok=True)
os.makedirs(config.CACHE_DIR, exist_ok=True)
for f in glob.glob(os.path.join(config.RESULTS_DIR, "*.*")):
    try:
        os.remove(f)
    except OSError:
        pass

# --- Data Loading ---
data_cache_path = os.path.join(config.CACHE_DIR, "full_df_cache.pkl")
xlsx_files_cache_path = os.path.join(config.CACHE_DIR, "xlsx_files_cache.pkl")
start_loading = time.time()

if config.USE_CACHE and os.path.exists(data_cache_path) and os.path.exists(xlsx_files_cache_path):
    print("--- Loading Data From Cache ---")
    full_df = pd.read_pickle(data_cache_path)
    xlsx_files = joblib.load(xlsx_files_cache_path)
    print(f"Loaded {len(full_df)} rows from {len(xlsx_files)} files from cache.")
else:
    print("--- Loading and Processing Data (Cache not found or disabled) ---")
    base_path = config.BASE_PATH
    print(f"Loading data: {base_path} ...")

    xlsx_files = sorted(glob.glob(os.path.join(base_path, "*.xlsx")))
    if not xlsx_files:
        print("ERROR: No files found! Check the path.")
        sys.exit()

    all_data = []
    for filepath in tqdm(xlsx_files, desc="Loading data", unit="file"):
        try:
            with pd.ExcelFile(filepath, engine='openpyxl') as xls:
                sheet_name = "Sheet Numeric SRA" if "Sheet Numeric SRA" in xls.sheet_names else 0
                df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
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
            
            df = df.reset_index(drop=True)
            df = df.groupby(df.index // config.DOWNSAMPLING_RATE).mean(numeric_only=True).reset_index(drop=True)
            df['File_ID'] = os.path.basename(filepath)
            
            if 'Esterified' in df.columns:
                df['Esterified'] = df['Esterified'].fillna(0).astype(int)
            
            if not df.empty:
                if len(all_data) == 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(df['Time'], df['COF'], label='Eredeti jel', color='silver', alpha=0.7)

                df['COF'] = df['COF'].rolling(window=config.ROLLING_WINDOW_SIZE, min_periods=1, center=False).mean()

                if len(all_data) == 0:
                    plt.plot(df['Time'], df['COF'], label='Filtered signal (Rolling Mean)', color='orange', linewidth=2.5)
                    plt.title(f"Effect of noise filtering: {os.path.basename(filepath)}")
                    plt.xlabel("Time [s]")
                    plt.ylabel("Coefficient of friction (COF) [-]")
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.4)
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering.png"), dpi=300, bbox_inches='tight')
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering.svg"), format='svg', bbox_inches='tight')
                    plt.close()

                all_data.append(df)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error: {os.path.basename(filepath)} - {e}")

    if not all_data:
        sys.exit()

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[full_df['Time'] > 0]
    full_df = full_df[(full_df['COF'] > 0) & (full_df['Friction absolute integral'] > 0)]
    full_df = create_features(full_df)

    full_df = filter_outliers_grouped(full_df, 'File_ID', ['COF', 'Friction absolute integral'])

    if 'Esterified' not in full_df.columns:
        full_df['Esterified'] = 0
    full_df['Esterified'] = full_df['Esterified'].fillna(0).astype(int)

    weight_cols = ['Concentration', 'Load', 'Temperature', 'Esterified']
    counts = full_df.groupby(weight_cols)['Time'].transform('count')
    full_df['Sample_Weight'] = 1.0 / counts
    full_df['Sample_Weight'] = full_df['Sample_Weight'] * (len(full_df) / full_df['Sample_Weight'].sum())

    if config.USE_CACHE:
        print("\nSaving data to cache...")
        full_df.to_pickle(data_cache_path)
        joblib.dump(xlsx_files, xlsx_files_cache_path)

loading_duration = time.time() - start_loading
print(f"Data loading/caching completed in {format_time(loading_duration)}")

print("\n--- Preparing Data and Cross-Validation Folds ---")
X = full_df[['Time', 'Log_Time', 'Time_Squared', 'Load', 'Temperature', 'Concentration', 'Esterified']]
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

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_SEED)
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
global_vif = VIFSelector(threshold=10.0)

X_train_interact = global_interact.fit_transform(X_train)
X_train = global_vif.fit_transform(X_train_interact)

X_test_interact = global_interact.transform(X_test)
X_test = global_vif.transform(X_test_interact)

X_interact = global_interact.transform(X)
X = global_vif.transform(X_interact)

models_config = {
    "XGBoost": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('xgb', MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=config.RANDOM_SEED)))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__xgb__estimator__n_estimators": [50, 100, 150],
            "regressor__xgb__estimator__learning_rate": [0.05, 0.1, 0.2],
            "regressor__xgb__estimator__max_depth": [3, 4, 5],
            "regressor__xgb__estimator__reg_alpha": [0, 0.1, 1],
            "regressor__xgb__estimator__reg_lambda": [5, 10, 20, 50]
        }
    },
    "Neural Network (MLP)": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('mlp', MLPRegressor(random_state=config.RANDOM_SEED, max_iter=500, early_stopping=True))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__mlp__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
            "regressor__mlp__alpha": [0.1, 1.0, 10.0],
            "regressor__mlp__activation": ['relu', 'tanh'],
            "regressor__mlp__learning_rate_init": [0.001, 0.01]
        }
    },
    "Random Forest": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),              # Skálázás Pandas kimenettel
                ('rf', RandomForestRegressor(random_state=config.RANDOM_SEED, n_jobs=-1))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__rf__n_estimators": [100, 200, 300],
            "regressor__rf__max_depth": [None, 10, 20],
            "regressor__rf__min_samples_leaf": [5, 10, 20] # Segít elkerülni a zajra való rátanulást
        }
    },
    "LightGBM": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('lgbm', MultiOutputRegressor(LGBMRegressor(random_state=config.RANDOM_SEED, n_jobs=-1, verbose=-1)))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__lgbm__estimator__n_estimators": [50, 100, 200],
            "regressor__lgbm__estimator__learning_rate": [0.05, 0.1, 0.2],
            "regressor__lgbm__estimator__max_depth": [3, 4, 5]
        }
    },
    "CatBoost": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('cat', MultiOutputRegressor(CatBoostRegressor(random_state=config.RANDOM_SEED, verbose=0, allow_writing_files=False)))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__cat__estimator__iterations": [100, 200, 500],
            "regressor__cat__estimator__learning_rate": [0.05, 0.1, 0.2],
            "regressor__cat__estimator__depth": [3, 4, 5]
        }
    },
    "KNN Regressor": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('knn', KNeighborsRegressor())
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__knn__n_neighbors": [5, 10, 20],
            "regressor__knn__weights": ['uniform']
        }
    },
    "Polynomial Ridge Regression": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('ridge', Ridge())
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__ridge__alpha": [0.1, 1.0, 10.0, 100.0]
        }
    }
}

range_conc = np.arange(0.0, 0.61, 0.05)
range_load = np.arange(10, 201, 20)
range_temp = np.arange(40, 121, 10)
combos = list(itertools.product(range_conc, range_load, range_temp))
grid_df = pd.DataFrame(combos, columns=['Concentration', 'Load', 'Temperature'])
grid_df['Esterified'] = config.PLOT_ESTERIFIED_STATE
grid_df['Time'] = 7200
grid_df = create_features(grid_df)
grid_df = grid_df[X_cols_raw]

gkf_cv = GroupKFold(n_splits=5)

# --- Model Training ---
models_cache_path = os.path.join(config.CACHE_DIR, "models_cache.pkl")
if config.USE_CACHE and os.path.exists(models_cache_path):
    print("\n--- Loading Trained Models From Cache ---")
    cached_models = joblib.load(models_cache_path)
    results = cached_models['results']
    best_model_overall = cached_models['best_model_overall']
    best_model_name = cached_models['best_model_name']
    best_r2_overall = max(r['R2_CV'] for r in results)
    print("Models loaded from cache.")
else:
    print("\n--- Training Models (Cache not found or disabled) ---")
    results = []
    best_model_overall = None
    best_r2_overall = -np.inf
    best_model_name = ""
    for name, cfg in tqdm(models_config.items(), desc="Training models"):
        start_model_total = time.time()
        best_params = {}
        
        if cfg["params"]:
            fit_params = {}
            if "Random Forest" in name: fit_params['rf__sample_weight'] = weights_train
            elif "XGBoost" in name: fit_params['xgb__sample_weight'] = weights_train
            elif "LightGBM" in name: fit_params['lgbm__sample_weight'] = weights_train
            elif "CatBoost" in name: fit_params['cat__sample_weight'] = weights_train
            elif "Polynomial" in name: fit_params['ridge__sample_weight'] = weights_train
            
            search = RandomizedSearchCV(cfg["model"], cfg["params"], n_iter=50, cv=gkf_cv, scoring='r2', n_jobs=-1, random_state=config.RANDOM_SEED)
            search.fit(X_train, y_train, groups=groups_train, **fit_params)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
        else:
            best_estimator = cfg["model"]
            best_estimator.fit(X_train, y_train)
            best_params = "Default"

        fit_params_final = {}
        if "Random Forest" in name: fit_params_final['rf__sample_weight'] = weights_train
        elif "XGBoost" in name: fit_params_final['xgb__sample_weight'] = weights_train
        elif "LightGBM" in name: fit_params_final['lgbm__sample_weight'] = weights_train
        elif "CatBoost" in name: fit_params_final['cat__sample_weight'] = weights_train
        elif "Polynomial" in name: fit_params_final['ridge__sample_weight'] = weights_train
            
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
        
        cv_scores = cross_validate(best_estimator, X, Y, cv=gkf_cv, groups=groups, scoring=['r2', 'neg_root_mean_squared_error'])
        avg_r2 = np.mean(cv_scores['test_r2'])
        
        feature_imp = None
        if "Random Forest" in name: feature_imp = best_estimator.regressor_.named_steps['rf'].feature_importances_
        elif "XGBoost" in name: feature_imp = np.mean([est.feature_importances_ for est in best_estimator.regressor_.named_steps['xgb'].estimators_], axis=0)
        elif "LightGBM" in name: feature_imp = np.mean([est.feature_importances_ for est in best_estimator.regressor_.named_steps['lgbm'].estimators_], axis=0)
        elif "CatBoost" in name: feature_imp = np.mean([est.feature_importances_ for est in best_estimator.regressor_.named_steps['cat'].estimators_], axis=0)

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
            
        results.append({
            "Name": name, "Model": best_estimator, "R2_Train": r2_train, "R2_Test": r2_test, "R2_COF": r2_cof, "R2_FAI": r2_fai, "R2_CV": avg_r2,
            "RMSE_Train": rmse_train, "RMSE_Test": rmse_test, "RMSE_COF": rmse_cof, "RMSE_FAI": rmse_fai, "MAE_Test": mae_test,
            "Tuning_Training_Time": tuning_training_time, "Pred_Time_ms": pred_time_ms, "Feature_Imp": feature_imp,
            "Opt_Conc": opt_conc, "Opt_Load": opt_load, "Opt_Temp": opt_temp, "Pred_COF": pred_cof_5m, "Pred_FAI": pred_fai_5m,
            "Best_Params": best_params, "Selected_Features": selected_features_model
        })
        
        if avg_r2 > best_r2_overall:
            best_r2_overall = avg_r2
            best_model_overall = best_estimator
            best_model_name = name

    if config.USE_CACHE:
        print("\nSaving trained models to cache...")
        joblib.dump({
            'results': results,
            'best_model_overall': best_model_overall,
            'best_model_name': best_model_name
        }, models_cache_path)

print(f"\nBest model found: {best_model_name} with average R2 CV: {best_r2_overall:.4f}")
print(f"Retraining {best_model_name} on the full dataset...")

fit_params_full = {}
if "Random Forest" in best_model_name: fit_params_full['rf__sample_weight'] = full_df['Sample_Weight']
elif "XGBoost" in best_model_name: fit_params_full['xgb__sample_weight'] = full_df['Sample_Weight']
elif "LightGBM" in best_model_name: fit_params_full['lgbm__sample_weight'] = full_df['Sample_Weight']
elif "CatBoost" in best_model_name: fit_params_full['cat__sample_weight'] = full_df['Sample_Weight']
elif "Polynomial" in best_model_name: fit_params_full['ridge__sample_weight'] = full_df['Sample_Weight']
best_model_overall.fit(X, Y, **fit_params_full)

template_df = all_data[0].dropna(subset=['Time', 'Load', 'Temperature']).sort_values('Time')
template_df = template_df[(template_df['Temperature'] != 0) & (template_df['Load'] != 0)]
template_df = template_df[template_df['Time'] > 0]
optimum_results = {}

print("\n--- Calculating Optimums over the Parameter Grid ---")

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
    
    if ester_state == 1:
        plot_pareto_front(config.RESULTS_DIR, preds, grid_df['Temperature'], title=f"Pareto front over the full parameter grid - Esterified (1)")

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
            stab_inputs.append({'Time': 7200, 'Load': l, 'Temperature': t, 'Concentration': opt_conc, 'Esterified': ester_state})
    
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
doe_cache_path = os.path.join(config.CACHE_DIR, "doe_cache.pkl")
if config.USE_CACHE and os.path.exists(doe_cache_path):
    print("\n--- Loading DoE Suggestions From Cache ---")
    cached_doe = joblib.load(doe_cache_path)
    doe_suggestions = cached_doe['doe_suggestions']
    doe_duration = cached_doe['doe_duration']
    print("DoE suggestions loaded from cache.")
else:
    print("\n--- Starting Design of Experiments (DoE) generation (Cache not found or disabled) ---")
    start_doe = time.time()
    selected_feats_doe = global_vif.selected_features_

    X_doe = X.copy()
    grid_doe = global_vif.transform(global_interact.transform(grid_df))

    doe_model = BaggingRegressor(estimator=RandomForestRegressor(n_estimators=20, random_state=config.RANDOM_SEED, n_jobs=1), n_estimators=10, random_state=config.RANDOM_SEED, n_jobs=1)
    print("Training DoE model on selected features...")
    doe_model.fit(X_doe, Y)

    print("Predicting uncertainty on the parameter grid...")
    # Ne használj .values kiterjesztést, maradjon DataFrame
    doe_preds = np.array([np.maximum(est.predict(grid_doe), config.PREDICTION_LOWER_BOUND) for est in doe_model.estimators_])
    std_cof = np.std(doe_preds[:, :, 0], axis=0)
    std_fai = np.std(doe_preds[:, :, 1], axis=0)

    doe_features = ['Concentration', 'Load', 'Temperature']
    scaler_doe = MinMaxScaler()
    # pd.DataFrame használata .values helyett
    X_grid_scaled = pd.DataFrame(scaler_doe.fit_transform(grid_df[doe_features]), columns=doe_features)
    X_existing_scaled = pd.DataFrame(scaler_doe.transform(full_df[doe_features]), columns=doe_features)

    nbrs = NearestNeighbors(n_neighbors=1).fit(X_existing_scaled)
    dist_metric = nbrs.kneighbors(X_grid_scaled)[0].flatten()

    norm_std_cof = (std_cof - std_cof.min()) / (std_cof.max() - std_cof.min() + 1e-9)
    norm_std_fai = (std_fai - std_fai.min()) / (std_fai.max() - std_fai.min() + 1e-9)
    avg_uncertainty = (norm_std_cof + norm_std_fai) / 2
    norm_dist = (dist_metric - dist_metric.min()) / (dist_metric.max() - dist_metric.min() + 1e-9)

    doe_grid = grid_df.copy()
    doe_grid['Uncertainty_COF'] = std_cof
    doe_grid['Uncertainty_FAI'] = std_fai
    doe_grid['Distance'] = dist_metric
    doe_grid['Score'] = 0.7 * avg_uncertainty + 0.3 * norm_dist

    existing_set = set((round(row['Concentration'], 2), int(row['Load']), int(row['Temperature'])) for _, row in full_df[['Concentration', 'Load', 'Temperature']].iterrows())
    doe_candidates = doe_grid[~doe_grid.apply(lambda row: (round(row['Concentration'], 2), int(row['Load']), int(row['Temperature'])) in existing_set, axis=1)].sort_values(by='Score', ascending=False)

    final_suggestions = []
    candidates_pool = doe_candidates.copy()
    for _ in range(5):
        if candidates_pool.empty: break
        best_candidate = candidates_pool.iloc[0]
        final_suggestions.append(best_candidate)
        mask_load = (candidates_pool['Load'] >= best_candidate['Load'] - 20) & (candidates_pool['Load'] <= best_candidate['Load'] + 20)
        mask_temp = (candidates_pool['Temperature'] >= best_candidate['Temperature'] - 10) & (candidates_pool['Temperature'] <= best_candidate['Temperature'] + 10)
        candidates_pool.loc[mask_load & mask_temp, 'Score'] *= 0.5
        candidates_pool = candidates_pool.drop(best_candidate.name).sort_values(by='Score', ascending=False)

    doe_suggestions = pd.DataFrame(final_suggestions)
    doe_duration = time.time() - start_doe

    if config.USE_CACHE:
        print("Saving DoE suggestions to cache...")
        joblib.dump({'doe_suggestions': doe_suggestions, 'doe_duration': doe_duration}, doe_cache_path)

doe_img_files = []
for i, (_, row) in enumerate(doe_suggestions.iterrows()):
    sim_input = create_features(pd.DataFrame({'Time': template_df['Time'], 'Load': row['Load'], 'Temperature': row['Temperature'], 'Concentration': row['Concentration'], 'Esterified': config.PLOT_ESTERIFIED_STATE}))[X_cols_raw]
    sim_input_trans = global_vif.transform(global_interact.transform(sim_input))
    curve_preds = np.maximum(best_model_overall.predict(sim_input_trans), config.PREDICTION_LOWER_BOUND)
    plt.figure(figsize=(10, 5))
    plt.plot(template_df['Time'], curve_preds[:, 0], color='purple', linewidth=2.5)
    plt.title(f"DoE suggestion #{i+1}: {row['Concentration']:.2f}% | {int(row['Load'])}N | {int(row['Temperature'])}°C")
    plt.grid(True, linestyle='--', alpha=0.4)
    fname = f"DoE_Suggestion_{i+1}.png"
    plt.savefig(os.path.join(config.RESULTS_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()
    doe_img_files.append(fname)
doe_suggestions['Image_File'] = doe_img_files

print("\n--- Generating Feature Importance Plot ---")
best_res = next(r for r in results if r['Name'] == best_model_name)
if best_res['Feature_Imp'] is not None:
    print(f"Generating feature importance plot for {best_model_name}...")
    # Csökkenő sorrend beállítása
    sorted_idx = np.argsort(best_res['Feature_Imp'])
    sorted_feats = [best_res['Selected_Features'][i] for i in sorted_idx]
    sorted_imp = best_res['Feature_Imp'][sorted_idx]
    
    display_feats = [config.NAME_MAPPING.get(f, f) for f in sorted_feats]
    
    plt.figure(figsize=(8, 5))
    plt.barh(display_feats, sorted_imp, color='purple')
    plt.title(f"Feature importance ({best_model_name})")
    plt.savefig(os.path.join(config.RESULTS_DIR, "Feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()

print("\n--- SHAP Analysis ---")
shap_analysis_text = ""
shap_duration = None
tree_models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]

# Mindig a legjobb famodellről készítünk SHAP elemzést
tree_results = [r for r in results if any(m in r['Name'] for m in tree_models)]
if tree_results:
    best_tree_res = sorted(tree_results, key=lambda x: x['R2_CV'], reverse=True)[0]
    shap_model = best_tree_res['Model']
    shap_model_name = best_tree_res['Name']
    print(f"Generating SHAP analysis for the best tree-based model: {shap_model_name}...")

    try:
        start_shap = time.time()
        scaler_step = shap_model.regressor_.named_steps['scaler']
        
        X_test_vif = X_test
        vif_feature_names = global_vif.get_feature_names_out()
        
        X_test_scaled = pd.DataFrame(
            scaler_step.transform(X_test_vif), 
            columns=vif_feature_names, 
            index=X_test.index
        )
        
        X_test_display = pd.DataFrame(X_test_vif.values, index=X_test.index, columns=vif_feature_names)
        X_test_display.rename(columns=config.NAME_MAPPING, inplace=True)
        
        model_step_name = None
        if "XGBoost" in shap_model_name: model_step_name = 'xgb'
        elif "LightGBM" in shap_model_name: model_step_name = 'lgbm'
        elif "CatBoost" in shap_model_name: model_step_name = 'cat'
        elif "Random Forest" in shap_model_name: model_step_name = 'rf'

        if model_step_name:
            if model_step_name == 'rf':
                model_obj = shap_model.regressor_.named_steps[model_step_name]
            else:
                model_obj = shap_model.regressor_.named_steps[model_step_name].estimators_[0]
                
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X_test_scaled)
            
            if isinstance(shap_values, list):
                shap_values_to_plot = shap_values[0]
            else:
                shap_values_to_plot = shap_values

            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_to_plot, X_test_display, show=False)
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_feature_impact.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            mean_shap = np.abs(shap_values_to_plot).mean(axis=0)
            top_3 = sorted(dict(zip(X_test_display.columns, mean_shap)).items(), key=lambda x: x[1], reverse=True)[:3]
            shap_analysis_text = "<ul>" + "".join([f"<li><strong>{f}</strong> (SHAP: {i:.4f})</li>" for f, i in top_3]) + "</ul>"
            shap_duration = time.time() - start_shap
            print("SHAP analysis completed.")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Warning: SHAP analysis failed - {e}")
else:
    print("No tree-based models found for SHAP analysis.")

print("\n--- Generating Evaluation Plots ---")
plt.figure(figsize=(10, 6))
plt.plot(optimum_results[0]['CurveTime'], optimum_results[0]['CurveCOF'], color='purple', label="Base Oil")
plt.plot(optimum_results[1]['CurveTime'], optimum_results[1]['CurveCOF'], color='orange', label="Esterified")
plt.title("Optimum curve comparison")
plt.legend()
plt.savefig(os.path.join(config.RESULTS_DIR, "Optimum_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
subset = full_df.groupby(['Load', 'Temperature', 'Concentration', 'Esterified'])['File_ID'].nunique().reset_index(name='Count')
ax.scatter(subset['Load'], subset['Temperature'], subset['Concentration'], c=subset['Esterified'], cmap='coolwarm', s=subset['Count']*100, alpha=0.6)
plt.savefig(os.path.join(config.RESULTS_DIR, "3D_distribution_of_input_data.png"), dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(full_df['Load'], full_df['Temperature'], full_df['Concentration'], 
           c='blue', marker='o', s=15, alpha=0.5, label='Existing Measurements')
ax.scatter(doe_suggestions['Load'], doe_suggestions['Temperature'], doe_suggestions['Concentration'], 
           c='red', s=800, alpha=0.2, label='DoE Space Coverage')
ax.scatter(doe_suggestions['Load'], doe_suggestions['Temperature'], doe_suggestions['Concentration'], 
           c='red', marker='x', s=50, label='DoE Points')
ax.set_xlabel('Load [N]')
ax.set_ylabel('Temperature [°C]')
ax.set_zlabel('Concentration [%]')
plt.legend()
plt.title('3D Map of Existing Data and DoE Suggestions')
plt.savefig(os.path.join(config.RESULTS_DIR, "DoE_3D_map.png"), dpi=300, bbox_inches='tight')
plt.close()

L_grid, T_grid = np.meshgrid(np.linspace(10, 200, 100), np.linspace(40, 120, 100))
heatmap_input = create_features(pd.DataFrame({'Time': 7200, 'Load': L_grid.ravel(), 'Temperature': T_grid.ravel(), 'Concentration': opt_conc, 'Esterified': config.PLOT_ESTERIFIED_STATE}))[X_cols_raw]
heatmap_input_trans = global_vif.transform(global_interact.transform(heatmap_input))
cof_grid = np.maximum(best_model_overall.predict(heatmap_input_trans), config.PREDICTION_LOWER_BOUND)[:, 0].reshape(L_grid.shape)
plt.figure(figsize=(10, 8))
contourf_plot = plt.contourf(L_grid, T_grid, cof_grid, levels=100, cmap='plasma')
cbar = plt.colorbar(contourf_plot)
cbar.set_label('Coefficient of friction (COF) [-]')
contours_lines = plt.contour(L_grid, T_grid, cof_grid, levels=10, colors='black', alpha=0.5)
plt.clabel(contours_lines, inline=True, fontsize=10, fmt='%.3f')
plt.xlabel("Load [N]")
plt.ylabel("Temperature [°C]")
plt.title("Estimated COF Heatmap")
plt.savefig(os.path.join(config.RESULTS_DIR, "COF_heatmap.png"), dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 10))
corr_cols = ['Time', 'Load', 'Temperature', 'Concentration', 'Esterified', 'COF', 'Friction absolute integral']
corr_labels = ['Time', 'Load', 'Temperature', 'Concentration', 'Esterified', 'COF', 'Friction Absolute Integral']
corr_matrix = full_df[corr_cols].corr()
im = plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(im)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
plt.xticks(range(len(corr_labels)), corr_labels, rotation=45, ha='right')
plt.yticks(range(len(corr_labels)), corr_labels)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, "Correlation_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

plot_learning_curve(best_model_overall, X, Y, title=f"Learning curve ({best_model_name})", cv=GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=config.RANDOM_SEED), results_dir=config.RESULTS_DIR, groups=groups, num_files=len(np.unique(groups)))

html_path = os.path.join(config.RESULTS_DIR, "Eredmenyek_Riport.html")
desc_df = full_df[['Time', 'Load', 'Temperature', 'Concentration', 'Esterified', 'COF', 'Friction absolute integral']].describe()
desc_df.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50% (Median)', '75%', 'Max']
desc_df.rename(columns=config.NAME_MAPPING, inplace=True)

timing_stats = {
    'total': format_time(time.time() - script_start), 
    'loading': format_time(loading_duration), 
    'shap': format_time(shap_duration) if shap_duration is not None else "N/A", 
    'doe': format_time(doe_duration)
}
html_content = generate_html_report(results, xlsx_files, full_df, desc_df, html_path, config.RESULTS_DIR, doe_suggestions, optimum_results, shap_analysis_text, timing_stats)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

excel_path = os.path.join(config.RESULTS_DIR, "Results_Tables.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    pd.DataFrame(results)[['Name', 'R2_Train', 'R2_Test', 'R2_CV', 'RMSE_Train', 'RMSE_Test', 'MAE_Test', 'Tuning_Training_Time', 'Pred_Time_ms']].to_excel(writer, sheet_name='Model_Metrics', index=False)
    opt_data = [{'Type': 'Esterified' if s == 1 else 'Base Oil', **r} for s, r in optimum_results.items()]
    pd.DataFrame(opt_data).drop(columns=['CurveTime', 'CurveCOF']).to_excel(writer, sheet_name='Optimums', index=False)
    doe_suggestions.drop(columns=['Image_File']).to_excel(writer, sheet_name='DoE_Suggestions', index=False)

joblib.dump(best_model_overall, os.path.join(config.RESULTS_DIR, f"Best_Model_{best_model_name.replace(' ', '_')}.pkl"))
print("\nPipeline completed successfully! Opening HTML report...")
webbrowser.open(html_path)
