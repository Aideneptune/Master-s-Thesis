import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
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
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, cross_validate, GroupKFold, ShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin, clone
import shap
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
print("\n--- Adatok beolvasása folyamatban... ---")
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
            # df = df[df['Time'] > 10.0] # Kezdeti vágás (Hard Trim): első 10 mp eldobása - Visszaállítva a korábbi állapotra
            df = df.reset_index(drop=True)
            df = df.groupby(df.index // config.DOWNSAMPLING_RATE).mean(numeric_only=True).reset_index(drop=True)
            df['File_ID'] = os.path.basename(filepath)
            
            if 'Esterified' in df.columns:
                df['Esterified'] = df['Esterified'].fillna(0).astype(int)
            
            if not df.empty:
                if len(all_data) == 0:
                    plt.figure(figsize=(6.3, 3.15))
                    plt.plot(df['Time'], df['COF'], label='Eredeti jel', color='silver', alpha=0.7)

                df['COF'] = df['COF'].rolling(window=config.ROLLING_WINDOW_SIZE, min_periods=1, center=False).mean() # Visszaállítva a korábbi rolling mean-re
                if len(all_data) == 0:
                    plt.plot(df['Time'], df['COF'], label='Filtered signal (Rolling Mean)', color='orange', linewidth=2.5) # Visszaállítva a korábbi label-re
                    plt.xlabel("Time [s]")
                    plt.ylabel("Coefficient of friction (COF) [-]")
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.4)
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering.svg"), format='svg', bbox_inches='tight', pad_inches=0.1)
                    plt.close()

                # Generate a second example plot from a different file
                if len(all_data) == 49: # Corresponds to the 50th file (0-indexed)
                    plt.figure(figsize=(6.3, 3.15))
                    original_cof_50 = df['COF'].copy()
                    plt.plot(df['Time'], original_cof_50, label='Eredeti jel (50. fájl)', color='silver', alpha=0.7)
                    filtered_cof_50 = df['COF'].rolling(window=config.ROLLING_WINDOW_SIZE, min_periods=1, center=False).mean()
                    plt.plot(df['Time'], filtered_cof_50, label='Filtered signal (Rolling Mean)', color='blue', linewidth=2.5)
                    plt.xlabel("Time [s]")
                    plt.ylabel("Coefficient of friction (COF) [-]")
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.4)
                    plt.savefig(os.path.join(config.RESULTS_DIR, "Effect_of_noise_filtering_2.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
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

    full_df = filter_outliers_grouped(full_df, 'File_ID', ['COF', 'Friction absolute integral'], low_q=0.05, high_q=0.95)

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
global_vif = VIFSelector(threshold=5.0)

X_train_interact = global_interact.fit_transform(X_train)
X_train = global_vif.fit_transform(X_train_interact)

X_test_interact = global_interact.transform(X_test)
X_test = global_vif.transform(X_test_interact)

X_interact = global_interact.transform(X)
X = global_vif.transform(X_interact)

range_conc = np.arange(0.0, 0.61, 0.05)
range_load = np.arange(10, 201, 20)
range_temp = np.arange(40, 121, 10)
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

class PreFittedVotingRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, estimators, names):
        self.estimators = estimators
        self.names = names
        
    def fit(self, X, y, **fit_params):
        for name, est in zip(self.names, self.estimators):
            fparams = {}
            if "Random Forest" in name: fparams['rf__sample_weight'] = fit_params.get('sample_weight')
            elif "XGBoost" in name: fparams['xgb__sample_weight'] = fit_params.get('sample_weight')
            elif "LightGBM" in name: fparams['lgbm__sample_weight'] = fit_params.get('sample_weight')
            elif "CatBoost" in name: fparams['cat__sample_weight'] = fit_params.get('sample_weight')
            elif "Polynomial" in name: fparams['ridge__sample_weight'] = fit_params.get('sample_weight')
            fparams = {k: v for k, v in fparams.items() if v is not None}
            est.fit(X, y, **fparams)
        return self
        
    def predict(self, X):
        return np.mean([est.predict(X) for est in self.estimators], axis=0)

# --- Model Training ---
print("\n--- Modellek betanítása és tuningolása... ---")
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
                    if any(isinstance(x, (tuple, list)) for x in param_values):
                        idx = trial.suggest_categorical(param_name + "_idx", list(range(len(param_values))))
                        sampled_params[param_name] = param_values[idx]
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

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="maximize")
            def logging_callback(study, frozen_trial):
                print(f"  Trial {frozen_trial.number} finished with R2 CV: {frozen_trial.value:.4f} | Best so far: {study.best_value:.4f} (Trial {study.best_trial.number})")
            study.optimize(objective, n_trials=50, callbacks=[logging_callback])

            try:
                import optuna.visualization.matplotlib as ovm
                if len(study.trials) > 1:
                    ovm.plot_param_importances(study)
                    safe_model_name = name.replace(' ', '_').replace('(', '').replace(')', '')
                    plot_filename = f"Optuna_Importances_{safe_model_name}.png"
                    fig = plt.gcf()
                    fig.set_size_inches(10, 6)
                    ax = plt.gca()
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
        plt.plot(curve_time_model, curve_cof_model, label=f'Optimized (Esterified={config.PLOT_ESTERIFIED_STATE})', color='orange', linewidth=2.5)
        plt.plot(curve_time_model, curve_cof_base, label='Base Oil (Esterified=0)', color='purple', linewidth=2, linestyle='--')
        plt.ylim(config.PLOT_SETTINGS['cof_ylim'])
        if run_in_model > 0:
            plt.axvline(x=run_in_model, color='grey', linestyle='--', label='Run-in time')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        plt.xlabel('Time [s]')
        plt.ylabel('Coefficient of friction (COF) [-]')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        curve_filename = f"optimum_curve_{safe_name}.png"
        ester_text = "Esterified" if config.PLOT_ESTERIFIED_STATE == 1 else "Not esterified"
        dynamic_descriptions[curve_filename] = f"Optimum Curve - {name} ({opt_conc:.2f}% | {int(opt_load)}N | {int(opt_temp)}°C, {ester_text}). Run-in time: {run_in_model:.1f} s."
        plt.savefig(os.path.join(config.RESULTS_DIR, curve_filename), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        results.append({
            "Name": name, "Model": best_estimator, "R2_Train": r2_train, "R2_Test": r2_test, "R2_COF": r2_cof, "R2_FAI": r2_fai, "R2_CV": avg_r2,
            "RMSE_Train": rmse_train, "RMSE_Test": rmse_test, "RMSE_COF": rmse_cof, "RMSE_FAI": rmse_fai, "MAE_Test": mae_test,
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
    top3_models = [r['Model'] for r in top3_results]
    top3_names = [r['Name'] for r in top3_results]
    print(f"Top 3 models selected for ensemble: {', '.join(top3_names)}")
    
    ensemble_name = "Ensemble (Top 3 Voting)"
    ensemble_model = PreFittedVotingRegressor(top3_models, top3_names)
    
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
    plt.plot(curve_time_model_ens, curve_cof_model_ens, label=f'Optimized (Esterified={config.PLOT_ESTERIFIED_STATE})', color='orange', linewidth=2.5)
    plt.plot(curve_time_model_ens, curve_cof_base_ens, label='Base Oil (Esterified=0)', color='purple', linewidth=2, linestyle='--')
    plt.ylim(config.PLOT_SETTINGS['cof_ylim'])
    if run_in_model_ens > 0:
        plt.axvline(x=run_in_model_ens, color='grey', linestyle='--', label='Run-in time')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    plt.xlabel('Time [s]')
    plt.ylabel('Coefficient of friction (COF) [-]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    safe_name_ens = ensemble_name.replace(' ', '_').replace('(', '').replace(')', '')
    curve_filename_ens = f"optimum_curve_{safe_name_ens}.png"
    ester_text_ens = "Esterified" if config.PLOT_ESTERIFIED_STATE == 1 else "Not esterified"
    dynamic_descriptions[curve_filename_ens] = f"Optimum Curve - {ensemble_name} ({opt_conc_ens:.2f}% | {int(opt_load_ens)}N | {int(opt_temp_ens)}°C, {ester_text_ens}). Run-in time: {run_in_model_ens:.1f} s."
    plt.savefig(os.path.join(config.RESULTS_DIR, curve_filename_ens), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

    results.append({
        "Name": ensemble_name, "Model": ensemble_model, "R2_Train": r2_train_ens, "R2_Test": r2_test_ens, "R2_COF": r2_cof_ens, "R2_FAI": r2_fai_ens, "R2_CV": avg_r2_ens,
        "RMSE_Train": rmse_train_ens, "RMSE_Test": rmse_test_ens, "RMSE_COF": rmse_cof_ens, "RMSE_FAI": rmse_fai_ens, "MAE_Test": mae_test_ens,
        "Tuning_Training_Time": tuning_training_time_ens, "Pred_Time_ms": pred_time_ms_ens, "Feature_Imp": None,
        "Opt_Conc": opt_conc_ens, "Opt_Load": opt_load_ens, "Opt_Temp": opt_temp_ens, "Pred_COF": pred_cof_5m_ens, "Pred_FAI": pred_fai_5m_ens,
        "Best_Params": f"Voting of: {', '.join(top3_names)}", "Selected_Features": global_vif.selected_features_,
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
print(f"Retraining {best_model_name} on the full dataset...")

fit_params_full = {}
if "Ensemble" in best_model_name: fit_params_full['sample_weight'] = full_df['Sample_Weight'].values
elif "Random Forest" in best_model_name: fit_params_full['rf__sample_weight'] = full_df['Sample_Weight'].values
elif "XGBoost" in best_model_name: fit_params_full['xgb__sample_weight'] = full_df['Sample_Weight'].values
elif "LightGBM" in best_model_name: fit_params_full['lgbm__sample_weight'] = full_df['Sample_Weight'].values
elif "CatBoost" in best_model_name: fit_params_full['cat__sample_weight'] = full_df['Sample_Weight'].values
elif "Polynomial" in best_model_name: fit_params_full['ridge__sample_weight'] = full_df['Sample_Weight'].values
best_model_overall.fit(X, Y, **fit_params_full)

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
print("\n--- DoE pontok számítása... ---")

all_doe_suggestions = {}
total_doe_duration = 0

for ester_state_doe in [0, 1]:
    state_str = "Esterified" if ester_state_doe == 1 else "Base Oil"
    doe_cache_path = os.path.join(config.CACHE_DIR, f"doe_cache_{ester_state_doe}.pkl")
    
    if config.USE_CACHE and os.path.exists(doe_cache_path):
        print(f"\n--- Loading DoE Suggestions From Cache for {state_str} ---")
        cached_doe = joblib.load(doe_cache_path)
        doe_suggestions = cached_doe['doe_suggestions']
        doe_duration = cached_doe['doe_duration']
    else:
        print(f"\n--- Starting DoE generation for {state_str} (Cache not found or disabled) ---")
        start_doe = time.time()
        
        top3_doe_models = [r['Model'] for r in sorted(results, key=lambda x: x['R2_CV'], reverse=True)[:3]]
        
        doe_combos = list(itertools.product(range_conc, range_load, range_temp, [ester_state_doe]))
        doe_grid_df = pd.DataFrame(doe_combos, columns=['Concentration', 'Load', 'Temperature', 'Esterified'])
        doe_grid_df['Time'] = 7200
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
        dist_metric = nbrs.kneighbors(X_grid_scaled)[0].flatten()

        norm_std_cof = (std_cof - std_cof.min()) / (std_cof.max() - std_cof.min() + 1e-9)
        norm_std_fai = (std_fai - std_fai.min()) / (std_fai.max() - std_fai.min() + 1e-9)
        avg_uncertainty = (norm_std_cof + norm_std_fai) / 2
        norm_dist = (dist_metric - dist_metric.min()) / (dist_metric.max() - dist_metric.min() + 1e-9)

        doe_grid = doe_grid_df.copy()
        doe_grid['Uncertainty_COF'] = std_cof
        doe_grid['Uncertainty_FAI'] = std_fai
        doe_grid['Distance'] = dist_metric
        doe_grid['Score'] = config.UNCERTAINTY_WEIGHT * avg_uncertainty + config.SPARSITY_WEIGHT * norm_dist

        existing_set = set((round(row['Concentration'], 2), int(row['Load']), int(row['Temperature']), int(row['Esterified'])) for _, row in existing_subset.iterrows())
        doe_candidates = doe_grid[~doe_grid.apply(lambda row: (round(row['Concentration'], 2), int(row['Load']), int(row['Temperature']), int(row['Esterified'])) in existing_set, axis=1)].sort_values(by='Score', ascending=False)

        final_suggestions = []
        candidates_pool = doe_candidates.copy()
        for _ in range(5):
            if candidates_pool.empty: break
            best_candidate = candidates_pool.iloc[0]
            final_suggestions.append(best_candidate)
            mask_load = (candidates_pool['Load'] >= best_candidate['Load'] - 20) & (candidates_pool['Load'] <= best_candidate['Load'] + 20)
            mask_temp = (candidates_pool['Temperature'] >= best_candidate['Temperature'] - 10) & (candidates_pool['Temperature'] <= best_candidate['Temperature'] + 10)
            mask_conc = (candidates_pool['Concentration'] >= best_candidate['Concentration'] - 0.1) & (candidates_pool['Concentration'] <= best_candidate['Concentration'] + 0.1)
            candidates_pool.loc[mask_load & mask_temp & mask_conc, 'Score'] *= 0.5
            candidates_pool = candidates_pool.drop(best_candidate.name).sort_values(by='Score', ascending=False)

        doe_suggestions = pd.DataFrame(final_suggestions)
        doe_duration = time.time() - start_doe

        if config.USE_CACHE:
            print(f"Saving DoE suggestions for {state_str} to cache...")
            joblib.dump({'doe_suggestions': doe_suggestions, 'doe_duration': doe_duration}, doe_cache_path)
    
    all_doe_suggestions[ester_state_doe] = doe_suggestions
    total_doe_duration += doe_duration

doe_img_files = []
doe_suggestions_combined = pd.concat(all_doe_suggestions.values()).reset_index(drop=True)
for i, (_, row) in enumerate(doe_suggestions_combined.iterrows()):
    sim_input = create_features(pd.DataFrame({'Time': template_df['Time'], 'Load': row['Load'], 'Temperature': row['Temperature'], 'Concentration': row['Concentration'], 'Esterified': row['Esterified']}))[X_cols_raw]
    sim_input_trans = global_vif.transform(global_interact.transform(sim_input))
    curve_preds = np.maximum(best_model_overall.predict(sim_input_trans), config.PREDICTION_LOWER_BOUND)
    fig, ax = plt.subplots(figsize=(6.3, 3.15))
    plt.plot(template_df['Time'], curve_preds[:, 0], color='purple', linewidth=2.5)
    plt.ylim(bottom=0.0, top=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
    plt.grid(True, linestyle='--', alpha=0.4)
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
    # Csökkenő sorrend beállítása
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
print("\n--- SHAP analízis generálása... ---")
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
        safe_mapping = {k: v for k, v in config.NAME_MAPPING.items() if k in X_test_display.columns}
        X_test_display.rename(columns=safe_mapping, inplace=True)
        
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

            plt.figure(figsize=(6.3, 3.15))
            shap.summary_plot(shap_values_to_plot, X_test_display, show=False)
            plt.savefig(os.path.join(config.RESULTS_DIR, "SHAP_feature_impact.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
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

print("\n--- Reziduális elemzés (Residual Plot) generálása... ---")
y_test_pred = np.maximum(best_model_overall.predict(X_test), config.PREDICTION_LOWER_BOUND)
residuals = y_test_pred[:, 0] - y_test['COF'].values
fig, ax = plt.subplots(figsize=(6.3, 3.15))
plt.scatter(y_test['COF'].values, residuals, alpha=0.6, color='blue', edgecolors='k')
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel("Actual COF")
plt.ylabel("Residual Error")
plt.grid(True, linestyle='--', alpha=0.4)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
dynamic_descriptions["residuals_best_model.png"] = f"Residual Plot - {best_model_name} (COF)."
plt.savefig(os.path.join(config.RESULTS_DIR, "residuals_best_model.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

print("\n--- Generating Evaluation Plots ---")
plt.figure(figsize=(6.3, 3.15))
plt.plot(optimum_results[0]['CurveTime'], optimum_results[0]['CurveCOF'], color='purple', label="Not esterified")
plt.plot(optimum_results[1]['CurveTime'], optimum_results[1]['CurveCOF'], color='orange', label="Esterified")
plt.ylim(config.PLOT_SETTINGS['cof_ylim'])
if optimum_results[0]['RunIn'] > 0:
    plt.axvline(x=optimum_results[0]['RunIn'], color='purple', linestyle='--', alpha=0.5, label='Run-in (Not esterified)')
if optimum_results[1]['RunIn'] > 0:
    plt.axvline(x=optimum_results[1]['RunIn'], color='orange', linestyle='--', alpha=0.5, label='Run-in (Esterified)')
plt.legend()
plt.savefig(os.path.join(config.RESULTS_DIR, "Optimum_comparison.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

print("\n--- Generating Trend Analysis Plot (Temperature vs COF) ---")
trend_temp_range = np.linspace(40, 120, 20)
trend_df_base = pd.DataFrame({'Time': 7200, 'Load': opt_load, 'Temperature': trend_temp_range, 'Concentration': opt_conc, 'Esterified': 0})
trend_df_ester = pd.DataFrame({'Time': 7200, 'Load': opt_load, 'Temperature': trend_temp_range, 'Concentration': opt_conc, 'Esterified': config.PLOT_ESTERIFIED_STATE})

trend_df_base = create_features(trend_df_base)[X_cols_raw]
trend_df_ester = create_features(trend_df_ester)[X_cols_raw]

trend_base_trans = global_vif.transform(global_interact.transform(trend_df_base))
trend_ester_trans = global_vif.transform(global_interact.transform(trend_df_ester))

preds_base = np.maximum(best_model_overall.predict(trend_base_trans), config.PREDICTION_LOWER_BOUND)[:, 0]
preds_ester = np.maximum(best_model_overall.predict(trend_ester_trans), config.PREDICTION_LOWER_BOUND)[:, 0]

fig, ax = plt.subplots(figsize=(6.3, 3.15))
plt.plot(trend_temp_range, preds_base, marker='o', color='purple', label='Not esterified (0)', linewidth=2.5)
plt.plot(trend_temp_range, preds_ester, marker='s', color='orange', label='Esterified Oil (1)', linewidth=2.5)
plt.fill_between(trend_temp_range, preds_base, preds_ester, color='grey', alpha=0.2, label='Ester Advantage')
plt.xlabel("Temperature [°C]")
plt.ylabel("Expected COF [-]")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
dynamic_descriptions["Temperature_Trend_Analysis.png"] = f"Temperature Trend Analysis (Load: {int(opt_load)}N, Conc: {opt_conc:.2f}%)."
plt.savefig(os.path.join(config.RESULTS_DIR, "Temperature_Trend_Analysis.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

# Load Trend Analysis
print("\n--- Generating Trend Analysis Plot (Load vs COF) ---")
trend_load_range = np.linspace(10, 200, 20)
trend_df_base_load = pd.DataFrame({'Time': 7200, 'Load': trend_load_range, 'Temperature': opt_temp, 'Concentration': opt_conc, 'Esterified': 0})
trend_df_ester_load = pd.DataFrame({'Time': 7200, 'Load': trend_load_range, 'Temperature': opt_temp, 'Concentration': opt_conc, 'Esterified': config.PLOT_ESTERIFIED_STATE})

trend_df_base_load = create_features(trend_df_base_load)[X_cols_raw]
trend_df_ester_load = create_features(trend_df_ester_load)[X_cols_raw]

trend_base_trans_load = global_vif.transform(global_interact.transform(trend_df_base_load))
trend_ester_trans_load = global_vif.transform(global_interact.transform(trend_df_ester_load))

preds_base_load = np.maximum(best_model_overall.predict(trend_base_trans_load), config.PREDICTION_LOWER_BOUND)[:, 0]
preds_ester_load = np.maximum(best_model_overall.predict(trend_ester_trans_load), config.PREDICTION_LOWER_BOUND)[:, 0]

fig, ax = plt.subplots(figsize=(6.3, 3.15))
plt.plot(trend_load_range, preds_base_load, marker='o', color='purple', label='Not esterified (0)', linewidth=2.5)
plt.plot(trend_load_range, preds_ester_load, marker='s', color='orange', label='Esterified Oil (1)', linewidth=2.5)
plt.fill_between(trend_load_range, preds_base_load, preds_ester_load, color='grey', alpha=0.2, label='Ester Advantage')
plt.xlabel("Load [N]")
plt.ylabel("Expected COF [-]")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
dynamic_descriptions["Load_Trend_Analysis.png"] = f"Load Trend Analysis (Temp: {int(opt_temp)}°C, Conc: {opt_conc:.2f}%)."
plt.savefig(os.path.join(config.RESULTS_DIR, "Load_Trend_Analysis.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

# Concentration Trend Analysis
print("\n--- Generating Trend Analysis Plot (Concentration vs COF) ---")
trend_conc_range = np.linspace(0, 0.6, 20)
trend_df_base_conc = pd.DataFrame({'Time': 7200, 'Load': opt_load, 'Temperature': opt_temp, 'Concentration': trend_conc_range, 'Esterified': 0})
trend_df_ester_conc = pd.DataFrame({'Time': 7200, 'Load': opt_load, 'Temperature': opt_temp, 'Concentration': trend_conc_range, 'Esterified': config.PLOT_ESTERIFIED_STATE})

trend_df_base_conc = create_features(trend_df_base_conc)[X_cols_raw]
trend_df_ester_conc = create_features(trend_df_ester_conc)[X_cols_raw]

trend_base_trans_conc = global_vif.transform(global_interact.transform(trend_df_base_conc))
trend_ester_trans_conc = global_vif.transform(global_interact.transform(trend_df_ester_conc))

preds_base_conc = np.maximum(best_model_overall.predict(trend_base_trans_conc), config.PREDICTION_LOWER_BOUND)[:, 0]
preds_ester_conc = np.maximum(best_model_overall.predict(trend_ester_trans_conc), config.PREDICTION_LOWER_BOUND)[:, 0]

fig, ax = plt.subplots(figsize=(6.3, 3.15))
plt.plot(trend_conc_range, preds_base_conc, marker='o', color='purple', label='Not esterified (0)', linewidth=2.5)
plt.plot(trend_conc_range, preds_ester_conc, marker='s', color='orange', label='Esterified Oil (1)', linewidth=2.5)
plt.fill_between(trend_conc_range, preds_base_conc, preds_ester_conc, color='grey', alpha=0.2, label='Ester Advantage')
plt.xlabel("Concentration [wt%]")
plt.ylabel("Expected COF [-]")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
dynamic_descriptions["Concentration_Trend_Analysis.png"] = f"Concentration Trend Analysis (Load: {int(opt_load)}N, Temp: {int(opt_temp)}°C)."
plt.savefig(os.path.join(config.RESULTS_DIR, "Concentration_Trend_Analysis.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
subset = full_df.groupby(['Load', 'Temperature', 'Concentration', 'Esterified'])['File_ID'].nunique().reset_index(name='Count')
ax.scatter(subset['Load'], subset['Temperature'], subset['Concentration'], c=subset['Esterified'], cmap='coolwarm', s=subset['Count']*100, alpha=0.6)
ax.set_xlabel('Load [N]', labelpad=10)
ax.set_ylabel('Temperature [°C]', labelpad=10)
ax.set_zlabel('Concentration [wt%]', labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=9)

custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)]
ax.legend(custom_lines, ['Not esterified (0)', 'Esterified (1)'], loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig(os.path.join(config.RESULTS_DIR, "3D_distribution_of_input_data.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

for ester_state_plot, suggestions in all_doe_suggestions.items():
    if suggestions.empty:
        continue

    state_str = "Esterified" if ester_state_plot == 1 else "NotEsterified"
    title_str = "Esterified" if ester_state_plot == 1 else "Not esterified"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(f'2D Projections of DoE Suggestions - {title_str}', fontsize=16)

    projections = [('Load', 'Temperature'), ('Load', 'Concentration'), ('Temperature', 'Concentration')]
    
    existing_data_subset = full_df[full_df['Esterified'] == ester_state_plot]

    for i, (x_ax, y_ax) in enumerate(projections):
        axes[i].scatter(existing_data_subset[x_ax], existing_data_subset[y_ax], c='grey', alpha=0.3, s=20, label='Existing Measurements')
        axes[i].scatter(suggestions[x_ax], suggestions[y_ax], c='red', marker='o', s=100, label='DoE Suggestions', edgecolors='black')
        
        axes[i].set_xlabel(f'{x_ax} [{config.NAME_MAPPING.get(x_ax, "wt%")}]')
        axes[i].set_ylabel(f'{y_ax} [{config.NAME_MAPPING.get(y_ax, "°C" if "Temp" in y_ax else "N")}]')
        axes[i].grid(True, linestyle='--', alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(config.RESULTS_DIR, f"DoE_2D_Projections_{state_str}.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
    plt.close()

L_grid, T_grid = np.meshgrid(np.linspace(10, 200, 100), np.linspace(40, 120, 100))
heatmap_input = create_features(pd.DataFrame({'Time': 7200, 'Load': L_grid.ravel(), 'Temperature': T_grid.ravel(), 'Concentration': opt_conc, 'Esterified': config.PLOT_ESTERIFIED_STATE}))[X_cols_raw]
heatmap_input_trans = global_vif.transform(global_interact.transform(heatmap_input))
cof_grid = np.maximum(best_model_overall.predict(heatmap_input_trans), config.PREDICTION_LOWER_BOUND)[:, 0].reshape(L_grid.shape)
plt.figure(figsize=(6.3, 3.15))
contourf_plot = plt.contourf(L_grid, T_grid, cof_grid, levels=100, cmap='plasma')
cbar = plt.colorbar(contourf_plot)
cbar.set_label('Coefficient of friction (COF) [-]')
contours_lines = plt.contour(L_grid, T_grid, cof_grid, levels=10, colors='black', alpha=0.5)
plt.clabel(contours_lines, inline=True, fontsize=8, fmt='%.3f')
plt.xlabel("Load [N]")
plt.ylabel("Temperature [°C]")
plt.savefig(os.path.join(config.RESULTS_DIR, "COF_heatmap.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

plt.figure(figsize=(10, 8))
corr_cols = ['Time', 'Load', 'Temperature', 'Concentration', 'Esterified', 'COF', 'Friction absolute integral']
corr_labels = ['Time', 'Load', 'Temperature', 'Concentration', 'Esterified', 'COF', 'Friction Absolute Integral']
corr_matrix = full_df[corr_cols].corr()
im = plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)
plt.colorbar(im)
plt.grid(False)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black", fontsize=10)
plt.xticks(range(len(corr_labels)), corr_labels, rotation=45, ha='right', fontsize=12)
plt.yticks(range(len(corr_labels)), corr_labels, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, "Correlation_matrix.png"), dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight', pad_inches=0.1)
plt.close()

plot_learning_curve(best_model_overall, X, Y, cv=GroupShuffleSplit(n_splits=config.CV_SPLITS, test_size=0.2, random_state=config.RANDOM_SEED), results_dir=config.RESULTS_DIR, groups=groups, num_files=len(np.unique(groups)))
dynamic_descriptions["Learning_Curve.png"] = f"Learning curve ({best_model_name})."

html_path = os.path.join(config.RESULTS_DIR, "Eredmenyek_Riport.html")
desc_df = full_df[['Time', 'Load', 'Temperature', 'Concentration', 'Esterified', 'COF', 'Friction absolute integral']].describe()
desc_df.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50% (Median)', '75%', 'Max']
desc_df.rename(columns=config.NAME_MAPPING, inplace=True)

timing_stats = {
    'total': format_time(time.time() - script_start), 
    'loading': format_time(loading_duration), 
    'shap': format_time(shap_duration) if shap_duration is not None else "N/A", 
    'doe': format_time(total_doe_duration)
}
html_content = generate_html_report(results, xlsx_files, full_df, desc_df, html_path, config.RESULTS_DIR, doe_suggestions_combined, optimum_results, shap_analysis_text, timing_stats, dynamic_descriptions)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

excel_path = os.path.join(config.RESULTS_DIR, "Results_Tables.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    pd.DataFrame(results)[['Name', 'R2_Train', 'R2_Test', 'R2_CV', 'RMSE_Train', 'RMSE_Test', 'MAE_Test', 'Tuning_Training_Time', 'Pred_Time_ms']].to_excel(writer, sheet_name='Model_Metrics', index=False)
    opt_data = [{'Type': 'Esterified' if s == 1 else 'Not esterified', **r} for s, r in optimum_results.items()]
    pd.DataFrame(opt_data).drop(columns=['CurveTime', 'CurveCOF']).to_excel(writer, sheet_name='Optimums', index=False)
    doe_suggestions_combined.to_excel(writer, sheet_name='DoE_Suggestions', index=False)

joblib.dump(best_model_overall, os.path.join(config.RESULTS_DIR, f"Best_Model_{best_model_name.replace(' ', '_')}.pkl"))
print("\nPipeline completed successfully! Opening HTML report...")
webbrowser.open(html_path)
