import os
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from transformers import PandasStandardScaler

# --- ELÉRÉSI UTAK ---
# A projekt fő könyvtára
PROJECT_ROOT = r"K:\Thesis"
# Az adatok forrása
BASE_PATH = os.path.join(os.getcwd(), "Test_Data")
# Az eredmények mentési helye
RESULTS_DIR = os.path.join(os.getcwd(), "Results")
# A gyorsítótár (cache) mentési helye
CACHE_DIR = os.path.join(PROJECT_ROOT, "Cache")

# --- ALAPVETŐ BEÁLLÍTÁSOK ---
USE_CACHE = True
RANDOM_SEED = 42
DOWNSAMPLING_RATE = 10
ROLLING_WINDOW_SIZE = 20
PREDICTION_LOWER_BOUND = 0.001
PLOT_ESTERIFIED_STATE = 1  # 1 = Esterified, 0 = Base Oil

# --- FIZIKAI ÁLLANDÓK (HERTZ-FESZÜLTSÉG) ---
E_MODULUS = 210000.0  # Rugalmassági modulusz [MPa]
POISSON_RATIO = 0.3   # Poisson-tényező [-]
BALL_RADIUS = 5.0     # Golyó sugara [mm]

# --- ÁBRÁK GLOBÁLIS BEÁLLÍTÁSAI ---
PLOT_SETTINGS = {
    'dpi': 300,
    'cof_ylim': (0.0, 0.25)
}

# --- DOE BEÁLLÍTÁSOK ---
UNCERTAINTY_WEIGHT = 0.5
SPARSITY_WEIGHT = 0.5

# --- AKADÉMIAI ÁBRAFORMÁZÁS BEÁLLÍTÁSA ---
def set_academic_plot_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Tahoma'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.labelcolor': 'black',
        'xtick.labelsize': 11,
        'xtick.color': 'black',
        'ytick.labelsize': 11,
        'ytick.color': 'black',
        'legend.fontsize': 11,
        'figure.figsize': (6.3, 3.15),
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.grid': False,
        'grid.color': 'black',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.minor.size': 1.5,
        'ytick.minor.size': 1.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.5,
        'legend.frameon': False,
        'axes.prop_cycle': cycler('color', ['k', 'r', 'b', 'g']) + cycler('ls', ['-', '--', ':', '-.'])
    })

# --- MODELL TANÍTÁSI PARAMÉTEREK ---
TEST_SIZE = 0.2
CV_SPLITS = 5
SEARCH_ITERATIONS = 50

# --- OSZLOPNEVEK ---
FEATURE_COLS = ['Time', 'Log_Time', 'Time_Squared', 'Load', 'Temperature', 'Concentration', 'Esterified', 'Hertz_Stress_MPa']
TARGET_COLS = ['COF', 'Friction absolute integral']

# --- NÉV-LEKÉPEZŐ SZÓTÁR (HTML és Plot feliratokhoz) ---
NAME_MAPPING = {
    'Log_Time': 'Logarithmic time',
    'Time_Squared': 'Squared time',
    'Load_x_Temperature': 'Load × Temperature',
    'Load_x_Concentration': 'Load × Concentration',
    'Temperature_x_Concentration': 'Temperature × Concentration',
    'Esterified_x_Temperature': 'Esterified × Temperature',
    'Esterified_x_Load': 'Esterified × Load',
    'Esterified_x_Concentration': 'Esterified × Concentration',
    'Load_div_Temperature': 'Load / Temperature',
    'Concentration_div_Load': 'Concentration / Load',
    'Temperature_div_Concentration': 'Temperature / Concentration',
    'Load': 'Load [N]',
    'Temperature': 'Temperature [°C]',
    'COF': 'Coefficient of Friction (COF) [-]',
    'Friction absolute integral': 'Friction Absolute Integral (FAI) [-]',
    'Sample_Weight': 'Sample weight',
    'R2_Header': 'R² (Train / Test / CV)',
    'R2_Split_Header': 'R² (COF / FAI)',
    'RMSE_Header': 'RMSE (Train / Test)',
    'RMSE_Split_Header': 'RMSE (COF / FAI)',
    'MAE_Header': 'MAE (Test)',
    'Optimum_Header': 'Optimum (Conc. / Load / Temp.)',
    'Pred_Header': 'Predicted value at the last 5 minutes (COF / FAI)',
    'Hertz_Stress_MPa': 'Max. Hertzian Stress [MPa]',
    'Time_Header': 'Tuning & Training Time / Prediction Time'
}

# --- HTML RIPORT LEÍRÁSOK ---
IMAGE_DESCRIPTIONS = {
    "Effect_of_noise_filtering.png": "Comparison of raw measurement data and the smoothed curve using a rolling mean filter on the first data file.",
    "Effect_of_noise_filtering_2.png": "Comparison of raw measurement data and the smoothed curve using a rolling mean filter on a different data file for verification.",
    "3D_distribution_of_input_data.png": "Distribution of measurement points in the Load, Temperature, and Concentration space.",
    "DoE_2D_Projections.png": "2D Projections of Existing Data and DoE Suggestions, showing where new measurements are proposed.",
    "COF_heatmap.png": "Estimated static friction coefficient (COF) as a function of Load and Temperature.",
    "Correlation_matrix.png": "Strength of linear relationships between variables (Pearson correlation).",
    "Model_comparison.png": "Comparison of model accuracy (R2) and error (RMSE) on the test dataset.",
    "Neural_network_structure.png": "Structure of the trained neural network (MLP).",
    "Learning_Curve.png": "Learning curve of the best model showing accuracy as a function of data size.",
    "SHAP_feature_impact.png": "SHAP plot showing how each feature pushes the prediction.",
    "SHAP_Dependence_Load.png": "SHAP Dependence Plot for Load, colored by Esterified state.",
    "SHAP_Dependence_Temperature.png": "SHAP Dependence Plot for Temperature, colored by Esterified state.",
    "Optimum_comparison.png": "Direct comparison of optimum curves for base oil and esterified oil.",
    "Pareto_Optimization.png": "Pareto front showing trade-offs between COF and FAI.",
    "Feature_importance.png": "Feature importance ranking for the best performing model.",
    "residuals_best_model.png": "Residual plot of the best performing model showing the distribution of prediction errors.",
    "Temperature_Trend_Analysis.png": "Effect of Temperature on the expected COF, comparing Esterified Oil with Base Oil.",
    "Load_Trend_Analysis.png": "Effect of Load on the expected COF, comparing Esterified Oil with Base Oil.",
    "Concentration_Trend_Analysis.png": "Effect of Concentration on the expected COF, comparing Esterified Oil with Base Oil."
}

# --- MODELLEK ÉS HIPERPARAMÉTER HÁLÓK (GRID) ---
models_config = {
    "XGBoost": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('xgb', MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=RANDOM_SEED)))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__xgb__estimator__n_estimators": [100, 200],
            "regressor__xgb__estimator__learning_rate": [0.01, 0.05, 0.1],
            "regressor__xgb__estimator__max_depth": [3, 5, 7],
            "regressor__xgb__estimator__reg_alpha": [5, 10, 20],
            "regressor__xgb__estimator__reg_lambda": [10, 20, 50]
        }
    },
    "Neural Network (MLP)": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('mlp', MLPRegressor(random_state=RANDOM_SEED, max_iter=500, early_stopping=True))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__mlp__hidden_layer_sizes": [(30,), (50,)],
            "regressor__mlp__alpha": [5.0, 10.0, 20.0],
            "regressor__mlp__activation": ['relu', 'tanh'],
            "regressor__mlp__learning_rate_init": [0.001, 0.005]
        }
    },
    "Random Forest": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('rf', RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__rf__n_estimators": [100, 200],
            "regressor__rf__max_depth": [5, 8, 12],
            "regressor__rf__min_samples_leaf": [10, 20, 50]
        }
    },
    "LightGBM": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('lgbm', MultiOutputRegressor(LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__lgbm__estimator__n_estimators": [100, 200],
            "regressor__lgbm__estimator__learning_rate": [0.01, 0.05, 0.1],
            "regressor__lgbm__estimator__max_depth": [4, 6, 8],
            "regressor__lgbm__estimator__reg_lambda": [10, 20, 50],
            "regressor__lgbm__estimator__min_child_samples": [20, 50]
        }
    },
    "CatBoost": {
        "model": TransformedTargetRegressor(
            regressor=Pipeline([
                ('scaler', PandasStandardScaler()),
                ('cat', MultiOutputRegressor(CatBoostRegressor(random_state=RANDOM_SEED, verbose=0, allow_writing_files=False)))
            ]),
            func=np.log,
            inverse_func=np.exp
        ),
        "params": {
            "regressor__cat__estimator__iterations": [100, 250],
            "regressor__cat__estimator__learning_rate": [0.03, 0.08],
            "regressor__cat__estimator__depth": [4, 6, 8],
            "regressor__cat__estimator__l2_leaf_reg": [20, 50, 100]
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
            "regressor__knn__n_neighbors": [5, 10, 15],
            "regressor__knn__weights": ['uniform']
        }
    },
    "Ridge Regression": {
        "model": Pipeline([
            ('scaler', PandasStandardScaler()),
            ('ridge', Ridge())
        ]),
        "params": {
            "ridge__alpha": [100.0, 500.0, 1000.0]
        }
    }
}
