import os

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
ROLLING_WINDOW_SIZE = 50
PREDICTION_LOWER_BOUND = 0.001
PLOT_ESTERIFIED_STATE = 1  # 1 = Esterified, 0 = Base Oil

# --- MODELL TANÍTÁSI PARAMÉTEREK ---
TEST_SIZE = 0.2
CV_SPLITS = 5
SEARCH_ITERATIONS = 50

# --- OSZLOPNEVEK ---
FEATURE_COLS = ['Time', 'Log_Time', 'Time_Squared', 'Load', 'Temperature', 'Concentration', 'Esterified']
TARGET_COLS = ['COF', 'Friction absolute integral']

# --- NÉV-LEKÉPEZŐ SZÓTÁR (HTML és Plot feliratokhoz) ---
NAME_MAPPING = {
    'Log_Time': 'Logarithmic time',
    'Time_Squared': 'Squared time',
    'Load_x_Temp': 'Load × Temperature',
    'Load_x_Conc': 'Load × Concentration',
    'Temp_x_Conc': 'Temperature × Concentration',
    'Ester_x_Temp': 'Esterified × Temperature',
    'Ester_x_Load': 'Esterified × Load',
    'Ester_x_Conc': 'Esterified × Concentration',
    'Sample_Weight': 'Sample weight',
    'R2_Header': 'R² (Train / Test / CV)',
    'R2_Split_Header': 'R² (COF / FAI)',
    'RMSE_Header': 'RMSE (Train / Test)',
    'RMSE_Split_Header': 'RMSE (COF / FAI)',
    'MAE_Header': 'MAE (Test)',
    'Optimum_Header': 'Optimum (Conc. / Load / Temp.)',
    'Pred_Header': 'Predicted value at the last 5 minutes (COF / FAI)',
    'Time_Header': 'Tuning & Training Time / Prediction Time'
}

# --- HTML RIPORT LEÍRÁSOK ---
IMAGE_DESCRIPTIONS = {
    "Effect_of_noise_filtering.png": "Comparison of raw measurement data and the smoothed curve using rolling mean.",
    "3D_distribution_of_input_data.png": "Distribution of measurement points in the Load, Temperature, and Concentration space.",
    "COF_heatmap.png": "Estimated static friction coefficient (COF) as a function of Load and Temperature.",
    "Correlation_matrix.png": "Strength of linear relationships between variables (Pearson correlation).",
    "Model_comparison.png": "Comparison of model accuracy (R2) and error (RMSE) on the test dataset.",
    "Neural_network_structure.png": "Structure of the trained neural network (MLP).",
    "Learning_Curve.png": "Learning curve of the best model showing accuracy as a function of data size.",
    "SHAP_feature_impact.png": "SHAP plot showing how each feature pushes the prediction.",
    "SHAP_Dependence_Load.png": "SHAP Dependence Plot for Load, colored by Esterified state.",
    "SHAP_Dependence_Temperature.png": "SHAP Dependence Plot for Temperature, colored by Esterified state.",
    "Optimum_comparison.png": "Direct comparison of optimum curves for base oil and esterified oil.",
    "Pareto_Optimization.png": "Pareto front showing trade-offs between COF and FAI."
}