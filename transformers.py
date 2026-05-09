import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class PandasStandardScaler(BaseEstimator, TransformerMixin):
    """Custom StandardScaler that returns a DataFrame to preserve column names."""
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

class InteractionFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for generating interaction features using learned means."""
    def __init__(self, load_col='Load', temp_col='Temperature', conc_col='Concentration', ester_col='Esterified'):
        self.load_col = load_col
        self.temp_col = temp_col
        self.conc_col = conc_col
        self.ester_col = ester_col
        self.l_mean_ = 0.0
        self.t_mean_ = 0.0
        self.c_mean_ = 0.0
        self.e_mean_ = 0.0
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.l_mean_ = X[self.load_col].mean()
            self.t_mean_ = X[self.temp_col].mean()
            self.c_mean_ = X[self.conc_col].mean()
            self.e_mean_ = X[self.ester_col].mean()
            self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Generate interactions with fitted means
        X_df[f'{self.load_col}_x_{self.temp_col}'] = (X_df[self.load_col] - self.l_mean_) * (X_df[self.temp_col] - self.t_mean_)
        X_df[f'{self.load_col}_x_{self.conc_col}'] = (X_df[self.load_col] - self.l_mean_) * (X_df[self.conc_col] - self.c_mean_)
        X_df[f'{self.temp_col}_x_{self.conc_col}'] = (X_df[self.temp_col] - self.t_mean_) * (X_df[self.conc_col] - self.c_mean_)
        X_df[f'{self.ester_col}_x_{self.temp_col}'] = (X_df[self.ester_col] - self.e_mean_) * (X_df[self.temp_col] - self.t_mean_)
        X_df[f'{self.ester_col}_x_{self.load_col}'] = (X_df[self.ester_col] - self.e_mean_) * (X_df[self.load_col] - self.l_mean_)
        X_df[f'{self.ester_col}_x_{self.conc_col}'] = (X_df[self.ester_col] - self.e_mean_) * (X_df[self.conc_col] - self.c_mean_)
        
        # Physics-based ratios (avoiding division by zero)
        X_df[f'{self.load_col}_div_{self.temp_col}'] = X_df[self.load_col] / (X_df[self.temp_col] + 1e-6)
        X_df[f'{self.conc_col}_div_{self.load_col}'] = X_df[self.conc_col] / (X_df[self.load_col] + 1e-6)
        X_df[f'{self.temp_col}_div_{self.conc_col}'] = X_df[self.temp_col] / (X_df[self.conc_col] + 1e-6)
        
        return X_df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        new_features = [
            f'{self.load_col}_x_{self.temp_col}', f'{self.load_col}_x_{self.conc_col}',
            f'{self.temp_col}_x_{self.conc_col}', f'{self.ester_col}_x_{self.temp_col}',
            f'{self.ester_col}_x_{self.load_col}', f'{self.ester_col}_x_{self.conc_col}',
            f'{self.load_col}_div_{self.temp_col}', f'{self.conc_col}_div_{self.load_col}',
            f'{self.temp_col}_div_{self.conc_col}'
        ]
        return np.array(list(input_features) + new_features)

class VIFSelector(BaseEstimator, TransformerMixin):
    """Custom transformer for VIF-based feature selection."""
    def __init__(self, threshold=10.0, sample_size=5000, protected_cols=None):
        self.threshold = threshold
        self.sample_size = sample_size
        self.protected_cols = protected_cols if protected_cols else [
            'Load', 'Temperature', 'Concentration', 'Esterified', 'Time', 'Log_Time', 'Time_Squared',
        ]
        self.selected_features_ = []
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=X.columns) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = X_df.columns.tolist()

        if len(X_df) > self.sample_size:
            X_sample = X_df.sample(n=self.sample_size, random_state=42)
        else:
            X_sample = X_df
          
        variables = list(X_df.columns)
      
        while True:
            if len(variables) < 2:
                break
            vif_data = []
            for var in variables:
                X_temp = X_sample[variables].drop(columns=[var])
                y_temp = X_sample[var]
                model = LinearRegression()
                model.fit(X_temp, y_temp)
                r_squared = model.score(X_temp, y_temp)
                vif = 1 / (1 - r_squared) if r_squared < 1.0 else float('inf')
                vif_data.append((var, vif))
            
            vif_data.sort(key=lambda x: x[1], reverse=True)
            best_drop = next((item for item in vif_data if item[1] > self.threshold and item[0] not in self.protected_cols), None)
            
            if best_drop:
                variables.remove(best_drop[0])
            else:
                break
        self.selected_features_ = variables
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X.copy()
        return X_df[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_features_)
