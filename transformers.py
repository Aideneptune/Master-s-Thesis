import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

class InteractionFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Egyedi transzformátor interakciós jellemzők generálásához tanult átlagokkal."""
    def __init__(self):
        self.l_mean_ = 0.0
        self.t_mean_ = 0.0
        self.c_mean_ = 0.0
        self.e_mean_ = 0.0
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.l_mean_ = X['Load'].mean()
            self.t_mean_ = X['Temperature'].mean()
            self.c_mean_ = X['Concentration'].mean()
            self.e_mean_ = X['Esterified'].mean()
            self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        
        # Interakciók generálása a fit során mentett átlagokkal
        X_df['Load_x_Temp'] = (X_df['Load'] - self.l_mean_) * (X_df['Temperature'] - self.t_mean_)
        X_df['Load_x_Conc'] = (X_df['Load'] - self.l_mean_) * (X_df['Concentration'] - self.c_mean_)
        X_df['Temp_x_Conc'] = (X_df['Temperature'] - self.t_mean_) * (X_df['Concentration'] - self.c_mean_)
        X_df['Ester_x_Temp'] = (X_df['Esterified'] - self.e_mean_) * (X_df['Temperature'] - self.t_mean_)
        X_df['Ester_x_Load'] = (X_df['Esterified'] - self.e_mean_) * (X_df['Load'] - self.l_mean_)
        X_df['Ester_x_Conc'] = (X_df['Esterified'] - self.e_mean_) * (X_df['Concentration'] - self.c_mean_)
        
        return X_df

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        new_features = ['Load_x_Temp', 'Load_x_Conc', 'Temp_x_Conc', 'Ester_x_Temp', 'Ester_x_Load', 'Ester_x_Conc']
        return np.array(list(input_features) + new_features)

class VIFSelector(BaseEstimator, TransformerMixin):
    """Egyedi transzformátor VIF-alapú jellemző-kiválasztáshoz."""
    def __init__(self, threshold=10.0, sample_size=5000, protected_cols=None):
        self.threshold = threshold
        self.sample_size = sample_size
        self.protected_cols = protected_cols if protected_cols else [
            'Load', 'Temperature', 'Concentration', 'Esterified', 'Time', 'Log_Time', 'Time_Squared'
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