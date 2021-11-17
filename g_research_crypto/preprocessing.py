import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def training_window_sliding(X, y, window):
    X_windows = []
    y_windows = []
    N = len(X)
    for i in range(window, N):
        X_t = X[i-window: i+1].copy()
        y_t = y[i].copy()
        if np.isnan(y_t).any() == False:
            X_windows.append(X_t)
            y_windows.append(y_t)
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    return X_windows, y_windows

def convert_df_to_X(df, window, base_features, is_train=False):
    X = df[base_features].values
    y = df[['Target']].values
    
    if is_train:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_X.fit(X)
        scaler_y.fit(y)
        
        return X, y, scaler_X, scaler_y
    else:
        return X, y
        
    
def get_valid_dataframe(df, window, start_date, end_date=None):
    if start_date is not None:
        if end_date is not None:
            min_idx = df[(df['datetime'] >= start_date)&(df['datetime'] <end_date)].index.min()
            max_idx = df[(df['datetime'] >= start_date)&(df['datetime'] <end_date)].index.max()
        else:
            min_idx = df[(df['datetime'] >= start_date)].index.min()
            max_idx = df[(df['datetime'] >= start_date)].index.max()
    else:
        min_idx = df[(df['datetime'] <= end_date)].index.min()
        max_idx = df[(df['datetime'] <= end_date)].index.max()
        
    df_valid = pd.DataFrame(df[(df.index >= min_idx - window)&(df.index<=max_idx)])
    return df_valid