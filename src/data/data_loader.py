import pandas as pd
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from ..config import DATA_CONFIG
from sklearn.model_selection import train_test_split

def load_cir_data(processed_dir: str, filter_keyword: str = None) -> pd.DataFrame:
    all_data = []
    for file in os.listdir(processed_dir):
        if file.endswith('_CIR.csv') and (filter_keyword is None or filter_keyword in file):
            filepath = os.path.join(processed_dir, file)
            df = pd.read_csv(filepath)
            df['source_file'] = file
            all_data.append(df)
    if not all_data:
        raise FileNotFoundError(f"No matching CIR files in {processed_dir}")
    return pd.concat(all_data, ignore_index=True)

def extract_features_and_target(df: pd.DataFrame, features=['PL', 'RMS'], target='r'):
    X = df[features]
    y = df[target]
    return X, y

def sequence_split(X, y, seq_len):
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len + 1):  # Fixed off-by-one error
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len-1])  # Predict current timestep, not future
    return np.array(X_seq), np.array(y_seq)

def scale_and_sequence(df, seq_len=10, features=['PL', 'RMS'], target='r'):
    """Improved scaling and sequencing for LSTM"""
    
    # Sort by position to ensure temporal consistency
    df_sorted = df.copy()
    
    X = df_sorted[features].values
    y = df_sorted[target].values
    
    # Use StandardScaler instead of MinMaxScaler for better gradient flow
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    # Fit scalers
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # Check for data issues
    print(f"Original y (r) range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Data shape: X={X_scaled.shape}, y={y_scaled.shape}")
    print(f"Scaled X range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    print(f"Scaled y range: [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")
    
    # Create sequences
    X_seq, y_seq = sequence_split(X_scaled, y_scaled, seq_len)
    
    print(f"Sequence shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    
    return (
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(y_seq, dtype=torch.float32),
        x_scaler,
        y_scaler
    )

def scale_and_sequenceap(df, seq_len=10, test_size=0.2):
    """Scale features and create sequences with trajectory-based split"""
    X, y = extract_features_and_target(df)
    X = X.values
    y = y.values
    # Calculate number of complete trajectories
    n_samples = len(df)
    trajectory_length = seq_len  # Each trajectory is seq_len steps
    n_trajectories = n_samples // trajectory_length
    # If data doesn't divide evenly, truncate to complete trajectories
    n_samples = n_trajectories * trajectory_length
    X = X[:n_samples]
    y = y[:n_samples]
    # Create trajectory indices
    trajectory_indices = np.arange(n_trajectories)
    # Randomly select test trajectories
    n_test_trajectories = int(n_trajectories * test_size)
    test_trajectories = np.random.choice(trajectory_indices, n_test_trajectories, replace=False)
    train_trajectories = np.array([i for i in trajectory_indices if i not in test_trajectories])
    # Create masks for train and test samples
    train_mask = np.zeros(n_samples, dtype=bool)
    test_mask = np.zeros(n_samples, dtype=bool)
    for traj_idx in train_trajectories:
        start = traj_idx * trajectory_length
        end = start + trajectory_length
        train_mask[start:end] = True
    for traj_idx in test_trajectories:
        start = traj_idx * trajectory_length
        end = start + trajectory_length
        test_mask[start:end] = True
    # Split data by trajectories
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    # Scale based on training data only
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_test_scaled = x_scaler.transform(X_test)  # Use training statistics
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    # Create sequences (each trajectory creates 1 sequence since seq_len = trajectory_length)
    X_train_seq, y_train_seq = sequence_split(X_train_scaled, y_train_scaled, seq_len)
    X_test_seq, y_test_seq = sequence_split(X_test_scaled, y_test_scaled, seq_len)
    print(f"Total trajectories: {n_trajectories}")
    print(f"Train trajectories: {len(train_trajectories)} ({X_train_seq.shape[0]} sequences)")
    print(f"Test trajectories: {len(test_trajectories)} ({X_test_seq.shape[0]} sequences)")
    print(f"Selected test trajectories: {sorted(test_trajectories)}")
    return (
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(y_train_seq, dtype=torch.float32),
        torch.tensor(X_test_seq, dtype=torch.float32),
        torch.tensor(y_test_seq, dtype=torch.float32),
        x_scaler,
        y_scaler
    ) 

def load_data(test_size=0.2, random_state=42):
    """
    Loads the main input file, splits into train/test, and returns X_train, X_test, y_train, y_test.
    Uses features ['PL', 'RMS'] and target 'r' as in config.
    """
    df = pd.read_csv(DATA_CONFIG['input_file'])
    X = df[['PL', 'RMS']]
    y = df[DATA_CONFIG['target_column']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test 