
from src.data.data_loader import load_cir_data, scale_and_sequenceap
from src.config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import random
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def train_gru_on_all(processed_dir: str, batch_size: int = None, epochs: int = None, lr: float = None):
    """
    Train GRU model for position estimation
    """
    # Get GRU-specific config
    gru_config = MODEL_CONFIG['gru']
    
    # Use provided parameters or fall back to GRU-specific config values
    batch_size = batch_size if batch_size is not None else gru_config['batch_size']
    epochs = epochs if epochs is not None else gru_config['epochs']
    lr = lr if lr is not None else gru_config['learning_rate']
    
    # Set random seed
    random_seed = TRAINING_CONFIG['random_seed']
    print(f"Using random seed: {random_seed}")
    
    # Set random seeds
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Use sequence length from config
    seq_len = MODEL_CONFIG['sequence_length']
    
    # Load data using dataset from config
    df = load_cir_data(processed_dir, filter_keyword=DATA_CONFIG['datasets'][0])
    
    # Use trajectory-based splitting instead of random train_test_split
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, x_scaler, y_scaler = scale_and_sequenceap(
        df, seq_len=seq_len, test_size=TRAINING_CONFIG['validation_split']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train_seq, y_train_seq), 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_seq, y_val_seq), 
        batch_size=batch_size,
        drop_last=False
    )
    
    # Create model with GRU-specific config values
    from src.models.gru import build_gru_model
    model = build_gru_model(
        input_dim=2,
        hidden_dim=gru_config['hidden_size'],
        num_layers=gru_config['num_layers'],
        dropout=gru_config['dropout']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    train_loss_hist, val_loss_hist = [], []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = gru_config['patience']
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
                val_batches += 1
        val_loss /= val_batches
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Generate predictions on validation set
    model.eval()
    all_val_preds = []
    all_val_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_val_preds.extend(preds)
            all_val_targets.extend(y_batch.numpy())

    # Convert to arrays and inverse transform
    val_preds_scaled = np.array(all_val_preds)
    val_targets_scaled = np.array(all_val_targets)

    val_preds = y_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).flatten()
    val_targets = y_scaler.inverse_transform(val_targets_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((val_targets - val_preds) ** 2))

    print(f"\nFinal Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Prediction range: [{val_preds.min():.2f}, {val_preds.max():.2f}]")
    print(f"Target range: [{val_targets.min():.2f}, {val_targets.max():.2f}]")

    return {
        'r_actual': val_targets.tolist(),
        'r_pred': val_preds.tolist(),
        'train_loss': train_loss_hist,
        'val_loss': val_loss_hist,
        'rmse': rmse,
        'original_df_size': len(df),
        'sequence_size': len(val_targets),
        'seq_len': seq_len
    }
