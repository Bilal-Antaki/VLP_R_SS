from src.data.data_loader import load_cir_data, scale_and_sequenceap
from torch.utils.data import DataLoader, TensorDataset
from src.config import MODEL_CONFIG, DATA_CONFIG
from src.models.model_registry import get_model
import torch.nn as nn
import numpy as np
import random
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def train_lstm_on_all(processed_dir):
    """Train LSTM model on all available data"""
    print("\nTraining LSTM model...")
    
    # Get LSTM-specific config
    lstm_config = MODEL_CONFIG['lstm']
    
    # Set random seed
    random_seed = 42
    print(f"Using random seed: {random_seed}")
    
    # Set random seeds
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Load data
    df = load_cir_data(processed_dir, filter_keyword='FCPR-D1')
    print(f"Loaded {len(df)} data points from FCPR-D1")
    
    # Use trajectory-based splitting instead of manual split
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, x_scaler, y_scaler = scale_and_sequenceap(
        df, seq_len=MODEL_CONFIG['sequence_length'], test_size=0.2
    )
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train_seq, y_train_seq), 
        batch_size=lstm_config['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_seq, y_val_seq), 
        batch_size=lstm_config['batch_size']
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model("lstm", input_dim=2, hidden_dim=lstm_config['hidden_size'], num_layers=lstm_config['num_layers'], dropout=lstm_config['dropout'])
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lstm_config['learning_rate'],
        weight_decay=1e-5
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = lstm_config['patience']
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Gradient clipping threshold
    grad_clip = 1.0
    
    for epoch in range(lstm_config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'results/models/lstm_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {current_lr:.6f}")
    
    # Load best model for evaluation
    checkpoint = torch.load('results/models/lstm_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Make predictions on validation set
    with torch.no_grad():
        y_pred = model(X_val_seq.to(device)).cpu().numpy()
    
    # Inverse transform predictions
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_val_orig = y_scaler.inverse_transform(y_val_seq.numpy().reshape(-1, 1)).flatten()
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_val_orig - y_pred) ** 2))
    
    print("\nFinal Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"Target range: [{y_val_orig.min():.2f}, {y_val_orig.max():.2f}]")
    print(f"Prediction std: {y_pred.std():.4f}")
    print(f"Target std: {y_val_orig.std():.4f}")
    
    return {
        'rmse': rmse,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'r_actual': y_val_orig.flatten(),
        'r_pred': y_pred.flatten()
    }

if __name__ == "__main__":
    train_lstm_on_all(DATA_CONFIG['processed_dir'])