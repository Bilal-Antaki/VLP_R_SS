import torch
import torch.nn as nn
from ..config import MODEL_CONFIG

class GRURegressor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=None, num_layers=None, dropout=None):
        super(GRURegressor, self).__init__()
        
        # Use provided parameters or fall back to GRU-specific config values
        gru_config = MODEL_CONFIG['gru']
        self.hidden_dim = hidden_dim if hidden_dim is not None else gru_config['hidden_size']
        self.num_layers = num_layers if num_layers is not None else gru_config['num_layers']
        self.dropout = dropout if dropout is not None else gru_config['dropout']
        
        # GRU layer
        self.gru = nn.GRU(
            input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        gru_out, _ = self.gru(x)
        # Take the last timestep output
        last_output = gru_out[:, -1]  # [batch, hidden_dim]
        output = self.fc(last_output)
        return output.squeeze(-1)  # [batch]

def build_gru_model(**kwargs):
    """Factory function for basic GRU model"""
    return GRURegressor(**kwargs)