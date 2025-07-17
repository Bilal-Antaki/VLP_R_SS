import torch
import torch.nn as nn
from ..config import MODEL_CONFIG

class CNNRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=None, num_layers=None, dropout=None, 
                 kernel_size=None, num_filters=None):
        super(CNNRegressor, self).__init__()
        
        cnn_config = MODEL_CONFIG['cnn']
        self.hidden_size = hidden_size if hidden_size is not None else cnn_config['hidden_size']
        self.num_layers = num_layers if num_layers is not None else cnn_config['num_layers']
        self.dropout = dropout if dropout is not None else cnn_config['dropout']
        self.kernel_size = kernel_size if kernel_size is not None else cnn_config['kernel_size']
        self.num_filters = num_filters if num_filters is not None else cnn_config['num_filters']
        
        # Stack of Conv1d layers
        layers = []
        in_channels = input_size
        for i in range(self.num_layers):
            layers.append(nn.Conv1d(in_channels, self.num_filters, kernel_size=self.kernel_size, padding=self.kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_channels = self.num_filters
        self.conv = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Simple regression head
        self.fc = nn.Linear(self.num_filters, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: [batch, seq_len, input_dim] -> [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.global_pool(x)  # [batch, num_filters, 1]
        x = x.squeeze(-1)        # [batch, num_filters]
        output = self.fc(x)      # [batch, 1]
        return output.squeeze(-1)  # [batch]

def build_cnn_model(**kwargs):
    """Factory function for CNN model"""
    return CNNRegressor(**kwargs)
