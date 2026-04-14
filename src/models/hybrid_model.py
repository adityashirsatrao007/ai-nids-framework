import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNLSTMModel, self).__init__()
        
        # CNN Part
        # Input shape: (Batch, 1, Input_Dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate dimension after CNN/Pooling
        # Example: input 32 -> conv1 32 -> pool1 16 -> conv2 16 -> pool2 8
        self.flatten_dim = 128 * (input_dim // 4)
        
        # LSTM Part
        # We'll treat the output of CNN as a sequence or just pass the flattened output
        # Standard CNN-LSTM often feeds CNN features into LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        
        # Fully Connected
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (Batch, Input_Dim) -> (Batch, 1, Input_Dim)
        x = x.unsqueeze(1)
        
        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Reshape for LSTM: (Batch, Channels, Seq) -> (Batch, Seq, Channels)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last time step
        x = lstm_out[:, -1, :]
        
        # FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_ensemble_model(rf_params=None, xgb_params=None):
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from xgboost import XGBClassifier
    
    rf = RandomForestClassifier(**(rf_params or {'n_estimators': 100, 'random_state': 42}))
    xgb = XGBClassifier(**(xgb_params or {'n_estimators': 100, 'random_state': 42}))
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb)],
        voting='hard'
    )
    return ensemble
