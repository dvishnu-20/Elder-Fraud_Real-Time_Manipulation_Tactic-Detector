import torch
import torch.nn as nn
from typing import Literal

class CausalLSTMRiskModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        pooling_head: Literal["final", "masked_mean"] = "final"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling_head = pooling_head
        
        # Strictly unidirectional (causal) LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Classification head outputting single logit for fraud probability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, features, mask, lengths=None):
        # features: (batch_size, seq_len, input_dim)
        # mask: (batch_size, seq_len)
        # lengths: (batch_size,)
        
        # Forward pass through causal LSTM
        lstm_out, (hn, cn) = self.lstm(features) # lstm_out: (batch_size, seq_len, hidden_dim)
        
        if self.pooling_head == "final":
            # Extract the hidden state at the last valid unpadded turn for each item in batch
            batch_size = features.size(0)
            if lengths is not None:
                # Clamp index to [0, seq_len-1]
                idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, self.hidden_dim)
                rep = lstm_out.gather(1, idx).squeeze(1) # (batch_size, hidden_dim)
            else:
                rep = hn[-1] # last layer final hidden state
        elif self.pooling_head == "masked_mean":
            # Masked average pooling over all valid unpadded turns
            expanded_mask = mask.unsqueeze(-1) # (batch_size, seq_len, 1)
            masked_out = lstm_out * expanded_mask
            sum_out = masked_out.sum(dim=1) # (batch_size, hidden_dim)
            lengths_clamped = mask.sum(dim=1, keepdim=True).clamp(min=1.0) # (batch_size, 1)
            rep = sum_out / lengths_clamped
        else:
            raise ValueError(f"Unknown pooling_head: {self.pooling_head}")
            
        logits = self.classifier(rep).squeeze(-1) # (batch_size,)
        return logits

    def predict_causal_step(self, step_features, prev_hidden_state=None):
        # step_features: (batch_size, 1, input_dim) - single turn arrival in real time
        # prev_hidden_state: tuple (h_prev, c_prev)
        lstm_out, new_hidden_state = self.lstm(step_features, prev_hidden_state)
        rep = lstm_out[:, -1, :] # (batch_size, hidden_dim)
        logit = self.classifier(rep).squeeze(-1)
        risk = torch.sigmoid(logit)
        return risk, new_hidden_state
