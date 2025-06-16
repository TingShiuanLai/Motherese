import torch
import torch.nn as nn

class WhisperProsodyEmbedding(nn.Module):
    def __init__(self, prosody_dim, hidden_size, max_len=512):
        super().__init__()
        self.prosody_projection = nn.Linear(prosody_dim, hidden_size)
        self.pos_embedding = nn.Embedding(max_len, hidden_size)

    def forward(self, prosody, positions=None):
        """
        prosody: (batch_size, seq_len, prosody_dim)
        positions: (batch_size, seq_len)
        """
        proj = self.prosody_projection(prosody)  # (B, T, H)

        if positions is None:
            positions = torch.arange(proj.size(1), device=proj.device).unsqueeze(0)  # (1, T)
        pos_embed = self.pos_embedding(positions)  # (B, T, H)

        min_len = min(proj.size(1), pos_embed.size(1))
        proj = proj[:, :min_len, :]
        pos_embed = pos_embed[:, :min_len, :]
        
        return torch.cat([proj, pos_embed], dim=-1)

        # return torch.cat([proj, pos_embed], dim=-1)  # (B, T, 2H)
