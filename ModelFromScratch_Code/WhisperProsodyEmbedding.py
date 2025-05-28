import torch
import torch.nn as nn

class WhisperProsodyEmbedding(nn.Module):
    def __init__(self, whisper_model, prosody_dim=7):
        super().__init__()
        embed_dim = whisper_model.model.decoder.embed_tokens.embedding_dim
        self.token_embedding = whisper_model.model.decoder.embed_tokens
        self.positional_embedding = whisper_model.model.decoder.embed_positions
        self.prosody_projection = nn.Linear(prosody_dim, embed_dim)

    def forward(self, token_ids, prosody_features):
        # token_ids: (B, T)
        # prosody_features: (B, T, prosody_dim)
        tok_emb = self.token_embedding(token_ids)
        pos_ids = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        pos_emb = self.positional_embedding(pos_ids)
        pros_emb = self.prosody_projection(prosody_features)
        return tok_emb + pos_emb + pros_emb
