import torch
import torch.nn as nn
from transformers import WhisperPreTrainedModel, WhisperModel, WhisperConfig
from WhisperProsodyEmbedding import WhisperProsodyEmbedding
import os
import json

class WhisperProsodyModel(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig, prosody_dim: int):
        super().__init__(config)
        self.config = config
        self.model = WhisperModel(config)

        self.hidden_size = config.d_model
        self.embedding_dim = self.model.decoder.embed_tokens.embedding_dim

        # Embed prosody with learned projection + positional embedding
        self.prosody_embedding = WhisperProsodyEmbedding(prosody_dim, self.hidden_size)

        # Fuse: [token_embedding ; prosody+position] → d_model
        self.input_projection = nn.Linear(self.embedding_dim + 2 * self.hidden_size, self.hidden_size)

        # Final classification head (optional)
        self.lm_head = nn.Linear(self.hidden_size, config.vocab_size, bias=False)

        self.init_weights()

    def forward(self, input_features, labels=None, prosody=None, positions=None):
        # Replace ignore index with 0 for safe embedding
        labels_clipped = labels.clone()
        labels_clipped[labels_clipped == -100] = 0

        # Decoder token embeddings: (B, T, E)
        token_embeds = self.model.decoder.embed_tokens(labels_clipped)

        if prosody is not None:
            prosody = prosody.to(token_embeds.device)

            # Generate default positions if not provided
            if positions is None:
                positions = torch.arange(
                    prosody.size(1), device=prosody.device
                ).unsqueeze(0).expand(prosody.size(0), -1)

            # Prosody embedding includes positional embedding: (B, T, 2H)
            prosody_embeds = self.prosody_embedding(prosody, positions)

            # Align lengths before concatenating
            min_len = min(token_embeds.size(1), prosody_embeds.size(1))
            token_embeds = token_embeds[:, :min_len, :]
            prosody_embeds = prosody_embeds[:, :min_len, :]

            # Fuse: concat + projection → decoder input
            decoder_inputs = torch.cat([token_embeds, prosody_embeds], dim=-1)
            decoder_inputs = self.input_projection(decoder_inputs)
        else:
            decoder_inputs = token_embeds

        # Encode audio (or silence) → encoder_hidden_states
        encoder_outputs = self.model.encoder(input_features=input_features)

        # Decode using fused inputs
        decoder_outputs = self.model.decoder(
            inputs_embeds=decoder_inputs,
            encoder_hidden_states=encoder_outputs.last_hidden_state
        )

        logits = self.lm_head(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return type("Output", (), {"loss": loss, "logits": logits})

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)

        with open(os.path.join(save_directory, "prosody_config.json"), "w") as f:
            json.dump({
                "prosody_dim": self.prosody_embedding.prosody_projection.in_features
            }, f)
