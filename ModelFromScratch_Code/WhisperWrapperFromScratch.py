import torch
import torch.nn as nn
from transformers import WhisperPreTrainedModel, WhisperModel, WhisperConfig
import os
import json


class WhisperProsodyModel(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig, prosody_dim: int):
        super().__init__(config)
        self.config = config

        # Base Whisper encoder-decoder model
        self.model = WhisperModel(config)

        # Project concatenated [token_embed ; prosody] back into d_model
        # self.prosody_projection = nn.Linear(7 + config.d_model, config.d_model)

        self.embedding_dim = self.model.decoder.embed_tokens.embedding_dim
        self.prosody_projection = nn.Linear(prosody_dim + self.embedding_dim, self.embedding_dim)
        # Final classification layer for predicting logits over vocabulary
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights per HF conventions
        self.init_weights()

    def forward(self, input_features, labels=None, prosody=None):
        # Check that all labels are within vocab range
        if labels is not None and torch.max(labels) >= self.config.vocab_size:
            raise ValueError(
                f"Token index {torch.max(labels).item()} exceeds vocab size {self.config.vocab_size}. "
                f"Check your tokenizer or pad token settings."
            )

        # print("Max label ID:", labels.max().item())
        # print("Vocab size:", self.model.decoder.embed_tokens.num_embeddings)
        # Token embeddings

        labels_clipped = labels.clone()
        labels_clipped[labels_clipped == -100] = 0  # Use <pad> or safe ID

        # print("Max label ID:", labels_clipped.max().item())
        # print("Vocab size:", self.model.decoder.embed_tokens.num_embeddings)


        # print("Labels shape:", labels_clipped.shape)
        # print("Labels dtype:", labels_clipped.dtype)
        # print("Labels min:", labels_clipped.min().item())
        # print("Labels max:", labels_clipped.max().item())
        # print("Vocab size:", self.model.decoder.embed_tokens.num_embeddings)
        # print("Unique labels > vocab size:",
        #     (labels_clipped >= self.model.decoder.embed_tokens.num_embeddings).sum().item())
        # print("Any negative labels:",
        #     (labels_clipped < 0).any().item())


        embedded_inputs = self.model.decoder.embed_tokens(labels_clipped)
        # embedded_inputs = self.model.decoder.embed_tokens(labels)

        # Concatenate prosody features if provided

        # if prosody is not None:
        #     # Ensure prosody is the same device and shape-compatible
        #     prosody = prosody.to(embedded_inputs.device)
        #     combined = torch.cat([embedded_inputs, prosody], dim=-1)
        #     decoder_inputs = self.prosody_projection(combined)
        # else:
        #     decoder_inputs = embedded_inputs


        if prosody is not None:
            prosody = prosody.to(embedded_inputs.device)
            combined = torch.cat([embedded_inputs, prosody], dim=-1)
            decoder_inputs = self.prosody_projection(combined)
        else:
            decoder_inputs = embedded_inputs

        # Encode input features (e.g., spectrogram)
        encoder_outputs = self.model.encoder(input_features=input_features)

        # Decode using modified embeddings
        decoder_outputs = self.model.decoder(
            inputs_embeds=decoder_inputs,
            encoder_hidden_states=encoder_outputs.last_hidden_state
        )

        # Output vocabulary logits
        logits = self.lm_head(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return type("Output", (), {"loss": loss, "logits": logits})
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Save the underlying HuggingFace Whisper model (weights + config)
        self.model.save_pretrained(save_directory)

        # Save this wrapper model's config (e.g., prosody details)
        config_path = os.path.join(save_directory, "prosody_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "prosody_dim": (self.prosody_projection.in_features - self.embedding_dim)
            }, f)

