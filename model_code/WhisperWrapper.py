from transformers import WhisperForConditionalGeneration
from WhisperProsodyEmbedding import WhisperProsodyEmbedding
import torch.nn as nn
import os
import json
import torch
import torch.nn.functional as F

class WhisperWithProsody(nn.Module):
    def __init__(self, model_name="openai/whisper-small", prosody_dim=7):
        super().__init__()
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.prosody_embed = WhisperProsodyEmbedding(self.model, prosody_dim)
        self.prosody_dim = prosody_dim

    # def forward(
    #     self,
    #     input_features,           # Audio encoder input
    #     decoder_input_ids,        # Token IDs for decoder
    #     prosody_features,         # (B, T, 7)
    #     labels=None,              # For loss calculation
    # ):
    #     decoder_input_embeds = self.prosody_embed(decoder_input_ids, prosody_features)
        
    #     return self.model(
    #         input_features=input_features,
    #         decoder_inputs_embeds=decoder_input_embeds,
    #         labels=labels,  # Needed for computing loss
    #         use_cache=False  # safer for training
    #     )
    
    def forward(
        self,
        input_features,
        decoder_input_ids,
        prosody_features,
        labels=None,
    ):
        decoder_input_embeds = self.prosody_embed(decoder_input_ids, prosody_features)

        if input_features is not None:
            return self.model(
                input_features=input_features,
                decoder_inputs_embeds=decoder_input_embeds,
                labels=labels,
                use_cache=False
            )
        else:
            # Decoder-only mode (used for evaluating text-only tasks like BLiMP)
            batch_size, seq_len, _ = decoder_input_embeds.size()

            # Create dummy encoder output (1 timestep)
            encoder_hidden = torch.zeros(batch_size, 1, self.model.config.d_model).to(decoder_input_embeds.device)

            # Forward pass through decoder only
            decoder_outputs = self.model.model.decoder(
                input_ids=None,
                inputs_embeds=decoder_input_embeds,
                encoder_hidden_states=encoder_hidden,
                return_dict=True
            )

            logits = self.model.proj_out(decoder_outputs.last_hidden_state)

            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=self.model.config.pad_token_id
                )
                return {"loss": loss, "logits": logits}
            else:
                return {"logits": logits}
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)

        # Save prosody config separately
        config_path = os.path.join(save_directory, "prosody_config.json")
        with open(config_path, "w") as f:
            json.dump({"prosody_dim": self.prosody_dim}, f)
