import torch
import torch.nn as nn
from torchtyping import TensorType
from einops import rearrange
from .perceiver import PerceiverResampler
from .encoders.audio_encoders import get_audio_encoder
from .encoders.vision_encoders import get_vision_encoder


ENCODER_OUT_DIMS = {
    "dinov2_base": 768, 
    "dinov2_large": 1024, 
    "dinov2_small": 384,
}

ENCODER_SEQ_LENS = {
    "dinov2_base": 257, 
    "dinov2_large": 257, 
    "dinov2_small": 257,
}

# MultModal Encoder for Vision and Audio 
class MultiModalEncoder(nn.Module):

    """
    Takes in a batch of visions and returns a batch of embeddings of the
    same dimensions as the LM's word embeddings.

    :param config: Neox args
    :param out_dim: output dimension of the embedding
    :param device: device to run the model on
    """

    def __init__(
        self,
        config,
        out_dim: int = 2048,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.modality = config.modality 
        if self.modality == "vision":
            self.encoder = get_vision_encoder(config.encoder_name, load_path=config.load_vision_encoder_path)
        elif self.modality == "audio":
            self.encoder = get_audio_encoder(config.encoder_name, load_path=config.load_audio_encoder_path)
        else:
            raise ValueError(f"modality {self.modality} not recognized")

        self.encoder_out_dim = ENCODER_OUT_DIMS[
            self.encoder_type
        ]  # out dim for vision encoder
        self.encoder_seq_len = ENCODER_SEQ_LENS[
            self.encoder_type
        ]
        self.out_dim = out_dim  # out dim for lm
        self.proj = nn.Linear(self.encoder_out_dim, self.out_dim)
        self.dropout = nn.Dropout(config.embed_dropout_prob)
        self.use_layernorm = config.use_embed_layernorm
        if self.use_layernorm:
            self.ln = nn.LayerNorm(self.out_dim)
        self.perceiver = PerceiverResampler(dim=config.encoder_seq_length)
    
    def forward(
        self, x: TensorType["b", "t", "c", "h", "w"] or TensorType["b", "t", "f", "c", "h", "w"]
    ) -> TensorType["b", "seq", "out_dim"]:

        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        if self.modality == "vision":
            B, T, F, C, H, W = x.shape
            x = rearrange(x, "b t f c h w -> (b t) f c h w")
        elif self.modality == "audio":
            B, T, C, H, W = x.shape
            x = rearrange(x, "b t c h w -> (b t) c h w")
        else:
            raise ValueError(f"modality {self.modality} not recognized")

        embeddings = self.encoder(x) # (B, T), N_E, E
        embeddings = rearrange(embeddings, "(b t) n_e e -> b t n_e e", b=B, t=T) # (B, T, N_E, E
        B, T, N_E, E = embeddings.shape
        assert N_E == self.encoder_seq_len
        
        embeddings = rearrange(embeddings, "b t n_e e -> (b t) n_e e")
        embeddings = self.perceiver(embeddings) # (B*T, N_E_new, E)

        logits = self.proj(logits) # (B*T, N_E_new, E_L)
        logits = self.dropout(logits) # (B*T, N_E_new, E_L)

        if self.use_layernorm:
            logits = self.ln(logits) # (B*T, N_E_new, E_L)

        logits = rearrange(logits, "(b t) n_e e_l -> b t n_e e_l", b=B, t=T) # (B, T, N_E_new, E_L)
    
        return logits