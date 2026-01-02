import torch
import torch.nn as nn
import math

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# ============================================================
# Transformer Generator (Decoder-only)
# Adattato a (x,y,speed,angle)
# ============================================================

class TransformerGenerator(nn.Module):
    def __init__(
        self,
        latent_dim,
        cond_dim=4,      # [x_init, y_init, x_final, y_final]
        seq_len=120,
        d_model=256,
        nhead=4,
        num_layers=4,
        ff_dim=512
    ):
        super().__init__()

        self.seq_len = seq_len
        self.input_dim = latent_dim + cond_dim

        # Proiezione token iniziale
        self.token_fc = nn.Linear(self.input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        # Stack Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1,
            activation="gelu"
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Maschera causale autoregressiva
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

        # OUTPUT: 4 feature → [x, y, speed, angle]
        self.output_fc = nn.Linear(d_model, 4)

    def forward(self, z, cond, pad_mask=None):
        """
        z:    (B, latent_dim)
        cond: (B, 4)
        pad_mask: (B, 120) True=padding (facoltativa)
        """
        token = torch.cat([z, cond], dim=1)      # (B, latent+4)
        x = self.token_fc(token)                 # (B, d_model)

        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B,T,d_model)
        x = self.pos_encoder(x)

        # Se passi pad_mask, il Transformer evita di "usare" i timestep padding
        output = self.decoder(x, mask=self.causal_mask, src_key_padding_mask=pad_mask)

        return self.output_fc(output) 

# ============================================================
# Transformer Discriminator (Critic)
# Adattato a (x,y,speed,angle)
# ============================================================

class TransformerDiscriminator(nn.Module):
    def __init__(
        self,
        cond_dim=4,      # [x_init, y_init, x_final, y_final]
        seq_len=120,
        d_model=256,
        nhead=4,
        num_layers=3,
        ff_dim=512
    ):
        super().__init__()

        self.seq_len = seq_len

        # Input: 4 feature dinamiche
        self.input_fc = nn.Linear(4, d_model)

        # Condizione → bias temporale
        self.cond_fc = nn.Linear(cond_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Critic output
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, traj, cond, pad_mask=None):
        """
        traj: (B, 120, 4)
        cond: (B, 4)
        pad_mask: (B, 120) boolean, True = padding (da ignorare)
        """

        batch = traj.size(0)

        x = self.input_fc(traj)  # (B,120,256)

        # Condizionamento timestep-wise
        cond_embed = self.cond_fc(cond).unsqueeze(1)   # (B,1,256)
        cond_embed = cond_embed.repeat(1, self.seq_len, 1)

        x = x + cond_embed
        x = self.pos_encoder(x)

        #src_key_padding_mask=True => ignora quei timestep
        x = self.encoder(x,src_key_padding_mask=pad_mask)

        # Mean pooling temporale
        if pad_mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = (~pad_mask).unsqueeze(-1).type_as(x)  # (B,T,1) 1=valido
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        return self.fc_out(pooled)  # (B,1)