"""Rede LSTM e dataset compartilhados pelo BC (bc.py) e pelo DAgger (dagger.py).

Contem:
  - Constantes da arquitetura (OBS_DIM, N_ACOES, dimensoes da rede).
  - ConjuntoDadosBC: organiza episodios em janelas de tamanho fixo para o LSTM.
  - RedeBCLSTM: a politica (encoder FC -> LSTM -> cabeca), com mascaramento de
    acoes invalidas (logits -1e8 antes do softmax/argmax).

Como BC e DAgger compartilham exatamente a mesma rede e o mesmo formato de dados,
estes componentes ficam em um modulo unico, importado pelos dois.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

# --- Constantes ---------------------------------------------------------------
OBS_DIM       = 148
N_ACOES       = 15
LEN_SEQUENCIA = 32   # janela temporal para treino do LSTM

# Hiperparâmetros de rede
DIM_ENCODER  = 256
DIM_LSTM     = 512
N_CAMADAS    = 2


# --- Dataset ------------------------------------------------------------------
class ConjuntoDadosBC(Dataset):
    """
    Organiza episódios em janelas de comprimento fixo para treino do LSTM.
    Nunca cruza fronteiras de episódio → hidden state zerado entre janelas.
    """
    def __init__(
        self,
        episodios: List[List[Dict]],
        obs_media: np.ndarray,
        obs_std: np.ndarray,
        len_seq: int = LEN_SEQUENCIA,
    ):
        self.obs_media = obs_media
        self.obs_std   = obs_std

        self.sequencias: List[Tuple] = []
        for ep in episodios:
            n = len(ep)
            for inicio in range(0, n - len_seq + 1, len_seq):
                janela   = ep[inicio : inicio + len_seq]
                obs_seq  = np.array([t['obs']     for t in janela], dtype=np.float32)
                mask_seq = np.array([t['mascara'] for t in janela], dtype=np.float32)
                acao_seq = np.array([t['acao']    for t in janela], dtype=np.int64)
                self.sequencias.append((obs_seq, mask_seq, acao_seq))

    def __len__(self) -> int:
        return len(self.sequencias)

    def __getitem__(self, idx: int):
        obs_seq, mask_seq, acao_seq = self.sequencias[idx]
        obs_norm = (obs_seq - self.obs_media) / (self.obs_std + 1e-8)
        return (
            torch.FloatTensor(obs_norm),
            torch.FloatTensor(mask_seq),
            torch.LongTensor(acao_seq),
        )


# --- Rede Política ------------------------------------------------------------
class RedeBCLSTM(nn.Module):
    """
    Política LSTM para Behavioral Cloning (compartilhada com o DAgger).

    Forward: (obs_seq, mask_seq, hidden?) → (logits_mascarados, nova_hidden)
      obs_seq:  (B, T, obs_dim)
      mask_seq: (B, T, n_acoes) — 1=válida, 0=inválida
    """
    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_acoes: int = N_ACOES,
        dim_enc: int = DIM_ENCODER,
        dim_lstm: int = DIM_LSTM,
        n_camadas: int = N_CAMADAS,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, dim_enc),
            nn.GELU(),
            nn.LayerNorm(dim_enc),
            nn.Linear(dim_enc, dim_enc),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            dim_enc, dim_lstm, n_camadas,
            batch_first=True, dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim_lstm),
            nn.Linear(dim_lstm, dim_enc),
            nn.GELU(),
            nn.Linear(dim_enc, n_acoes),
        )
        self.dim_lstm  = dim_lstm
        self.n_camadas = n_camadas

    def forward(
        self,
        obs_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        B, T, _ = obs_seq.shape
        enc = self.encoder(obs_seq.view(B * T, -1)).view(B, T, -1)
        lstm_out, hidden = self.lstm(enc, hidden)
        logits = self.head(lstm_out)                           # (B, T, n_acoes)
        masked_logits = logits + (1.0 - mask_seq) * (-1e8)   # mascara ações inválidas
        return masked_logits, hidden

    def inicializar_hidden(self, batch_size: int, device: str = 'cpu') -> Tuple:
        return (
            torch.zeros(self.n_camadas, batch_size, self.dim_lstm, device=device),
            torch.zeros(self.n_camadas, batch_size, self.dim_lstm, device=device),
        )
