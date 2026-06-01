"""
Agente de Imitation Learning — Behavioral Cloning + DAgger com LSTM
====================================================================

Aprende a clonar o comportamento do BotPerfeito a partir das trajetórias
de especialista (18 250 transições) usando duas fases:

  Fase 1 — Behavioral Cloning (BC):
    Aprendizado supervisionado puro nas trajetórias salvas pelo bot.
    Loss: cross-entropy mascarada (ações inválidas recebem logit=-inf).

  Fase 2 — DAgger (Dataset Aggregation):
    O agente executa no ambiente real; o bot rotula cada estado visitado
    com a ação ótima. Dados agregados → fine-tuning iterativo.
    Trata distribuição shift que BC puro não cobre.

Arquitetura:
  obs(148) → Encoder FC(256, GELU, LN) → LSTM(512, 2 camadas) → Head → logits(15)
  Mascaramento: logits[inválida] -= 1e8 antes de argmax/softmax

Uso:
  python il_agent.py                  # BC (50 épocas) + DAgger (5 rounds)
  python il_agent.py --so-bc          # só Behavioral Cloning
  python il_agent.py --n-epocas-bc 30 --n-rounds-dagger 3
"""

import os, sys, json, time, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# --- Paths --------------------------------------------------------------------
_DIR     = os.path.dirname(os.path.abspath(__file__))
_ENV_DIR = os.path.join(_DIR, '../../environment')
_PPO_DIR = os.path.join(_DIR, '../ppo-lstm-masked')
sys.path.insert(0, _ENV_DIR)
sys.path.insert(0, _PPO_DIR)
sys.path.insert(0, _DIR)
for _sub in ['plant', 'birds', 'pollinators', 'soil', 'weather',
             'weeds', 'pest', 'cides-fertilizers', 'facilities']:
    sys.path.insert(0, os.path.join(_ENV_DIR, _sub))

from env import AmbienteFazendaGym
from ppo_lstm_masked import InvolucroMascaraAcoes
from bot_perfeito import BotPerfeito, NOMES_ACOES

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
    Política LSTM para Behavioral Cloning.

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


# --- Agente IL ----------------------------------------------------------------
class AgenteBCLSTM:
    """
    Agente de Imitation Learning com Behavioral Cloning + DAgger.

    Uso típico:
        agente = AgenteBCLSTM()
        episodios, _ = agente.carregar_trajetorias('results/bot_perfeito/**/*.json')
        agente.calcular_normalizacao(episodios)
        agente.treinar_bc(episodios, n_epocas=50)
        agente.treinar_dagger(episodios, env_wrapped, bot, n_rounds=5)
        agente.salvar('results/il_agent/il_dagger.pt')
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_acoes: int = N_ACOES,
        dim_encoder: int = DIM_ENCODER,
        dim_lstm: int = DIM_LSTM,
        n_camadas: int = N_CAMADAS,
        dispositivo: Optional[str] = None,
    ):
        self.dispositivo = dispositivo or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.rede = RedeBCLSTM(obs_dim, n_acoes, dim_encoder, dim_lstm, n_camadas)
        self.rede.to(self.dispositivo)
        self.obs_media = np.zeros(obs_dim, dtype=np.float32)
        self.obs_std   = np.ones(obs_dim, dtype=np.float32)
        self.historico: Dict[str, List] = defaultdict(list)

        n_params = sum(p.numel() for p in self.rede.parameters())
        print(f"[rede] {n_params:,} parâmetros | dispositivo: {self.dispositivo}")

    # -- Dados -----------------------------------------------------------------

    def carregar_trajetorias(self, padrao_glob: str) -> Tuple[List[List[Dict]], str]:
        """Carrega o arquivo de trajetórias mais recente."""
        arquivos = sorted(glob.glob(padrao_glob, recursive=True))
        if not arquivos:
            raise FileNotFoundError(f"Nenhuma trajetória em: {padrao_glob}")
        caminho = arquivos[-1]
        with open(caminho, encoding='utf-8') as f:
            raw = json.load(f)
        episodios = []
        for ep_raw in raw:
            ep = [{
                'obs':     np.array(t['obs'],     dtype=np.float32),
                'acao':    int(t['acao']),
                'mascara': np.array(t['mascara'], dtype=np.float32),
            } for t in ep_raw]
            episodios.append(ep)
        n_trans = sum(len(e) for e in episodios)
        print(f"[dados] {len(episodios)} episódios, {n_trans} transições — {os.path.basename(os.path.dirname(caminho))}")
        return episodios, caminho

    def calcular_normalizacao(self, episodios: List[List[Dict]]):
        """Calcula média/std por feature em todas as observações."""
        todas = np.stack([t['obs'] for ep in episodios for t in ep])
        self.obs_media = todas.mean(axis=0).astype(np.float32)
        self.obs_std   = todas.std(axis=0).astype(np.float32)
        print(f"[norm] obs_dim={todas.shape[1]} | média global={self.obs_media.mean():.3f} "
              f"| std global={self.obs_std.mean():.3f}")

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.obs_media) / (self.obs_std + 1e-8)

    # -- Behavioral Cloning ----------------------------------------------------

    def treinar_bc(
        self,
        episodios: List[List[Dict]],
        n_epocas: int = 50,
        batch_size: int = 64,
        lr: float = 3e-4,
        len_seq: int = LEN_SEQUENCIA,
        env_aval=None,
        n_eps_aval: int = 10,
        freq_aval: int = 10,
    ) -> Dict:
        """
        Fase 1: Behavioral Cloning por supervisão direta nas trajetórias do bot.

        Loss = cross-entropy(logits_mascarados, acao_especialista)
        """
        dataset = ConjuntoDadosBC(episodios, self.obs_media, self.obs_std, len_seq)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        n_seqs  = len(dataset)

        otimizador = torch.optim.Adam(self.rede.parameters(), lr=lr, weight_decay=1e-5)
        agendador  = torch.optim.lr_scheduler.CosineAnnealingLR(
            otimizador, T_max=n_epocas, eta_min=lr * 0.01,
        )

        _sep = '-' * 62
        print(f"\n{_sep}")
        print(f"  BEHAVIORAL CLONING — {n_epocas} épocas")
        print(f"  Sequências: {n_seqs} × {len_seq} passos | batch: {batch_size} | LR: {lr:.0e}")
        print(_sep)

        t0_total = time.time()
        hist: Dict[str, List] = defaultdict(list)

        for epoca in range(1, n_epocas + 1):
            t0_ep = time.time()
            self.rede.train()
            perdas, acertos, total = [], 0, 0

            for obs_seq, mask_seq, acao_seq in loader:
                obs_seq  = obs_seq.to(self.dispositivo)
                mask_seq = mask_seq.to(self.dispositivo)
                acao_seq = acao_seq.to(self.dispositivo)

                hidden = self.rede.inicializar_hidden(obs_seq.size(0), self.dispositivo)
                logits, _ = self.rede(obs_seq, mask_seq, hidden)

                B, T, A = logits.shape
                loss = F.cross_entropy(logits.view(B * T, A), acao_seq.view(B * T))

                otimizador.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.rede.parameters(), 1.0)
                otimizador.step()

                perdas.append(loss.item())
                acertos += (logits.argmax(-1) == acao_seq).sum().item()
                total   += B * T

            agendador.step()

            perda_m  = float(np.mean(perdas))
            acuracia = acertos / total
            tempo_ep = time.time() - t0_ep
            lr_atual = agendador.get_last_lr()[0]

            hist['perda'].append(perda_m)
            hist['acuracia'].append(acuracia)
            hist['tempo_ep'].append(tempo_ep)
            self.historico['bc_perda'].append(perda_m)
            self.historico['bc_acuracia'].append(acuracia)

            if epoca % 5 == 0 or epoca == 1:
                t_total = time.time() - t0_total
                print(f"  Época {epoca:3d}/{n_epocas} | "
                      f"Perda: {perda_m:.4f} | "
                      f"Acur: {acuracia:.1%} | "
                      f"LR: {lr_atual:.2e} | "
                      f"Ep: {tempo_ep:.1f}s | Total: {t_total:.0f}s")

            if env_aval is not None and epoca % freq_aval == 0:
                stats = self.avaliar(env_aval, n_eps_aval)
                hist['aval_recompensa'].append(stats['media'])
                hist['aval_epoca'].append(epoca)
                self.historico['bc_aval_recompensa'].append(stats['media'])
                print(f"    → avaliação: {stats['media']:.1f} ± {stats['std']:.1f}")

        t_bc = time.time() - t0_total
        print(f"\n  BC concluído em {t_bc:.1f}s ({t_bc/60:.1f} min)")
        print(f"  Acurácia final: {hist['acuracia'][-1]:.1%} | Perda: {hist['perda'][-1]:.4f}")
        return dict(hist)

    # -- DAgger ----------------------------------------------------------------

    @staticmethod
    def _env_bruto(env_wrapped) -> 'AmbienteFazendaGym':
        """Retorna o AmbienteFazendaGym desembrulhado para o bot."""
        env = env_wrapped
        while hasattr(env, 'env'):
            env = env.env
        return env

    def _coletar_dagger(
        self,
        env_wrapped,
        bot: 'BotPerfeito',
        n_passos: int,
        beta: float = 0.0,
        temperatura: float = 0.0,
        epsilon: float = 0.0,
    ) -> List[List[Dict]]:
        """
        Coleta n_passos transições para DAgger.

        O agente executa no ambiente (com prob 1-beta).
        O bot SEMPRE fornece o label correto (oráculo).
        beta=0.5 → agente e bot compartilham execução (início do treino)
        beta=0.0 → só o agente executa (DAgger padrão)

        Exploração do agente (apenas durante coleta DAgger):
          temperatura > 0 → amostra de softmax(logits/τ) sobre ações válidas
          epsilon > 0     → ε-greedy: ação uniforme entre válidas com prob ε

        Retorna lista de episódios (listas de transições) para uso no dataset.
        """
        self.rede.eval()
        env_raw = self._env_bruto(env_wrapped)

        episodios: List[List[Dict]] = []
        ep_atual:  List[Dict] = []

        obs, info = env_wrapped.reset()
        bot.resetar_episodio()
        hidden = self.rede.inicializar_hidden(1, self.dispositivo)
        passos = 0

        while passos < n_passos:
            mask = info.get('mascara_acoes', np.ones(N_ACOES, dtype=np.float32))

            # Bot rotula o estado atual
            acao_bot = bot.selecionar_acao(env_raw)

            # Agente computa sua própria ação (com possível exploração)
            acao_agente, hidden = self.selecionar_acao(
                obs, mask, hidden,
                temperatura=temperatura,
                epsilon=epsilon,
            )

            # Executa: bot ou agente?
            acao_exec = acao_bot if np.random.random() < beta else acao_agente

            # Salva transição com label do bot (oráculo)
            ep_atual.append({
                'obs':     obs.copy(),
                'acao':    acao_bot,      # label do especialista
                'mascara': mask.copy(),
            })
            passos += 1

            obs, _, done, trunc, info = env_wrapped.step(acao_exec)

            if done or trunc:
                if ep_atual:
                    episodios.append(ep_atual)
                ep_atual = []
                obs, info = env_wrapped.reset()
                bot.resetar_episodio()
                hidden = self.rede.inicializar_hidden(1, self.dispositivo)

        if ep_atual:
            episodios.append(ep_atual)

        return episodios

    def treinar_dagger(
        self,
        episodios_base: List[List[Dict]],
        env_wrapped,
        bot: 'BotPerfeito',
        n_rounds: int = 5,
        passos_por_round: int = 3650,
        epocas_por_round: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,
        beta_inicial: float = 0.5,
        beta_final: float = 0.0,
        env_aval=None,
        n_eps_aval: int = 10,
    ) -> Dict:
        """
        Fase 2: DAgger — agrega dados do agente rotulados pelo bot iterativamente.

        A cada round:
          1. Agente executa no ambiente (parcialmente guiado pelo bot via beta)
          2. Bot rotula cada estado visitado com ação ótima
          3. Dados são agregados ao conjunto de treino
          4. Fine-tuning da rede nas transições agregadas
        """
        _sep = '-' * 62
        print(f"\n{_sep}")
        print(f"  DAGGER — {n_rounds} rounds × {passos_por_round} passos/round")
        print(f"  Épocas/round: {epocas_por_round} | LR: {lr:.0e} | beta: {beta_inicial:.1f}→{beta_final:.1f}")
        print(_sep)

        t0_total = time.time()
        hist: Dict[str, List] = defaultdict(list)

        # Dataset acumulado: começa com dados do especialista
        episodios_agg = list(episodios_base)

        for rnd in range(1, n_rounds + 1):
            beta = (beta_inicial + (beta_final - beta_inicial) * (rnd - 1) / max(n_rounds - 1, 1))
            t0_rnd = time.time()

            print(f"\n  -- Round {rnd}/{n_rounds} (beta={beta:.2f}) --")

            # 1. Coleta
            t0_col = time.time()
            novos_eps = self._coletar_dagger(env_wrapped, bot, passos_por_round, beta=beta)
            n_novos   = sum(len(e) for e in novos_eps)
            episodios_agg.extend(novos_eps)
            n_total = sum(len(e) for e in episodios_agg)
            print(f"     Coleta: {n_novos} transições em {time.time()-t0_col:.1f}s | "
                  f"acumulado: {n_total}")

            # 2. Dataset e DataLoader sobre dados agregados
            dataset = ConjuntoDadosBC(episodios_agg, self.obs_media, self.obs_std, LEN_SEQUENCIA)
            loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            otimizador = torch.optim.Adam(self.rede.parameters(), lr=lr, weight_decay=1e-5)

            # 3. Fine-tuning
            t0_treino = time.time()
            for epoca in range(1, epocas_por_round + 1):
                self.rede.train()
                perdas, acertos, total = [], 0, 0
                for obs_seq, mask_seq, acao_seq in loader:
                    obs_seq  = obs_seq.to(self.dispositivo)
                    mask_seq = mask_seq.to(self.dispositivo)
                    acao_seq = acao_seq.to(self.dispositivo)
                    hidden = self.rede.inicializar_hidden(obs_seq.size(0), self.dispositivo)
                    logits, _ = self.rede(obs_seq, mask_seq, hidden)
                    B, T, A = logits.shape
                    loss = F.cross_entropy(logits.view(B * T, A), acao_seq.view(B * T))
                    otimizador.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.rede.parameters(), 1.0)
                    otimizador.step()
                    perdas.append(loss.item())
                    acertos += (logits.argmax(-1) == acao_seq).sum().item()
                    total   += B * T

            perda_m  = float(np.mean(perdas))
            acuracia = acertos / total
            t_treino = time.time() - t0_treino
            print(f"     Treino ({epocas_por_round} épocas): perda={perda_m:.4f} "
                  f"acur={acuracia:.1%} em {t_treino:.1f}s")
            hist[f'r{rnd}_perda'].append(perda_m)
            hist[f'r{rnd}_acuracia'].append(acuracia)
            self.historico['dagger_perda'].append(perda_m)
            self.historico['dagger_acuracia'].append(acuracia)

            # 4. Avaliação do round
            if env_aval is not None:
                stats = self.avaliar(env_aval, n_eps_aval)
                hist[f'r{rnd}_aval'] = stats['media']
                self.historico['dagger_aval'].append(stats['media'])
                self.historico['dagger_aval_round'].append(rnd)
                t_rnd = time.time() - t0_rnd
                print(f"     Avaliação: {stats['media']:.1f} ± {stats['std']:.1f} "
                      f"| round em {t_rnd:.0f}s")

        t_dag = time.time() - t0_total
        print(f"\n  DAgger concluído em {t_dag:.1f}s ({t_dag/60:.1f} min)")
        return dict(hist)

    # -- Avaliação --------------------------------------------------------------

    def avaliar(self, env_wrapped, n_episodios: int = 20) -> Dict:
        """Avalia o agente no ambiente real, mantendo hidden state por episódio."""
        self.rede.eval()
        recompensas = []

        for _ in range(n_episodios):
            obs, info = env_wrapped.reset()
            hidden = self.rede.inicializar_hidden(1, self.dispositivo)
            rec_ep = 0.0
            done   = False

            while not done:
                mask = info.get('mascara_acoes', np.ones(N_ACOES, dtype=np.float32))
                obs_t  = torch.FloatTensor(self._norm(obs)).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
                mask_t = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(self.dispositivo)

                with torch.no_grad():
                    logits, hidden = self.rede(obs_t, mask_t, hidden)

                acao = int(logits.squeeze().argmax().item())
                obs, r, done_f, trunc, info = env_wrapped.step(acao)
                rec_ep += r
                done    = done_f or trunc

            recompensas.append(rec_ep)

        return {
            'media': float(np.mean(recompensas)),
            'std':   float(np.std(recompensas)),
            'min':   float(np.min(recompensas)),
            'max':   float(np.max(recompensas)),
        }

    def selecionar_acao(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        hidden: Optional[Tuple],
        temperatura: float = 0.0,
        epsilon: float = 0.0,
    ) -> Tuple[int, Tuple]:
        """Interface de inferência: retorna (acao, nova_hidden).

        Args:
            temperatura: softmax temperature τ. Se τ>0, amostra de softmax(logits/τ)
                          sobre ações válidas. τ=0 ⇒ argmax (determinístico).
            epsilon: ε-greedy. Se ε>0, com prob ε escolhe ação UNIFORME entre as
                      válidas; caso contrário usa temperatura/argmax.

        Os dois mecanismos são compostos: ε-greedy é avaliado primeiro
        (com prob ε escolhe uniforme). Se NÃO entrar no ε-branch, então
        avalia τ (sample softmax ou argmax).
        """
        self.rede.eval()
        # 1) ε-greedy: com prob ε, ação UNIFORME entre as válidas
        if epsilon > 0.0 and np.random.random() < epsilon:
            validas = np.where(mask > 0.5)[0]
            acao_eps = int(np.random.choice(validas)) if len(validas) else 14
            # ainda precisamos atualizar hidden para a próxima chamada
            obs_t  = torch.FloatTensor(self._norm(obs)).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
            with torch.no_grad():
                _, hidden = self.rede(obs_t, mask_t, hidden)
            return acao_eps, hidden

        # 2) Forward da rede + mascaramento
        obs_t  = torch.FloatTensor(self._norm(obs)).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
        mask_t = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
        with torch.no_grad():
            logits, hidden = self.rede(obs_t, mask_t, hidden)
        logits = logits.squeeze()                                # (n_acoes,)
        logits = logits.masked_fill(mask_t.squeeze() < 0.5, -1e8)  # garante validade

        # 3) Softmax com temperatura ou argmax
        if temperatura > 0.0:
            probs = torch.nn.functional.softmax(logits / temperatura, dim=-1)
            acao = int(torch.distributions.Categorical(probs).sample().item())
        else:
            acao = int(logits.argmax().item())
        return acao, hidden

    def inicializar_hidden(self) -> Tuple:
        return self.rede.inicializar_hidden(1, self.dispositivo)

    # -- Persistência -----------------------------------------------------------

    def salvar(self, caminho: str):
        os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
        torch.save({
            'estado_rede': self.rede.state_dict(),
            'obs_media':   self.obs_media,
            'obs_std':     self.obs_std,
            'historico':   dict(self.historico),
            'config': dict(
                obs_dim=OBS_DIM, n_acoes=N_ACOES,
                dim_enc=DIM_ENCODER, dim_lstm=DIM_LSTM, n_camadas=N_CAMADAS,
            ),
        }, caminho)
        print(f"[salvo] {caminho}")

    def carregar(self, caminho: str):
        ckpt = torch.load(caminho, map_location=self.dispositivo, weights_only=False)
        self.rede.load_state_dict(ckpt['estado_rede'])
        self.obs_media = ckpt['obs_media']
        self.obs_std   = ckpt['obs_std']
        if 'historico' in ckpt:
            self.historico = defaultdict(list, ckpt['historico'])
        print(f"[carregado] {caminho}")


# --- Script principal ---------------------------------------------------------

def _baseline_random(env_wrapped, n_episodios: int = 10) -> Dict:
    recompensas = []
    for _ in range(n_episodios):
        obs, info = env_wrapped.reset()
        total = 0.0
        done  = False
        while not done:
            acao = env_wrapped.action_space.sample()
            obs, r, done_f, trunc, info = env_wrapped.step(acao)
            total += r
            done   = done_f or trunc
        recompensas.append(total)
    return {'media': float(np.mean(recompensas)), 'std': float(np.std(recompensas))}


def _resumo_final(stats_bc, stats_dag, stats_rnd, bot_ref=8637):
    _sep = '=' * 62
    print(f"\n{_sep}")
    print(f"  RESUMO FINAL")
    print(_sep)
    print(f"  Bot Perfeito  (referência) : {bot_ref:>8.0f}")
    print(f"  IL após BC                 : {stats_bc['media']:>8.1f} ± {stats_bc['std']:.1f}"
          f"  ({100*stats_bc['media']/bot_ref:.1f}% do bot)")
    if stats_dag is not None:
        print(f"  IL após DAgger             : {stats_dag['media']:>8.1f} ± {stats_dag['std']:.1f}"
              f"  ({100*stats_dag['media']/bot_ref:.1f}% do bot)")
    print(f"  Random        (referência) : {stats_rnd['media']:>8.1f} ± {stats_rnd['std']:.1f}")
    print(_sep)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Agente IL (BC + DAgger)')
    ap.add_argument('--so-bc',           action='store_true', help='Só BC, sem DAgger')
    ap.add_argument('--n-epocas-bc',     type=int, default=50)
    ap.add_argument('--n-rounds-dagger', type=int, default=5)
    ap.add_argument('--passos-dagger',   type=int, default=3650, help='Passos por round DAgger')
    ap.add_argument('--epocas-dagger',   type=int, default=10,   help='Épocas de fine-tuning/round')
    ap.add_argument('--n-aval',          type=int, default=10,   help='Episódios de avaliação')
    ap.add_argument('--carregar',        type=str, default=None,  help='Checkpoint .pt para continuar')
    args = ap.parse_args()

    # Ambiente e wrappers
    env_base    = AmbienteFazendaGym(passos_maximos=365)
    env_wrapped = InvolucroMascaraAcoes(env_base)
    bot         = BotPerfeito()

    # Agente
    agente = AgenteBCLSTM()
    if args.carregar:
        agente.carregar(args.carregar)

    # Trajetórias
    padrao = os.path.join(_DIR, '../../results/bot_perfeito/**/trajetorias_especialista.json')
    episodios, _ = agente.carregar_trajetorias(padrao)
    agente.calcular_normalizacao(episodios)

    dir_saida = os.path.join(_DIR, '../../results/il_agent')
    os.makedirs(dir_saida, exist_ok=True)

    t_inicio = time.time()
    print(f"\n[início] {time.strftime('%H:%M:%S')}")

    # -- Fase 1: BC ------------------------------------------------------------
    agente.treinar_bc(
        episodios,
        n_epocas=args.n_epocas_bc,
        env_aval=env_wrapped,
        n_eps_aval=args.n_aval,
        freq_aval=10,
    )

    stats_bc = agente.avaliar(env_wrapped, args.n_aval * 2)
    print(f"\n[BC] avaliação final: {stats_bc['media']:.1f} ± {stats_bc['std']:.1f}")
    agente.salvar(os.path.join(dir_saida, 'il_bc.pt'))

    stats_dag = None

    # -- Fase 2: DAgger --------------------------------------------------------
    if not args.so_bc:
        agente.treinar_dagger(
            episodios,
            env_wrapped,
            bot,
            n_rounds=args.n_rounds_dagger,
            passos_por_round=args.passos_dagger,
            epocas_por_round=args.epocas_dagger,
            env_aval=env_wrapped,
            n_eps_aval=args.n_aval,
        )
        stats_dag = agente.avaliar(env_wrapped, args.n_aval * 2)
        print(f"\n[DAgger] avaliação final: {stats_dag['media']:.1f} ± {stats_dag['std']:.1f}")
        agente.salvar(os.path.join(dir_saida, 'il_dagger.pt'))

    # -- Baseline random -------------------------------------------------------
    print("\n[random] calculando baseline...")
    stats_rnd = _baseline_random(env_wrapped, args.n_aval)

    t_total = time.time() - t_inicio
    print(f"\n[tempo total] {t_total:.0f}s ({t_total/60:.1f} min)")

    _resumo_final(stats_bc, stats_dag, stats_rnd)
