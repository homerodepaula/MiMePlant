#!/usr/bin/env python3
"""
Experimento de Hiperparametros -- Imitation Learning (BC + DAgger)

3 hiperparametros variados (um de cada vez, baseline fixo nos demais):
  1. lr       -- taxa de aprendizado BC     (afeta velocidade/estabilidade)
  2. dim_lstm -- dimensao oculta do LSTM    (afeta capacidade de memoria)
  3. len_seq  -- comprimento da sequencia   (afeta contexto temporal visto)

5 configuracoes:
  baseline   : lr=3e-4, dim_lstm=512, len_seq=32  (configuracao padrao)
  lr_baixo   : lr=5e-5, dim_lstm=512, len_seq=32  (lr menor)
  lr_alto    : lr=1e-3, dim_lstm=512, len_seq=32  (lr maior)
  lstm_menor : lr=3e-4, dim_lstm=256, len_seq=32  (LSTM com metade da capacidade)
  seq_longa  : lr=3e-4, dim_lstm=512, len_seq=64  (sequencias 2x mais longas)

Resultados salvos em:
  results/experiments/il_bc/{config}/     (so Behavioral Cloning)
  results/experiments/il_dagger/{config}/ (BC + DAgger)

Metricas salvas em formato compativel com run_experiments.py para
comparacao unificada entre todos os agentes.
"""

import sys, os, time, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

RAIZ    = os.path.dirname(os.path.abspath(__file__))
DIR_IL  = os.path.join(RAIZ, 'agents', 'imitation-learning')
DIR_AMB = os.path.join(RAIZ, 'environment')
DIR_PPO = os.path.join(RAIZ, 'agents', 'ppo-lstm-masked')

sys.path.insert(0, DIR_IL)
sys.path.insert(0, DIR_AMB)
sys.path.insert(0, DIR_PPO)
for sub in ['plant', 'birds', 'pollinators', 'soil', 'weather',
            'weeds', 'pest', 'cides-fertilizers', 'facilities']:
    sys.path.insert(0, os.path.join(DIR_AMB, sub))

from env import AmbienteFazendaGym
from ppo_lstm_masked import InvolucroMascaraAcoes
from bot_perfeito import BotPerfeito
from il_agent import AgenteBCLSTM, OBS_DIM, N_ACOES

# ── Configuracao geral ────────────────────────────────────────────────────────
PASSOS_MAXIMOS   = 365
# Protocolo v2.0: valores padrão; podem ser sobrescritos via CLI
N_EPOCAS_BC      = 100   # epocas BC (default protocolo v2.0)
N_ROUNDS_DAGGER  = 7     # rounds DAgger
PASSOS_DAG_ROUND = 3650  # 10 episodios de 365 passos
EPOCAS_DAG_ROUND = 10    # epocas de fine-tuning por round
N_AVAL           = 50    # episodios de avaliacao no ambiente real
JANELA_FINAL     = 50
SEED_EVAL        = 999   # seed fixo para avaliação (mesmo para todos os agentes)
NOMES_ACOES = [
    'plantar', 'colher', 'regar', 'fertilizar_N', 'fertilizar_P',
    'fertilizar_K', 'fertilizar_C', 'herbicida', 'pesticida',
    'espantalho_basico', 'espantalho_avancado', 'remover_espantalho',
    'colocar_cerca', 'observar', 'esperar'
]

# ── 3 Hiperparametros variados — 5 configuracoes ─────────────────────────────
#
#  Hiperparametro   | baseline | lr_baixo | lr_alto  | lstm_menor | seq_longa
#  lr               |   3e-4   |   5e-5   |   1e-3   |    3e-4    |   3e-4
#  dim_lstm         |   512    |   512    |   512    |    256     |   512
#  len_seq          |    32    |    32    |    32    |     32     |    64
#
IL_CONFIGS = {
    'baseline':   dict(lr=3e-4, dim_lstm=512, len_seq=32),
    'lr_baixo':   dict(lr=5e-5, dim_lstm=512, len_seq=32),
    'lr_alto':    dict(lr=1e-3, dim_lstm=512, len_seq=32),
    'lstm_menor': dict(lr=3e-4, dim_lstm=256, len_seq=32),
    'seq_longa':  dict(lr=3e-4, dim_lstm=512, len_seq=64),
}
NOMES_CONFIGS = list(IL_CONFIGS.keys())

# ── Cores e rotulos ───────────────────────────────────────────────────────────
CORES_CONFIGS = {
    'baseline':   '#2196F3',
    'lr_baixo':   '#4CAF50',
    'lr_alto':    '#F44336',
    'lstm_menor': '#FF9800',
    'seq_longa':  '#9C27B0',
}
EXIBICAO_CONFIGS = {
    'baseline':   'baseline  (lr=3e-4, lstm=512, seq=32)',
    'lr_baixo':   'lr_baixo  (lr=5e-5, lstm=512, seq=32)',
    'lr_alto':    'lr_alto   (lr=1e-3, lstm=512, seq=32)',
    'lstm_menor': 'lstm_menor(lr=3e-4, lstm=256, seq=32)',
    'seq_longa':  'seq_longa (lr=3e-4, lstm=512, seq=64)',
}


# ── Utilitarios ───────────────────────────────────────────────────────────────
def _estilizar_eixo(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _aplicar_clima_ood(env_wrapped, modo):
    """Após reset, ajusta parâmetros do clima do env para simular OOD.

    Modos suportados:
      'dry'   : alta temperatura, pouca chuva, baixa umidade do ar
      'humid' : temperatura moderada, muita chuva, alta umidade
      None    : sem modificação (in-distribution)

    Define clima.modo_ood para que _gerar_clima use os *_base diretamente
    em CADA step (persistência da modificação ao longo do episódio).
    """
    base = env_wrapped
    while hasattr(base, 'env'):
        base = base.env
    clima = getattr(base, 'clima', None)
    if clima is None:
        return
    if modo is None or modo == 'normal':
        # Defensivamente garantir que estamos em baseline (sem OOD pendente).
        clima.temperatura_base         = 20.0
        clima.umidade_ar_base          = 0.5
        clima.probabilidade_chuva_base = 0.3
        clima.modo_ood = None
        if hasattr(clima, 'resetar'):
            clima.resetar()
        return
    if modo == 'dry':
        clima.temperatura_base       = 32.0
        clima.umidade_ar_base        = 0.25
        clima.probabilidade_chuva_base = 0.05
        clima.modo_ood = 'dry'
    elif modo == 'humid':
        clima.temperatura_base       = 20.0
        clima.umidade_ar_base        = 0.85
        clima.probabilidade_chuva_base = 0.70
        clima.modo_ood = 'humid'
    # Regenerar clima inicial com novos parâmetros
    if hasattr(clima, 'resetar'):
        clima.resetar()


def _avaliar_agente(agente, env_wrapped, n_eps, seed_eval=None, ood_modo=None):
    """Avalia o agente para n_eps episodios; retorna (recompensas, duracoes, contagem_acoes, kpis).

    Args:
        seed_eval: se fornecido, reseta o env com seed_eval + i a cada episódio
                    (avaliação reprodutível, igual para todos os agentes).
        ood_modo: 'dry' | 'humid' | None. Aplica clima OOD após cada reset.
    """
    recompensas, duracoes = [], []
    contagem = defaultdict(int)
    kpis_ep = {
        'produtividade': [],
        'qualidade_solo': [],
        'uso_quimicos': [],
        'biodiversidade': []
    }
    for i in range(n_eps):
        if seed_eval is not None:
            try:
                obs, info = env_wrapped.reset(seed=int(seed_eval) + i)
            except TypeError:
                env_wrapped.reset()
                obs, info = env_wrapped.reset()
        else:
            obs, info = env_wrapped.reset()
        # Aplicar clima OOD se solicitado
        if ood_modo is not None:
            _aplicar_clima_ood(env_wrapped, ood_modo)
            # Re-obter obs após mudança climática (método correto é _obter_observacao)
            if hasattr(env_wrapped.env, '_obter_observacao'):
                obs = env_wrapped.env._obter_observacao()
        hidden = agente.rede.inicializar_hidden(1, agente.dispositivo)
        rec, dur = 0.0, 0
        done = False
        while not done:
            mask = info.get('mascara_acoes', np.ones(N_ACOES, dtype=np.float32))
            acao, hidden = agente.selecionar_acao(obs, mask, hidden)
            contagem[NOMES_ACOES[acao]] += 1
            obs, r, done_f, trunc, info = env_wrapped.step(acao)
            rec += r; dur += 1
            done = done_f or trunc
            if done and 'kpis' in info:
                kpis_ep['produtividade'].append(info['kpis']['produtividade_acumulada'])
                kpis_ep['qualidade_solo'].append(info['kpis']['qualidade_solo_media'])
                kpis_ep['uso_quimicos'].append(info['kpis']['uso_quimicos_total'])
                kpis_ep['biodiversidade'].append(info['kpis']['densidade_polinizadores'])
        recompensas.append(rec)
        duracoes.append(dur)
    return recompensas, duracoes, dict(contagem), kpis_ep


def _salvar_graficos_il(metricas_bc, metricas_dag, nome_cfg, hp, dir_bc, dir_dag):
    """Gera e salva graficos de treinamento e avaliacao para uma configuracao."""
    fig, eixos = plt.subplots(2, 3, figsize=(18, 10))
    titulo = (f'IL — {nome_cfg}  '
              f'(lr={hp["lr"]:.0e}, lstm={hp["dim_lstm"]}, seq={hp["len_seq"]})')
    fig.suptitle(titulo, fontsize=13, fontweight='bold', y=0.98)

    # BC loss curve
    ax = eixos[0, 0]
    perdas = metricas_bc.get('perdas_politica', [])
    ax.plot(perdas, color='#2196F3', linewidth=1.5)
    _estilizar_eixo(ax, 'Perda BC por Epoca', 'Epoca', 'CrossEntropy')

    # BC accuracy curve
    ax = eixos[0, 1]
    acurs = metricas_bc.get('acuracoes_bc', [])
    ax.plot([a * 100 for a in acurs], color='#4CAF50', linewidth=1.5)
    ax.set_ylim(0, 105)
    _estilizar_eixo(ax, 'Acuracia BC por Epoca (%)', 'Epoca', 'Acuracia (%)')

    # Avaliacao BC vs DAgger
    ax = eixos[0, 2]
    r_bc  = metricas_bc['recompensas_episodios']
    r_dag = metricas_dag['recompensas_episodios']
    ax.hist(r_bc,  bins=15, alpha=0.6, color='#2196F3', label=f'BC  m={np.mean(r_bc):.0f}')
    ax.hist(r_dag, bins=15, alpha=0.6, color='#E91E63', label=f'DAg m={np.mean(r_dag):.0f}')
    ax.legend(fontsize=9)
    _estilizar_eixo(ax, 'Distribuicao de Recompensas (Avaliacao)', 'Recompensa', 'Contagem')

    # DAgger loss por round
    ax = eixos[1, 0]
    perdas_dag = metricas_dag.get('perdas_politica', [])
    if perdas_dag:
        ax.plot(perdas_dag, color='#E91E63', linewidth=1.5)
    _estilizar_eixo(ax, 'Perda DAgger por Round', 'Round', 'CrossEntropy')

    # BC vs DAgger recompensa por episodio
    ax = eixos[1, 1]
    ax.plot(r_bc,  'o-', color='#2196F3', alpha=0.7, markersize=3, label='BC')
    ax.plot(r_dag, 's-', color='#E91E63', alpha=0.7, markersize=3, label='DAgger')
    ax.axhline(np.mean(r_bc),  linestyle='--', color='#2196F3', linewidth=1.5, alpha=0.6)
    ax.axhline(np.mean(r_dag), linestyle='--', color='#E91E63', linewidth=1.5, alpha=0.6)
    ax.legend(fontsize=9)
    _estilizar_eixo(ax, 'Recompensa por Episodio (Avaliacao)', 'Episodio', 'Recompensa')

    # Distribuicao de acoes (DAgger)
    ax = eixos[1, 2]
    acoes = metricas_dag.get('contagem_acoes', {})
    if acoes:
        nomes_ord = sorted(acoes.keys(), key=lambda k: acoes[k], reverse=True)
        vals = [acoes[n] for n in nomes_ord]
        ax.barh(range(len(nomes_ord)), vals, color='#607D8B', alpha=0.8)
        ax.set_yticks(range(len(nomes_ord)))
        ax.set_yticklabels(nomes_ord, fontsize=7)
    _estilizar_eixo(ax, 'Distribuicao de Acoes (Avaliacao DAgger)', 'Contagem', '')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    for d in [dir_bc, dir_dag]:
        fig.savefig(os.path.join(d, 'graficos.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _novas_metricas_il():
    return {
        'recompensas_episodios': [],
        'duracao_episodios': [],
        'perdas_politica': [],   # BC loss por epoca
        'perdas_valor': [],      # N/A para IL
        'perdas_entropia': [],   # N/A para IL
        'acuracoes_bc': [],      # acuracia BC por epoca
        'contagem_acoes': defaultdict(int),
        'kpis_agricolas': {
            'produtividade': [],
            'qualidade_solo': [],
            'uso_quimicos': [],
            'biodiversidade': []
        }
    }


def _finalizar_metricas_il(metricas, hp, nome_agente, nome_config, t_treino, t_aval):
    r = metricas['recompensas_episodios']
    r_final = r[-JANELA_FINAL:] if len(r) >= JANELA_FINAL else r
    metricas['nome_config']                = nome_config
    metricas['nome_agente']               = nome_agente
    metricas['hiperparametros_variados']  = hp
    metricas['hiperparametros_fixos']     = {
        'n_epocas_bc': N_EPOCAS_BC, 'n_rounds_dagger': N_ROUNDS_DAGGER,
        'passos_dag': PASSOS_DAG_ROUND, 'epocas_dag': EPOCAS_DAG_ROUND,
        'obs_dim': OBS_DIM, 'n_acoes': N_ACOES,
        'batch_size': 64, 'weight_decay': 1e-5, 'grad_clip': 1.0,
        'n_camadas_lstm': 2, 'dropout': 0.1,
    }
    metricas['num_episodios']               = N_AVAL
    metricas['passos_maximos']              = PASSOS_MAXIMOS
    metricas['tempo_treinamento_segundos']  = t_treino
    metricas['tempo_avaliacao_segundos']    = t_aval
    metricas['media_recompensa_final']      = float(np.mean(r_final))
    metricas['desvio_recompensa_final']     = float(np.std(r_final))
    metricas['melhor_recompensa']           = float(np.max(r)) if r else 0
    metricas['pior_recompensa']             = float(np.min(r)) if r else 0
    metricas['acuracia_bc_final']           = float(metricas['acuracoes_bc'][-1]) if metricas['acuracoes_bc'] else 0
    metricas['perda_bc_final']              = float(metricas['perdas_politica'][-1]) if metricas['perdas_politica'] else 0
    
    # Médias dos KPIs Agrícolas
    for kpi, valores in metricas['kpis_agricolas'].items():
        if valores:
            metricas[f'media_{kpi}'] = float(np.mean(valores))
            
    for k in list(metricas.keys()):
        if isinstance(metricas[k], defaultdict):
            metricas[k] = dict(metricas[k])
    return metricas


# ── Executor principal de uma config ─────────────────────────────────────────
def executar_config(nome_cfg, hp, dir_resultados, env_wrapped, bot, episodios_expert,
                     seed=42, exp_categoria='det', exp_valor=0.0, exp_decay=False,
                     exp_eps0=0.1):
    """
    Args:
        exp_categoria: 'det' | 'softmax' | 'epsilon_fixo' | 'epsilon_decay'
                       | 'softmax_eps_decay'
        exp_valor: τ (softmax/softmax_eps_decay) ou ε₀ (epsilon_*). Ignorado em det.
        exp_decay: marcação interna (True se categoria *_decay)
        exp_eps0: ε₀ inicial quando categoria=softmax_eps_decay (combinada)
    """
    # Diretórios organizados por categoria de exploração, valor e seed
    if exp_categoria == 'det':
        rotulo = 'det'
    elif exp_categoria == 'softmax_eps_decay':
        rotulo = f'softmax_eps_decay_tau{exp_valor:g}_eps{exp_eps0:g}'
    else:
        rotulo = f'{exp_categoria}_{exp_valor:g}' + ('_decay' if exp_decay else '')
    sub = f'seed_{seed}'
    dir_bc  = os.path.join(dir_resultados, 'il_bc',     rotulo, nome_cfg, sub)
    dir_dag = os.path.join(dir_resultados, 'il_dagger', rotulo, nome_cfg, sub)
    os.makedirs(dir_bc,  exist_ok=True)
    os.makedirs(dir_dag, exist_ok=True)

    # ── Fixar TODAS as fontes de aleatoriedade ────────────────────────────────
    import random as _random
    import torch
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # determinismo na GPU (custa ~5% velocidade mas garante reprodutibilidade)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f'\n  seed={seed}  lr={hp["lr"]:.0e} | dim_lstm={hp["dim_lstm"]} | len_seq={hp["len_seq"]}')

    # Criar agente com dim_lstm da config
    agente = AgenteBCLSTM(dim_lstm=hp['dim_lstm'])
    agente.calcular_normalizacao(episodios_expert)

    metricas_bc  = _novas_metricas_il()
    metricas_dag = _novas_metricas_il()

    # ── Fase 1: Behavioral Cloning ───────────────────────────────────────────
    print(f'  [BC] {N_EPOCAS_BC} epocas...')
    t0_bc = time.time()

    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from il_agent import ConjuntoDadosBC

    dataset_bc = ConjuntoDadosBC(episodios_expert, agente.obs_media, agente.obs_std, hp['len_seq'])
    loader_bc  = DataLoader(dataset_bc, batch_size=64, shuffle=True, drop_last=True)
    otim_bc    = torch.optim.Adam(agente.rede.parameters(), lr=hp['lr'], weight_decay=1e-5)
    sched_bc   = torch.optim.lr_scheduler.CosineAnnealingLR(otim_bc, T_max=N_EPOCAS_BC, eta_min=hp['lr'] * 0.01)

    for epoca in range(1, N_EPOCAS_BC + 1):
        agente.rede.train()
        perdas, acertos, total = [], 0, 0
        for obs_seq, mask_seq, acao_seq in loader_bc:
            obs_seq  = obs_seq.to(agente.dispositivo)
            mask_seq = mask_seq.to(agente.dispositivo)
            acao_seq = acao_seq.to(agente.dispositivo)
            hidden = agente.rede.inicializar_hidden(obs_seq.size(0), agente.dispositivo)
            logits, _ = agente.rede(obs_seq, mask_seq, hidden)
            B, T, A = logits.shape
            loss = F.cross_entropy(logits.view(B * T, A), acao_seq.view(B * T))
            otim_bc.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(agente.rede.parameters(), 1.0)
            otim_bc.step()
            perdas.append(loss.item())
            acertos += (logits.argmax(-1) == acao_seq).sum().item()
            total   += B * T
        sched_bc.step()
        perda_ep = float(np.mean(perdas))
        acur_ep  = acertos / total
        metricas_bc['perdas_politica'].append(perda_ep)
        metricas_bc['acuracoes_bc'].append(acur_ep)
        if epoca % 10 == 0 or epoca == 1:
            print(f'    ep {epoca:2d}/{N_EPOCAS_BC} | perda={perda_ep:.4f} | acur={acur_ep:.1%}')

    t_bc = time.time() - t0_bc
    print(f'  [BC] concluido em {t_bc:.1f}s')

    # Avaliar BC
    t0_aval = time.time()
    r_bc, dur_bc, acoes_bc, kpis_bc = _avaliar_agente(agente, env_wrapped, N_AVAL, seed_eval=SEED_EVAL)
    t_aval_bc = time.time() - t0_aval
    metricas_bc['recompensas_episodios'] = r_bc
    metricas_bc['duracao_episodios']     = dur_bc
    metricas_bc['contagem_acoes']        = acoes_bc
    metricas_bc['kpis_agricolas']        = kpis_bc
    print(f'  [BC aval] media={np.mean(r_bc):.1f} std={np.std(r_bc):.1f} em {t_aval_bc:.1f}s')

    # Salvar checkpoint BC
    agente.salvar(os.path.join(dir_bc, 'modelo_final.pt'))
    _finalizar_metricas_il(metricas_bc, hp, 'IL-BC', nome_cfg, t_bc, t_aval_bc)
    with open(os.path.join(dir_bc, 'metricas.json'), 'w', encoding='utf-8') as f:
        json.dump(metricas_bc, f, indent=2, ensure_ascii=False)
    with open(os.path.join(dir_bc, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'agente': 'il_bc', 'config': nome_cfg,
                   'hiperparametros_variados': hp}, f, indent=2, ensure_ascii=False)

    # ── Fase 2: DAgger ───────────────────────────────────────────────────────
    print(f'  [DAgger] {N_ROUNDS_DAGGER} rounds x {PASSOS_DAG_ROUND} passos...')
    t0_dag = time.time()
    episodios_agg = list(episodios_expert)

    for rnd in range(1, N_ROUNDS_DAGGER + 1):
        beta = 0.5 * (1 - (rnd - 1) / max(N_ROUNDS_DAGGER - 1, 1))
        # ── Exploração na coleta DAgger ────────────────────────────────────
        # exp_categoria define qual tipo de exploração; round k começa em 0
        k = rnd - 1
        tau_round = 0.0
        eps_round = 0.0
        if exp_categoria == 'softmax':
            tau_round = float(exp_valor)                      # τ fixo
        elif exp_categoria == 'epsilon_fixo':
            eps_round = float(exp_valor)                      # ε fixo
        elif exp_categoria == 'epsilon_decay':
            # ε(k) = ε₀ × (0.5)^k
            eps_round = float(exp_valor) * (0.5 ** k)
        elif exp_categoria == 'softmax_eps_decay':
            # AMBOS decaem exponencialmente: τ(k) = τ₀·0.5^k, ε(k) = ε₀·0.5^k
            tau_round = float(exp_valor) * (0.5 ** k)
            eps_round = float(exp_eps0)   * (0.5 ** k)
        t0_col = time.time()
        novos_eps = agente._coletar_dagger(
            env_wrapped, bot, PASSOS_DAG_ROUND, beta=beta,
            temperatura=tau_round, epsilon=eps_round,
        )
        episodios_agg.extend(novos_eps)
        n_novos = sum(len(e) for e in novos_eps)
        n_total = sum(len(e) for e in episodios_agg)

        # Dataset agregado
        tam_pseudo = hp['len_seq'] * 10
        pseudo_eps = []
        trans_flat = [t for ep in episodios_agg for t in ep]
        for ini in range(0, len(trans_flat) - tam_pseudo + 1, tam_pseudo):
            pseudo_eps.append(trans_flat[ini : ini + tam_pseudo])
        if not pseudo_eps:
            pseudo_eps = [trans_flat]

        dataset_dag = ConjuntoDadosBC(pseudo_eps, agente.obs_media, agente.obs_std, hp['len_seq'])
        loader_dag  = DataLoader(dataset_dag, batch_size=64, shuffle=True, drop_last=True)
        otim_dag    = torch.optim.Adam(agente.rede.parameters(), lr=1e-4, weight_decay=1e-5)

        perdas_r, acertos_r, total_r = [], 0, 0
        agente.rede.train()
        for _ in range(EPOCAS_DAG_ROUND):
            for obs_seq, mask_seq, acao_seq in loader_dag:
                obs_seq  = obs_seq.to(agente.dispositivo)
                mask_seq = mask_seq.to(agente.dispositivo)
                acao_seq = acao_seq.to(agente.dispositivo)
                hidden = agente.rede.inicializar_hidden(obs_seq.size(0), agente.dispositivo)
                logits, _ = agente.rede(obs_seq, mask_seq, hidden)
                B, T, A = logits.shape
                loss = F.cross_entropy(logits.view(B * T, A), acao_seq.view(B * T))
                otim_dag.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agente.rede.parameters(), 1.0)
                otim_dag.step()
                perdas_r.append(loss.item())
                acertos_r += (logits.argmax(-1) == acao_seq).sum().item()
                total_r   += B * T

        perda_rnd = float(np.mean(perdas_r))
        acur_rnd  = acertos_r / total_r
        metricas_dag['perdas_politica'].append(perda_rnd)
        metricas_dag['acuracoes_bc'].append(acur_rnd)
        t_col = time.time() - t0_col
        print(f'    round {rnd}/{N_ROUNDS_DAGGER} | beta={beta:.2f} | +{n_novos} trans '
              f'| total={n_total} | perda={perda_rnd:.4f} | {t_col:.0f}s')

    t_dag = time.time() - t0_dag

    # Avaliar DAgger
    t0_aval2 = time.time()
    r_dag, dur_dag, acoes_dag, kpis_dag = _avaliar_agente(agente, env_wrapped, N_AVAL, seed_eval=SEED_EVAL)
    t_aval_dag = time.time() - t0_aval2
    metricas_dag['recompensas_episodios'] = r_dag
    metricas_dag['duracao_episodios']     = dur_dag
    metricas_dag['contagem_acoes']        = acoes_dag
    metricas_dag['kpis_agricolas']        = kpis_dag
    print(f'  [DAgger aval] media={np.mean(r_dag):.1f} std={np.std(r_dag):.1f} em {t_aval_dag:.1f}s')

    # ── Avaliação OOD (dry e humid) ────────────────────────────────────────
    print(f'  [OOD dry]   avaliando 50 eps...')
    r_bc_dry, _, _, _ = _avaliar_agente(agente, env_wrapped, N_AVAL,
                                          seed_eval=SEED_EVAL, ood_modo='dry')
    # Recarregar pesos BC para avaliar BC em OOD também
    # (não — após DAgger o agente é o DAgger; o BC já foi avaliado)
    metricas_dag['ood_dry_recompensas']   = r_bc_dry
    print(f'    media={np.mean(r_bc_dry):.1f} std={np.std(r_bc_dry):.1f}')

    print(f'  [OOD humid] avaliando 50 eps...')
    r_dag_humid, _, _, _ = _avaliar_agente(agente, env_wrapped, N_AVAL,
                                             seed_eval=SEED_EVAL, ood_modo='humid')
    metricas_dag['ood_humid_recompensas'] = r_dag_humid
    print(f'    media={np.mean(r_dag_humid):.1f} std={np.std(r_dag_humid):.1f}')

    # Restaurar clima normal para próximas iterações (o atributo do clima foi mexido)
    _restaurar_clima_normal(env_wrapped)

    # Salvar DAgger
    agente.salvar(os.path.join(dir_dag, 'modelo_final.pt'))
    _finalizar_metricas_il(metricas_dag, hp, 'IL-DAgger', nome_cfg, t_bc + t_dag, t_aval_dag)
    # Adicionar metadata de exploração ao JSON
    metricas_dag['exp_categoria'] = exp_categoria
    metricas_dag['exp_valor']     = float(exp_valor)
    metricas_dag['exp_decay']     = bool(exp_decay)
    with open(os.path.join(dir_dag, 'metricas.json'), 'w', encoding='utf-8') as f:
        json.dump(metricas_dag, f, indent=2, ensure_ascii=False)
    with open(os.path.join(dir_dag, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'agente': 'il_dagger', 'config': nome_cfg,
                   'exp_categoria': exp_categoria, 'exp_valor': float(exp_valor),
                   'exp_decay': bool(exp_decay),
                   'hiperparametros_variados': hp}, f, indent=2, ensure_ascii=False)

    return metricas_bc, metricas_dag


def _restaurar_clima_normal(env_wrapped):
    """Restaura parâmetros default do clima após avaliação OOD."""
    base = env_wrapped
    while hasattr(base, 'env'):
        base = base.env
    clima = getattr(base, 'clima', None)
    if clima is None:
        return
    clima.temperatura_base         = 20.0
    clima.umidade_ar_base          = 0.5
    clima.probabilidade_chuva_base = 0.3
    clima.modo_ood = None
    if hasattr(clima, 'resetar'):
        clima.resetar()


# ── Graficos comparativos entre configs ──────────────────────────────────────
def gerar_comparacao_il(dir_resultados):
    fig, eixos = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        'Impacto dos 3 Hiperparametros IL (lr, dim_lstm, len_seq)\n'
        'Recompensa Media em Avaliacao (30 episodios)',
        fontsize=14, fontweight='bold'
    )

    medias_bc, medias_dag, stds_bc, stds_dag = {}, {}, {}, {}
    for cfg in NOMES_CONFIGS:
        for agente_dir, medias, stds in [('il_bc', medias_bc, stds_bc), ('il_dagger', medias_dag, stds_dag)]:
            arq = os.path.join(dir_resultados, agente_dir, cfg, 'metricas.json')
            if os.path.exists(arq):
                with open(arq) as f:
                    m = json.load(f)
                r = m['recompensas_episodios']
                medias[cfg] = float(np.mean(r))
                stds[cfg]   = float(np.std(r))

    # Painel 1: BC vs DAgger por config (barras agrupadas)
    ax = eixos[0]
    x = np.arange(len(NOMES_CONFIGS))
    w = 0.35
    cfgs_disp = [c for c in NOMES_CONFIGS if c in medias_bc or c in medias_dag]
    m_bc  = [medias_bc.get(c, 0)  for c in cfgs_disp]
    m_dag = [medias_dag.get(c, 0) for c in cfgs_disp]
    s_bc  = [stds_bc.get(c, 0)   for c in cfgs_disp]
    s_dag = [stds_dag.get(c, 0)  for c in cfgs_disp]
    b1 = ax.bar(x - w/2, m_bc,  w, color='#2196F3', alpha=0.8, label='IL-BC',    yerr=s_bc,  capsize=3)
    b2 = ax.bar(x + w/2, m_dag, w, color='#E91E63', alpha=0.8, label='IL-DAgger', yerr=s_dag, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs_disp, rotation=15, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    _estilizar_eixo(ax, 'Recompensa Media por Config', 'Config', 'Recompensa')

    # Painel 2: Impacto de lr (configs baseline, lr_baixo, lr_alto)
    ax = eixos[1]
    cfgs_lr = ['lr_baixo', 'baseline', 'lr_alto']
    lrs = [IL_CONFIGS[c]['lr'] for c in cfgs_lr if c in IL_CONFIGS]
    m_lr_bc  = [medias_bc.get(c, 0)  for c in cfgs_lr]
    m_lr_dag = [medias_dag.get(c, 0) for c in cfgs_lr]
    ax.plot(lrs, m_lr_bc,  'o-', color='#2196F3', linewidth=2, label='IL-BC',    markersize=8)
    ax.plot(lrs, m_lr_dag, 's-', color='#E91E63', linewidth=2, label='IL-DAgger', markersize=8)
    ax.set_xscale('log')
    for lr_v, m_bc, m_dag in zip(lrs, m_lr_bc, m_lr_dag):
        ax.annotate(f'{m_bc:.0f}',  (lr_v, m_bc),  textcoords='offset points', xytext=(0, 8),  fontsize=8, color='#2196F3')
        ax.annotate(f'{m_dag:.0f}', (lr_v, m_dag), textcoords='offset points', xytext=(0, -15), fontsize=8, color='#E91E63')
    ax.legend(fontsize=9)
    _estilizar_eixo(ax, 'Efeito do Learning Rate (lr)', 'lr', 'Recompensa Media')

    # Painel 3: Ganho DAgger sobre BC por config
    ax = eixos[2]
    ganhos = [(medias_dag.get(c, 0) - medias_bc.get(c, 0)) for c in cfgs_disp]
    cores_barra = ['#4CAF50' if g >= 0 else '#F44336' for g in ganhos]
    barras = ax.bar(range(len(cfgs_disp)), ganhos, color=cores_barra, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(cfgs_disp)))
    ax.set_xticklabels(cfgs_disp, rotation=15, ha='right', fontsize=8)
    for barra, g in zip(barras, ganhos):
        ax.text(barra.get_x() + barra.get_width() / 2,
                barra.get_height() + (2 if g >= 0 else -12),
                f'{g:+.0f}', ha='center', fontsize=9, fontweight='bold')
    _estilizar_eixo(ax, 'Ganho DAgger sobre BC (Delta Recompensa)', 'Config', 'Delta Recompensa')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    saida = os.path.join(dir_resultados, 'comparacao', 'il_comparacao_configs.png')
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    fig.savefig(saida, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvo: {saida}')


def gerar_mapa_calor_il(dir_resultados):
    """Mapa de calor: configs x agentes (BC/DAgger) com recompensa media."""
    matrix_bc  = []
    matrix_dag = []
    for cfg in NOMES_CONFIGS:
        for agente_dir, matrix in [('il_bc', matrix_bc), ('il_dagger', matrix_dag)]:
            arq = os.path.join(dir_resultados, agente_dir, cfg, 'metricas.json')
            if os.path.exists(arq):
                with open(arq) as f:
                    m = json.load(f)
                r = m['recompensas_episodios']
                matrix.append(float(np.mean(r)))
            else:
                matrix.append(float('nan'))

    fig, eixos = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sensibilidade dos Hiperparametros IL\n(lr, dim_lstm, len_seq x Recompensa Media)',
                 fontsize=13, fontweight='bold')

    dados = np.array([matrix_bc, matrix_dag])
    im = eixos[0].imshow(dados, cmap='YlOrRd', aspect='auto')
    eixos[0].set_xticks(range(len(NOMES_CONFIGS)))
    eixos[0].set_xticklabels(NOMES_CONFIGS, rotation=30, ha='right', fontsize=9)
    eixos[0].set_yticks([0, 1])
    eixos[0].set_yticklabels(['IL-BC', 'IL-DAgger'], fontsize=11)
    for i in range(2):
        for j in range(len(NOMES_CONFIGS)):
            v = dados[i, j]
            if not np.isnan(v):
                cor = 'white' if v > np.nanmean(dados) else 'black'
                eixos[0].text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=10, color=cor)
    fig.colorbar(im, ax=eixos[0], shrink=0.8, label='Recompensa Media')
    _estilizar_eixo(eixos[0], 'Mapa de Calor: Configs vs Agentes', '', '')

    # Amplitude por HP variado
    ax = eixos[1]
    grupos = {
        'lr\n(lr_baixo,baseline,lr_alto)': ['lr_baixo', 'baseline', 'lr_alto'],
        'dim_lstm\n(lstm_menor,baseline)': ['lstm_menor', 'baseline'],
        'len_seq\n(baseline,seq_longa)':   ['baseline', 'seq_longa'],
    }
    labels, amp_bc, amp_dag = [], [], []
    for label, cfgs in grupos.items():
        vals_bc  = [matrix_bc[NOMES_CONFIGS.index(c)]  for c in cfgs if c in NOMES_CONFIGS]
        vals_dag = [matrix_dag[NOMES_CONFIGS.index(c)] for c in cfgs if c in NOMES_CONFIGS]
        labels.append(label)
        amp_bc.append(max(vals_bc) - min(vals_bc) if vals_bc else 0)
        amp_dag.append(max(vals_dag) - min(vals_dag) if vals_dag else 0)
    x = np.arange(len(labels))
    ax.bar(x - 0.2, amp_bc,  0.35, label='IL-BC',     color='#2196F3', alpha=0.8)
    ax.bar(x + 0.2, amp_dag, 0.35, label='IL-DAgger', color='#E91E63', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=9)
    _estilizar_eixo(ax, 'Amplitude de Recompensa por Hiperparametro\n(max - min entre configs)', '', 'Amplitude')

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    saida = os.path.join(dir_resultados, 'comparacao', 'il_mapa_calor.png')
    fig.savefig(saida, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Salvo: {saida}')


def gerar_resumo_il(dir_resultados):
    resumo = {'configs': {c: IL_CONFIGS[c] for c in NOMES_CONFIGS}, 'resultados': {}}
    for agente_dir in ['il_bc', 'il_dagger']:
        resumo['resultados'][agente_dir] = {}
        for cfg in NOMES_CONFIGS:
            arq = os.path.join(dir_resultados, agente_dir, cfg, 'metricas.json')
            if not os.path.exists(arq):
                continue
            with open(arq) as f:
                m = json.load(f)
            r = m['recompensas_episodios']
            resumo['resultados'][agente_dir][cfg] = {
                'media': float(np.mean(r)),
                'std': float(np.std(r)),
                'min': float(np.min(r)) if r else 0,
                'max': float(np.max(r)) if r else 0,
                'tempo_treino_s': m.get('tempo_treinamento_segundos', 0),
                'acuracia_bc': m.get('acuracia_bc_final', 0),
            }
    saida = os.path.join(dir_resultados, 'comparacao', 'il_resumo.json')
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    with open(saida, 'w', encoding='utf-8') as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f'  Salvo: {saida}')
    return resumo


# ── Script principal ──────────────────────────────────────────────────────────
def principal():
    ap = argparse.ArgumentParser(description='Experimento IL (BC + DAgger) - Protocolo v3.0 (exploração)')
    ap.add_argument('--config',  type=str, default='baseline', choices=NOMES_CONFIGS,
                    help='Config HP (default: baseline)')
    ap.add_argument('--seeds',   type=int, nargs='+', default=[42, 123, 2024, 7, 1337],
                    help='Seeds a executar')
    ap.add_argument('--exp-categoria', type=str, default='det',
                    choices=['det', 'softmax', 'epsilon_fixo',
                              'epsilon_decay', 'softmax_eps_decay'],
                    help='Categoria de exploração na coleta DAgger')
    ap.add_argument('--exp-valor', type=float, default=0.0,
                    help='τ (softmax/softmax_eps_decay) ou ε₀ (epsilon_*). Ignorado em det.')
    ap.add_argument('--exp-eps0', type=float, default=0.1,
                    help='ε₀ inicial quando categoria=softmax_eps_decay (default: 0.1)')
    ap.add_argument('--retomar', action='store_true',
                    help='Pular runs ja concluidos (com metricas.json)')
    args = ap.parse_args()

    dir_resultados = os.path.join(RAIZ, 'results', 'experiments')
    configs_exec   = [args.config] if args.config else NOMES_CONFIGS
    seeds_exec     = list(args.seeds)

    # Carregar trajetorias (compartilhadas entre todas as configs)
    import glob
    padrao = os.path.join(RAIZ, 'results', 'bot_perfeito', '**', 'trajetorias_especialista.json')
    arquivos = sorted(glob.glob(padrao, recursive=True))
    if not arquivos:
        raise FileNotFoundError('Trajetorias do BotPerfeito nao encontradas.')
    with open(arquivos[-1], encoding='utf-8') as f:
        raw = json.load(f)
    episodios_expert = [[{
        'obs':     np.array(t['obs'],     dtype=np.float32),
        'acao':    int(t['acao']),
        'mascara': np.array(t['mascara'], dtype=np.float32),
    } for t in ep] for ep in raw]
    n_trans = sum(len(e) for e in episodios_expert)
    print(f'[dados] {len(episodios_expert)} episodios, {n_trans} transicoes')

    # Ambiente e bot
    env_base    = AmbienteFazendaGym(passos_maximos=PASSOS_MAXIMOS)
    env_wrapped = InvolucroMascaraAcoes(env_base)
    bot         = BotPerfeito()

    print('=' * 70)
    print('  EXPERIMENTO DE HIPERPARAMETROS — IMITATION LEARNING')
    print(f'  Configs: {configs_exec}')
    print(f'  BC: {N_EPOCAS_BC} epocas | DAgger: {N_ROUNDS_DAGGER} rounds x {PASSOS_DAG_ROUND} passos')
    print(f'  Avaliacao: {N_AVAL} episodios por config')
    print()
    print(f'  {"Config":<12} {"lr":>8} {"dim_lstm":>10} {"len_seq":>10}')
    for cfg, hp in IL_CONFIGS.items():
        marker = ' <-- varia lr' if cfg in ('lr_baixo', 'lr_alto') else \
                 ' <-- varia dim_lstm' if cfg == 'lstm_menor' else \
                 ' <-- varia len_seq'  if cfg == 'seq_longa'  else ''
        print(f'  {cfg:<12} {hp["lr"]:>8.0e} {hp["dim_lstm"]:>10} {hp["len_seq"]:>10}{marker}')
    print('=' * 70)

    t0_total = time.time()
    concluidas = 0
    total = len(configs_exec) * len(seeds_exec)

    # Resolver rótulo da condição experimental
    exp_decay = (args.exp_categoria == 'epsilon_decay')
    if args.exp_categoria == 'det':
        rotulo = 'det'
    elif args.exp_categoria == 'softmax_eps_decay':
        rotulo = f'softmax_eps_decay_tau{args.exp_valor:g}_eps{args.exp_eps0:g}'
    else:
        rotulo = f'{args.exp_categoria}_{args.exp_valor:g}' + ('_decay' if exp_decay else '')
    print(f'  Condição exploratória: {rotulo}')
    print('=' * 70)

    for nome_cfg in configs_exec:
        hp = IL_CONFIGS[nome_cfg]
        for seed in seeds_exec:
            sub = f'seed_{seed}'
            arq_metricas_bc = os.path.join(dir_resultados, 'il_bc', rotulo, nome_cfg, sub, 'metricas.json')

            if args.retomar and os.path.exists(arq_metricas_bc):
                print(f'\n  PULANDO {rotulo}/{nome_cfg} seed={seed} (ja existe)')
                concluidas += 1
                continue

            print(f'\n{"=" * 60}')
            print(f'  [{concluidas+1}/{total}] {rotulo} | CONFIG: {nome_cfg} | SEED: {seed}')
            print(f'{"=" * 60}')

            t0_cfg = time.time()
            executar_config(nome_cfg, hp, dir_resultados, env_wrapped, bot,
                            episodios_expert, seed=seed,
                            exp_categoria=args.exp_categoria,
                            exp_valor=args.exp_valor,
                            exp_decay=exp_decay,
                            exp_eps0=args.exp_eps0)
            t_cfg = time.time() - t0_cfg
            concluidas += 1
            print(f'  {rotulo}/{nome_cfg} seed={seed} concluida em {t_cfg:.0f}s ({t_cfg/60:.1f} min)')

    t_total = time.time() - t0_total
    print(f'\n{"=" * 60}')
    print(f'  Todas as configs concluidas em {t_total:.0f}s ({t_total/60:.1f} min)')
    print(f'{"=" * 60}')

    # No protocolo v3.0, a análise comparativa é feita por _analise.py
    # após todas as condições rodarem (não mais aqui).
    print('\n  Pipeline desta condição concluído. Rode _analise.py após todas as condições.')

    print('\n' + '=' * 70)
    print('  EXPERIMENTO IL CONCLUIDO')
    print('=' * 70)


if __name__ == '__main__':
    principal()
