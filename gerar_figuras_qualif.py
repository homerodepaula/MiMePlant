"""Gera as duas figuras da qualificacao a partir dos metricas.json.
Salva em results/figuras/fig_sensibilidade.png e results/figuras/fig_ood_configs.png.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(RAIZ, 'results', 'figuras')
os.makedirs(SAIDA, exist_ok=True)
CONFIGS = ['baseline', 'lr_baixo', 'lr_alto', 'lstm_menor', 'seq_longa']
SEEDS = [42, 123, 2024, 7, 1337]
BOT = {'in': 7187.1, 'dry': 3832.8, 'humid': 8724.0}


def ler(agente, cfg):
    ins, dry, hum = [], [], []
    for s in SEEDS:
        f = os.path.join(RAIZ, 'results', 'experiments', agente, 'det', cfg, f'seed_{s}', 'metricas.json')
        if not os.path.exists(f):
            continue
        m = json.load(open(f, encoding='utf-8'))
        ins.append(m['media_recompensa_final'])
        if m.get('ood_dry_recompensas'):
            dry.append(np.mean(m['ood_dry_recompensas']))
        if m.get('ood_humid_recompensas'):
            hum.append(np.mean(m['ood_humid_recompensas']))
    return ins, dry, hum


# Coleta
bc_m, bc_s, da_m, da_s = [], [], [], []
ood = {'in': [], 'dry': [], 'hum': []}
for c in CONFIGS:
    bin_, _, _ = ler('il_bc', c)
    din, ddry, dhum = ler('il_dagger', c)
    bc_m.append(np.mean(bin_)); bc_s.append(np.std(bin_))
    da_m.append(np.mean(din));  da_s.append(np.std(din))
    ood['in'].append(np.mean(din)); ood['dry'].append(np.mean(ddry)); ood['hum'].append(np.mean(dhum))

rotulos = ['baseline', 'lr\\_baixo', 'lr\\_alto', 'lstm\\_menor', 'seq\\_longa']
rot = ['baseline', 'lr_baixo', 'lr_alto', 'lstm_menor', 'seq_longa']

# ── Figura 1: sensibilidade BC vs DAgger ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(CONFIGS)); w = 0.38
ax.bar(x - w/2, bc_m, w, yerr=bc_s, capsize=4, color='#90CAF9', label='BC', edgecolor='#1565C0')
ax.bar(x + w/2, da_m, w, yerr=da_s, capsize=4, color='#1565C0', label='DAgger')
ax.axhline(BOT['in'], ls='--', color='green', lw=1.5, label=f"Oráculo ({BOT['in']:.0f})")
ax.set_xticks(x); ax.set_xticklabels(rot, rotation=12, ha='right', fontsize=9)
ax.set_ylabel('Recompensa média (in-distribution)')
ax.set_title('Sensibilidade aos hiperparâmetros: BC vs DAgger')
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 9500)
plt.tight_layout()
f1 = os.path.join(SAIDA, 'fig_sensibilidade.png')
fig.savefig(f1, dpi=150, bbox_inches='tight'); plt.close(fig)
print('salvo:', f1)

# ── Figura 2: OOD por configuração ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(CONFIGS)); w = 0.26
ax.bar(x - w, ood['in'], w, color='#1565C0', label='in-dist')
ax.bar(x, ood['dry'], w, color='#EF6C00', label='dry (seco)')
ax.bar(x + w, ood['hum'], w, color='#2E7D32', label='humid (úmido)')
ax.axhline(BOT['in'], ls='--', color='#1565C0', lw=1, alpha=0.7)
ax.axhline(BOT['dry'], ls='--', color='#EF6C00', lw=1, alpha=0.7)
ax.axhline(BOT['humid'], ls='--', color='#2E7D32', lw=1, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(rot, rotation=12, ha='right', fontsize=9)
ax.set_ylabel('Recompensa média (DAgger)')
ax.set_title('Robustez fora da distribuição por configuração\n(linhas tracejadas = oráculo em cada regime)')
ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, 9500)
plt.tight_layout()
f2 = os.path.join(SAIDA, 'fig_ood_configs.png')
fig.savefig(f2, dpi=150, bbox_inches='tight'); plt.close(fig)
print('salvo:', f2)
print('OK')
