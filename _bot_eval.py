"""Avalia o BotPerfeito sob as mesmas condições do IL (in-dist + OOD dry/humid).

Mesma seed_eval (999) dos agentes IL e do random. Salva em
results/experiments/oraculo/avaliacao.json.
"""
import os, sys, json
import numpy as np

RAIZ    = os.path.dirname(os.path.abspath(__file__))
DIR_IL  = os.path.join(RAIZ, 'agents', 'imitation-learning')
DIR_AMB = os.path.join(RAIZ, 'environment')
DIR_PPO = os.path.join(RAIZ, 'agents', 'ppo-lstm-masked')
sys.path.insert(0, DIR_IL); sys.path.insert(0, DIR_AMB); sys.path.insert(0, DIR_PPO)
for sub in ['plant', 'birds', 'pollinators', 'soil', 'weather',
            'weeds', 'pest', 'cides-fertilizers', 'facilities']:
    sys.path.insert(0, os.path.join(DIR_AMB, sub))

from env import AmbienteFazendaGym
from ppo_lstm_masked import InvolucroMascaraAcoes
from bot_perfeito import BotPerfeito


def aplicar_clima_ood(env_wrapped, modo):
    if modo is None or modo == 'normal':
        return
    base = env_wrapped
    while hasattr(base, 'env'):
        base = base.env
    clima = getattr(base, 'clima', None)
    if clima is None:
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
    if hasattr(clima, 'resetar'):
        clima.resetar()


def env_raw(env_wrapped):
    e = env_wrapped
    while hasattr(e, 'env'):
        e = e.env
    return e


def avaliar(modo=None, n_eps=50, seed_eval=999, imperfeicao=0.14, lazy_steps=8,
            seed_bot=12345):
    env = InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=365))
    # Bot subotimo (mesma configuracao da geracao de demos):
    bot = BotPerfeito(imperfeicao=imperfeicao, lazy_steps=lazy_steps, seed=seed_bot)
    recs = []
    for i in range(n_eps):
        try:
            obs, info = env.reset(seed=seed_eval + i)
        except TypeError:
            obs, info = env.reset()
        aplicar_clima_ood(env, modo)
        bot.resetar_episodio()
        total, done = 0.0, False
        while not done:
            acao = bot.selecionar_acao(env_raw(env))
            obs, r, d, t, info = env.step(acao)
            total += r
            done = d or t
        recs.append(total)
    return {
        'modo': modo or 'in_dist',
        'n_eps': n_eps,
        'seed_eval': seed_eval,
        'recompensas': recs,
        'media': float(np.mean(recs)),
        'std': float(np.std(recs)),
        'min': float(np.min(recs)),
        'max': float(np.max(recs)),
    }


def main():
    print('[Bot Eval] avaliando oráculo em 3 condições (50 eps cada, seed_eval=999)...')

    res_in    = avaliar(modo=None)
    print(f'  in-dist : media={res_in["media"]:.1f} std={res_in["std"]:.1f}')

    res_dry   = avaliar(modo='dry')
    print(f'  OOD dry : media={res_dry["media"]:.1f} std={res_dry["std"]:.1f}')

    res_humid = avaliar(modo='humid')
    print(f'  OOD humid : media={res_humid["media"]:.1f} std={res_humid["std"]:.1f}')

    dir_out = os.path.join(RAIZ, 'results', 'experiments', 'oraculo')
    os.makedirs(dir_out, exist_ok=True)
    with open(os.path.join(dir_out, 'avaliacao.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'in_dist':  res_in,
            'ood_dry':  res_dry,
            'ood_humid': res_humid,
        }, f, indent=2, ensure_ascii=False)
    print(f'Salvo: {dir_out}/avaliacao.json')


if __name__ == '__main__':
    main()
