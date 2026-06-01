"""Random baseline (piso) - resposta ao critique #11.
Agente de acoes aleatorias (entre as validas), 50 eps, 3 regimes. Mesma seed_eval.
Salva em results/experiments/random/avaliacao.json.
"""
import os, sys, json
import numpy as np
RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
import run_il_experiments as R
from env import AmbienteFazendaGym
from mascara_acoes import InvolucroMascaraAcoes

N_ACOES = 15

def aval(env, modo, n=50, seed_eval=999):
    recs = []
    for i in range(n):
        obs, info = env.reset(seed=seed_eval+i)
        if modo:
            R._aplicar_clima_ood(env, modo)
        rec, done = 0.0, False
        while not done:
            mask = info.get('mascara_acoes', np.ones(N_ACOES, dtype=np.float32))
            validas = np.where(mask > 0.5)[0]
            a = int(np.random.choice(validas)) if len(validas) else 14
            obs, r, d, t, info = env.step(a); rec += r; done = d or t
        recs.append(rec)
    R._restaurar_clima_normal(env)
    return recs

def main():
    np.random.seed(12345)
    env = InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=365))
    out = {}
    for nome, modo in [('in_dist', None), ('ood_dry','dry'), ('ood_humid','humid')]:
        recs = aval(env, modo)
        out[nome] = {'media': float(np.mean(recs)), 'std': float(np.std(recs)), 'recompensas': recs}
        print(f'  random {nome}: {np.mean(recs):.0f} +/- {np.std(recs):.0f}')
    d = os.path.join(RAIZ,'results','experiments','random'); os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d,'avaliacao.json'),'w',encoding='utf-8'), indent=2, ensure_ascii=False)
    print('[random] salvo')

if __name__ == '__main__':
    main()
