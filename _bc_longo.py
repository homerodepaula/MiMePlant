"""Controle BC-LONGO (resposta ao critique #4).

Treina BC por 170 épocas (= 100 BC + 7x10 do DAgger) sobre as MESMAS demos,
para 5 configs x 5 seeds. Avalia in-dist + dry + humid. Salva em
results/experiments/il_bc_longo/det/<config>/seed_<N>/metricas.json.

Objetivo: testar se o ganho do DAgger sobre o BC vem da CORREÇÃO de covariate
shift (dados agregados) ou apenas de MAIS TREINO (mais épocas). Se BC-170 ~ BC-100
e << DAgger, o ganho do DAgger é dos dados, não das épocas.
"""
import os, sys, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
# Reaproveita toda a infra do run_il_experiments
import run_il_experiments as R
from bc import AgenteBC
from rede_lstm import ConjuntoDadosBC
from env import AmbienteFazendaGym
from mascara_acoes import InvolucroMascaraAcoes
from bot_perfeito import BotPerfeito

CONFIGS = ['baseline', 'lr_baixo', 'lr_alto', 'lstm_menor', 'seq_longa']
SEEDS = [42, 123, 2024, 7, 1337]
N_EPOCAS = 170  # = 100 (BC) + 7*10 (DAgger) -> mesmo orçamento de épocas


def main():
    episodios = None
    import glob
    padrao = os.path.join(RAIZ, 'results', 'bot_perfeito', '**', 'trajetorias_especialista.json')
    arq = sorted(glob.glob(padrao, recursive=True))[-1]
    raw = json.load(open(arq, encoding='utf-8'))
    episodios = [[{
        'obs': np.array(t['obs'], dtype=np.float32),
        'acao': int(t['acao']),
        'mascara': np.array(t['mascara'], dtype=np.float32),
    } for t in ep] for ep in raw]
    print(f'[BC-longo] {len(episodios)} eps, {sum(len(e) for e in episodios)} transicoes | {N_EPOCAS} epocas')

    env = InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=365))

    for cfg in CONFIGS:
        hp = R.IL_CONFIGS[cfg]
        for seed in SEEDS:
            dir_out = os.path.join(RAIZ, 'results', 'experiments', 'il_bc_longo', 'det', cfg, f'seed_{seed}')
            if os.path.exists(os.path.join(dir_out, 'metricas.json')):
                print(f'  PULANDO {cfg}/seed_{seed} (ja existe)'); continue
            t0 = time.time()
            import random as _rnd
            _rnd.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

            agente = AgenteBC(dim_lstm=hp['dim_lstm'])
            agente.calcular_normalizacao(episodios)
            ds = ConjuntoDadosBC(episodios, agente.obs_media, agente.obs_std, hp['len_seq'])
            ld = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
            otim = torch.optim.Adam(agente.rede.parameters(), lr=hp['lr'], weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=N_EPOCAS, eta_min=hp['lr']*0.01)
            acur_final = 0.0
            for ep in range(1, N_EPOCAS+1):
                agente.rede.train(); acertos=tot=0
                for o,m,a in ld:
                    o,m,a = o.to(agente.dispositivo),m.to(agente.dispositivo),a.to(agente.dispositivo)
                    hid = agente.rede.inicializar_hidden(o.size(0),agente.dispositivo)
                    lg,_ = agente.rede(o,m,hid); B,T,A = lg.shape
                    loss = F.cross_entropy(lg.reshape(B*T,A), a.reshape(B*T))
                    otim.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(agente.rede.parameters(),1.0); otim.step()
                    acertos += (lg.argmax(-1)==a).sum().item(); tot += B*T
                sched.step(); acur_final = acertos/tot

            # Avaliacao 3 regimes
            r_in, dur, aco, kpi = R._avaliar_agente(agente, env, R.N_AVAL, seed_eval=R.SEED_EVAL)
            r_dry,_,_,_ = R._avaliar_agente(agente, env, R.N_AVAL, seed_eval=R.SEED_EVAL, ood_modo='dry')
            r_hum,_,_,_ = R._avaliar_agente(agente, env, R.N_AVAL, seed_eval=R.SEED_EVAL, ood_modo='humid')
            R._restaurar_clima_normal(env)

            os.makedirs(dir_out, exist_ok=True)
            m = {
                'recompensas_episodios': r_in,
                'media_recompensa_final': float(np.mean(r_in)),
                'desvio_recompensa_final': float(np.std(r_in)),
                'ood_dry_recompensas': r_dry, 'ood_humid_recompensas': r_hum,
                'media_ood_dry': float(np.mean(r_dry)), 'media_ood_humid': float(np.mean(r_hum)),
                'acuracia_bc_final': float(acur_final),
                'contagem_acoes': aco, 'kpis_agricolas': kpi,
                'nome_agente': 'IL-BC-longo', 'nome_config': cfg, 'n_epocas': N_EPOCAS,
                'media_produtividade': float(np.mean(kpi['produtividade'])) if kpi['produtividade'] else 0,
                'media_uso_quimicos': float(np.mean(kpi['uso_quimicos'])) if kpi['uso_quimicos'] else 0,
                'media_qualidade_solo': float(np.mean(kpi['qualidade_solo'])) if kpi['qualidade_solo'] else 0,
            }
            json.dump(m, open(os.path.join(dir_out,'metricas.json'),'w',encoding='utf-8'), indent=2, ensure_ascii=False)
            print(f'  {cfg}/seed_{seed}: in={np.mean(r_in):.0f} dry={np.mean(r_dry):.0f} hum={np.mean(r_hum):.0f} '
                  f'acur={acur_final:.1%} ({time.time()-t0:.0f}s)')
    print('[BC-longo] CONCLUIDO')


if __name__ == '__main__':
    main()
