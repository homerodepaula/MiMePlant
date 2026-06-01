"""
Bot Perfeito — Oracle determinístico ótimo para o AmbienteFazendaGym.

Acessa diretamente o estado interno do ambiente para tomar decisões ótimas
sem incerteza nem exploração. Serve como:
  1. Gerador de trajetórias de especialista para Imitation Learning
  2. Upper-bound de recompensa para comparação com agentes RL
  3. Benchmark de desempenho da estrutura de recompensas

Análise do espaço de recompensa:
  - peso_fruto acumulado na fase FRUTIFICACAO é CONSTANTE = 25 por planta
    (a quantidade de steps × taxa por step = limiar de progressão, independente do fator)
  - Logo, o fator de crescimento não muda o rendimento por colheita, mas sim
    a VELOCIDADE do ciclo → mais ciclos em 365 passos = mais recompensa
  - Prioridade: maximizar fator de crescimento → solo rico, sem ervas, sem pragas
  - A recompensa por colheita = 25 × 5.0 = 125 por planta
  - Com 3 ciclos × 16 plantas = 48 colheitas → 6 000 de recompensa de colheita
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, '../../environment'))
for _sub in ['plant', 'birds', 'pollinators', 'soil', 'weather',
             'weeds', 'pest', 'cides-fertilizers', 'facilities']:
    sys.path.insert(0, os.path.join(_DIR, f'../../environment/{_sub}'))

from plant import EstagioPlanta

# ─── Índices de ação ──────────────────────────────────────────────────────────
PLANTAR              = 0
COLHER               = 1
REGAR                = 2
FERTILIZAR_N         = 3
FERTILIZAR_P         = 4
FERTILIZAR_K         = 5
FERTILIZAR_C         = 6
HERBICIDA            = 7
PESTICIDA            = 8
ESP_BASICO           = 9
ESP_AVANCADO         = 10
REMOVER_ESP          = 11
COLOCAR_CERCA        = 12
OBSERVAR             = 13
ESPERAR              = 14

NOMES_ACOES = [
    'plantar', 'colher', 'regar', 'fertilizar_N', 'fertilizar_P',
    'fertilizar_K', 'fertilizar_C', 'herbicida', 'pesticida',
    'espantalho_basico', 'espantalho_avancado', 'remover_espantalho',
    'colocar_cerca', 'observar', 'esperar',
]

# ─── Parâmetros de controle ───────────────────────────────────────────────────
# Watering: soil moisture drops ~0.03-0.05/step; water adds 0.2 → needed every ~6 steps
LIMIAR_UMIDADE          = 0.40   # regar se umidade média das células plantadas < este valor
LIMIAR_CHUVA_ATIVA      = 3.0    # mm de chuva no step atual para considerar "chovendo agora"
LIMIAR_UMIDADE_AR_ALTA  = 0.75   # umidade_ar acima disto sugere chuva iminente
LIMIAR_ERVAS            = 0.15   # herbicida se densidade máxima de ervas > este valor
LIMIAR_DOENCA           = 0.10   # pesticida se impacto máximo de doença > este valor
LIMIAR_NUTRIENTE        = 0.42   # fertilizar se algum nutriente médio < este valor
# Nutrientes: N decai mais rápido com uso (-0.01/step), eficiência alta (0.15/unit aplicado)
# P tem eficiência baixíssima (0.05/unit), K médio (0.12), C médio-baixo (0.08)
PRIORIDADE_NUTRIENTES   = ['N', 'P', 'K', 'C']  # ordem de prioridade quando deficientes
# Threshold para replantio: com crescimento otimizado, ciclo leva ~120-135 steps
PASSOS_MINIMOS_REPLANTE = 125
# Espantalho avançado a cada N passos para manter proteção (decaimento 0.001/step → renova ~2×)
FREQ_ESPANTALHO         = 120


class BotPerfeito:
    """
    Oracle com acesso direto ao estado interno do AmbienteFazendaGym.

    Política hierárquica determinística (em ordem de prioridade):
      1. COLHER    — planta em COLHEITA perde valor se não colhida (morre em ~13 steps)
      2. PLANTAR   — preencher grade imediatamente; replantio se há tempo (≥125 steps)
      3. REGAR     — umidade média das parcelas < 0.40
      4. HERBICIDA — densidade máxima de ervas > 0.15 (reduz fator de crescimento)
      5. PESTICIDA — impacto de doença > 0.10 (penaliza por passo)
      6. FERTILIZAR — nutriente mais deficiente abaixo do limiar (ordem N→P→K→C)
      7. ESPANTALHO — proteção periódica contra pássaros
      8. ESPERAR   — quando tudo está em ordem
    """

    def __init__(self, imperfeicao: float = 0.0, lazy_steps: int = 5,
                  seed: Optional[int] = None):
        """
        Args:
            imperfeicao: probabilidade p ∈ [0,1] por step de o bot entrar em
                modo PREGUIÇOSO (rajada de ESPERARs consecutivos). Cria um
                oráculo "quase ótimo" com lacunas de inação suficientemente
                longas para deixar plantas morrerem, soil secar, etc.
                p=0.0 ⇒ comportamento ótimo (default).
            lazy_steps: nº de steps consecutivos em ESPERAR quando entra em
                modo preguiçoso (default 5).
            seed: seed do RNG interno. Se None, usa np.random global.
        """
        self.imperfeicao = float(imperfeicao)
        self.lazy_steps  = int(lazy_steps)
        self._rng = np.random.RandomState(seed) if seed is not None else None
        self.resetar_episodio()

    def _rand(self) -> float:
        if self._rng is not None:
            return float(self._rng.random())
        return float(np.random.random())

    def resetar_episodio(self):
        self._ultimo_espantalho = -FREQ_ESPANTALHO  # dispara na primeira verificação
        self._lazy_remaining = 0

    # ── Interface principal ───────────────────────────────────────────────────

    def selecionar_acao(self, env_raw) -> int:
        """Retorna o índice da ação ótima dado acesso direto ao ambiente.

        Se self.imperfeicao > 0, periodicamente entra em modo "preguiçoso":
        rajada de ESPERARs por self.lazy_steps steps consecutivos, durante
        os quais plantas em COLHEITA podem morrer, soil seca, ervas crescem
        — i.e., criando lacunas de inação suficientemente longas para impactar.
        """
        if self._lazy_remaining > 0:
            self._lazy_remaining -= 1
            return ESPERAR
        acao = self._selecionar_acao_otima(env_raw)
        if self.imperfeicao > 0.0 and self._rand() < self.imperfeicao:
            self._lazy_remaining = max(0, self.lazy_steps - 1)
            return ESPERAR
        return acao

    def _selecionar_acao_otima(self, env_raw) -> int:
        """Política hierárquica determinística pura (sem imperfeição)."""
        plantas        = env_raw.planta.plantas
        passo          = env_raw.passo_atual
        passos_max     = env_raw.passos_maximos
        linhas, colunas = env_raw.tamanho_grade
        n_celulas       = linhas * colunas
        n_plantas       = len(plantas)
        espacos_vazios  = n_celulas - n_plantas
        passos_restantes = passos_max - passo

        # ── 1. COLHER ─────────────────────────────────────────────────────────
        # Máxima prioridade: planta em COLHEITA começa a morrer em ~13 steps
        for planta_obj in plantas.values():
            if planta_obj.estagio == EstagioPlanta.COLHEITA:
                return COLHER

        # ── 2. PLANTAR ────────────────────────────────────────────────────────
        # Plantar apenas se há espaço E tempo suficiente para outro ciclo completo
        if espacos_vazios > 0 and passos_restantes >= PASSOS_MINIMOS_REPLANTE:
            return PLANTAR

        # ── 3. REGAR ─────────────────────────────────────────────────────────
        # Umidade baixa → growth factor cai → ciclos mais lentos.
        # Domain knowledge: NÃO regar se está chovendo agora OU se umidade do ar
        # está alta (chuva iminente) — a chuva faz o papel da rega naturalmente,
        # liberando o step para colher/plantar/fertilizar.
        if n_plantas > 0 and self._umidade_plantas(env_raw) < LIMIAR_UMIDADE:
            if not self._chovendo_ou_iminente(env_raw):
                return REGAR

        # ── 4. HERBICIDA ──────────────────────────────────────────────────────
        # Ervas: reduzem growth factor em (1 - 0.5 × densidade)
        # Ex: densidade 0.3 → penalidade de 15% no fator de crescimento
        if self._ervas_max(env_raw) > LIMIAR_ERVAS:
            return HERBICIDA

        # ── 5. PESTICIDA ──────────────────────────────────────────────────────
        # Doença: penaliza a recompensa por passo
        if self._doenca_max(env_raw) > LIMIAR_DOENCA:
            return PESTICIDA

        # ── 6. FERTILIZAR ─────────────────────────────────────────────────────
        # Manter nutrientes altos → soil quality alta → growth factor alto
        # N decai em -0.01/step; fertilizar_N aplica 2 units × 0.15 efic. = +0.30 N
        acao_fert = self._acao_fertilizar(env_raw)
        if acao_fert is not None:
            return acao_fert

        # ── 7. ESPANTALHO AVANÇADO ────────────────────────────────────────────
        # Renovar periodicamente para manter força > 1.0
        if n_plantas > 0 and (passo - self._ultimo_espantalho) >= FREQ_ESPANTALHO:
            self._ultimo_espantalho = passo
            return ESP_AVANCADO

        # ── 8. ESPERAR ────────────────────────────────────────────────────────
        return ESPERAR

    # ── Helpers de estado ─────────────────────────────────────────────────────

    def _chovendo_ou_iminente(self, env_raw) -> bool:
        """Retorna True se está chovendo agora (chuva > LIMIAR_CHUVA_ATIVA) ou
        se a umidade do ar está alta (LIMIAR_UMIDADE_AR_ALTA), sugerindo
        chuva iminente. Em ambos os casos, a chuva natural dispensará a rega."""
        clima = getattr(env_raw, 'clima', None)
        if clima is None:
            return False
        atual = getattr(clima, 'clima_atual', None) or {}
        if atual.get('chuva', 0.0) > LIMIAR_CHUVA_ATIVA:
            return True
        if atual.get('umidade_ar', 0.0) > LIMIAR_UMIDADE_AR_ALTA:
            return True
        return False

    def _umidade_plantas(self, env_raw) -> float:
        """Umidade média das células com plantas (as que precisam de água)."""
        plantas = env_raw.planta.plantas
        if not plantas:
            return 1.0
        total = sum(
            env_raw.solo.obter_solo_em(pos[0], pos[1]).umidade
            for pos in plantas
        )
        return total / len(plantas)

    def _ervas_max(self, env_raw) -> float:
        """Densidade máxima de ervas daninhas nas células com plantas."""
        plantas = env_raw.planta.plantas
        if not plantas:
            return 0.0
        return max(
            env_raw.ervas_daninhas.obter_densidade_em(pos[0], pos[1])
            for pos in plantas
        )

    def _doenca_max(self, env_raw) -> float:
        """Impacto máximo de doença/praga nas células com plantas."""
        plantas = env_raw.planta.plantas
        if not plantas:
            return 0.0
        return max(
            env_raw.sistema_doencas.calcular_impacto_doenca(pos[0], pos[1])
            for pos in plantas
        )

    def _acao_fertilizar(self, env_raw) -> Optional[int]:
        """
        Fertiliza o nutriente mais deficiente abaixo do limiar.
        Prioridade: N (maior eficiência e mais crítico) → P → K → C
        """
        linhas, colunas = env_raw.tamanho_grade
        n = linhas * colunas
        medias = {'N': 0.0, 'P': 0.0, 'K': 0.0, 'C': 0.0}

        for x in range(linhas):
            for y in range(colunas):
                solo = env_raw.solo.obter_solo_em(x, y)
                for k in medias:
                    medias[k] += solo.nutrientes.get(k, 0.0)
        for k in medias:
            medias[k] /= n

        mapa = {'N': FERTILIZAR_N, 'P': FERTILIZAR_P, 'K': FERTILIZAR_K, 'C': FERTILIZAR_C}
        # Prioridade: o mais deficiente em relação ao limiar
        deficiente = min(PRIORIDADE_NUTRIENTES, key=lambda k: medias[k])
        if medias[deficiente] < LIMIAR_NUTRIENTE:
            return mapa[deficiente]
        return None


# ─── Treinador / Executor ─────────────────────────────────────────────────────

class ExecutorBot:
    """
    Executa o BotPerfeito, coleta estatísticas e salva trajetórias de especialista
    para uso posterior em Imitation Learning.
    """

    def __init__(self, env, dir_resultados: str, imperfeicao: float = 0.0,
                  lazy_steps: int = 5, seed: Optional[int] = None):
        self.env = env
        self.bot = BotPerfeito(imperfeicao=imperfeicao, lazy_steps=lazy_steps, seed=seed)
        self.dir = os.path.join(dir_resultados, f"bot_perfeito_{int(time.time())}")
        os.makedirs(self.dir, exist_ok=True)

        self.recompensas   = []
        self.colheitas     = []
        self.duracoes      = []
        self.acoes_total   = defaultdict(int)

    def _env_raw(self):
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    def executar(self, num_episodios: int = 100, passos_maximos: int = 365,
                 salvar_trajetorias: bool = True) -> str:

        print(f"\n{'='*60}")
        print(f"  BOT PERFEITO — {num_episodios} episódios × {passos_maximos} passos")
        print(f"  Dir: {self.dir}")
        print(f"{'='*60}\n")

        todas_trajetorias = []
        t0 = time.time()

        for ep in range(num_episodios):
            obs, info = self.env.reset()
            self.bot.resetar_episodio()
            env_raw = self._env_raw()

            rec_ep  = 0.0
            colh_ep = 0
            traj_ep = [] if salvar_trajetorias else None

            for _ in range(passos_maximos):
                acao    = self.bot.selecionar_acao(env_raw)
                obs_ant = obs.copy()
                mask    = info.get('mascara_acoes', np.ones(15, dtype=np.float32))

                obs, r, done, trunc, info = self.env.step(acao)
                rec_ep += r
                self.acoes_total[NOMES_ACOES[acao]] += 1

                if acao == COLHER:
                    colh_ep += 1

                if salvar_trajetorias:
                    traj_ep.append({
                        'obs':      obs_ant.tolist(),
                        'acao':     int(acao),
                        'recompensa': float(r),
                        'prox_obs': obs.tolist(),
                        'feito':    bool(done or trunc),
                        'mascara':  mask.tolist(),
                    })

                if done or trunc:
                    break

            self.recompensas.append(rec_ep)
            self.colheitas.append(colh_ep)
            self.duracoes.append(_ + 1)

            if salvar_trajetorias and traj_ep:
                todas_trajetorias.append(traj_ep)

            if (ep + 1) % 10 == 0:
                ult = self.recompensas[-10:]
                print(f"  Ep {ep+1:4d}/{num_episodios} | "
                      f"Média(10): {np.mean(ult):9.2f} | "
                      f"Última: {rec_ep:9.2f} | "
                      f"Colheitas: {colh_ep:3d}")

        tempo = time.time() - t0

        if salvar_trajetorias and todas_trajetorias:
            n_trans = sum(len(t) for t in todas_trajetorias)
            cam = os.path.join(self.dir, "trajetorias_especialista.json")
            with open(cam, 'w') as f:
                json.dump(todas_trajetorias, f)
            print(f"\n  Trajetórias: {cam}")
            print(f"  {len(todas_trajetorias)} episódios, {n_trans} transições")

        self._salvar_stats(tempo)
        self._gerar_graficos()
        self._resumo()

        return self.dir

    def _salvar_stats(self, tempo: float):
        stats = {
            'recompensas':       self.recompensas,
            'colheitas':         self.colheitas,
            'duracoes':          self.duracoes,
            'acoes_total':       dict(self.acoes_total),
            'tempo_segundos':    tempo,
            'media_recompensa':  float(np.mean(self.recompensas)),
            'melhor_recompensa': float(np.max(self.recompensas)),
            'pior_recompensa':   float(np.min(self.recompensas)),
            'std_recompensa':    float(np.std(self.recompensas)),
            'media_colheitas':   float(np.mean(self.colheitas)),
        }
        with open(os.path.join(self.dir, 'estatisticas.json'), 'w') as f:
            json.dump(stats, f, indent=2)

    def _gerar_graficos(self):
        try:
            fig, eixos = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('Bot Perfeito — Oracle Ótimo', fontsize=14, fontweight='bold')

            recs = self.recompensas
            colh = self.colheitas

            # Recompensa por episódio
            eixos[0].plot(recs, alpha=0.35, linewidth=0.8, color='steelblue')
            mm = [np.mean(recs[max(0, i-10):i+1]) for i in range(len(recs))]
            eixos[0].plot(mm, color='red', linewidth=2, label=f'MM-10')
            eixos[0].axhline(np.mean(recs), color='green', linestyle='--',
                             linewidth=1.5, label=f'Média: {np.mean(recs):.0f}')
            eixos[0].set_title('Recompensa por Episódio')
            eixos[0].set_xlabel('Episódio')
            eixos[0].set_ylabel('Recompensa')
            eixos[0].legend()
            eixos[0].grid(True, alpha=0.3)

            # Colheitas por episódio
            eixos[1].plot(colh, color='darkorange', alpha=0.7, linewidth=1.2)
            eixos[1].axhline(np.mean(colh), color='red', linestyle='--',
                             linewidth=1.5, label=f'Média: {np.mean(colh):.1f}')
            eixos[1].set_title('Colheitas por Episódio')
            eixos[1].set_xlabel('Episódio')
            eixos[1].set_ylabel('Número de colheitas')
            eixos[1].legend()
            eixos[1].grid(True, alpha=0.3)

            # Distribuição de ações
            acoes_ord = sorted(self.acoes_total.items(), key=lambda x: x[1], reverse=True)
            if acoes_ord:
                nomes, vals = zip(*acoes_ord)
                cores = ['#2196F3' if v > np.mean(vals) else '#90CAF9' for v in vals]
                eixos[2].barh(nomes, vals, color=cores)
                eixos[2].set_title('Ações Executadas (total)')
                eixos[2].set_xlabel('Frequência')
                eixos[2].grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(os.path.join(self.dir, 'desempenho.png'), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  [aviso] Erro nos gráficos: {e}")

    def _resumo(self):
        recs = self.recompensas
        colh = self.colheitas
        print(f"\n{'='*60}")
        print(f"  BOT PERFEITO — RESULTADO FINAL")
        print(f"{'='*60}")
        print(f"  Episódios executados:   {len(recs)}")
        print(f"  Recompensa média:       {np.mean(recs):>10.2f}")
        print(f"  Melhor episódio:        {np.max(recs):>10.2f}")
        print(f"  Pior episódio:          {np.min(recs):>10.2f}")
        print(f"  Desvio padrão:          {np.std(recs):>10.2f}")
        print(f"  Colheitas por episódio: {np.mean(colh):>10.2f}")
        print(f"{'='*60}")
        print(f"  Top ações:")
        for nome, cnt in sorted(self.acoes_total.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {nome:25s}: {cnt:6d}")
        print(f"{'='*60}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(_DIR, '../ppo-lstm-masked'))
    from env import AmbienteFazendaGym
    from ppo_lstm_masked import InvolucroMascaraAcoes

    # --- argumentos CLI (protocolo v2.0) ---
    import argparse
    import random
    parser = argparse.ArgumentParser(description='Geração de demonstrações do BotPerfeito')
    parser.add_argument('--n-eps', type=int, default=200,
                        help='Nº de episódios de demonstração (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed para reprodutibilidade (default: 42)')
    parser.add_argument('--passos-maximos', type=int, default=365,
                        help='Passos por episódio (default: 365)')
    parser.add_argument('--imperfeicao', type=float, default=0.14,
                        help='Prob por step de entrar em modo preguicoso (default: 0.14 ~ 7500 rec)')
    parser.add_argument('--lazy-steps', type=int, default=8,
                        help='Steps consecutivos em ESPERAR ao entrar em modo preguicoso (default: 8)')
    args = parser.parse_args()

    # Fixar todas as fontes de aleatoriedade
    np.random.seed(args.seed)
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    env = InvolucroMascaraAcoes(AmbienteFazendaGym(passos_maximos=args.passos_maximos))
    try:
        env.reset(seed=args.seed)
    except TypeError:
        env.reset()

    dir_res = os.path.join(_DIR, '../../results/bot_perfeito')
    os.makedirs(dir_res, exist_ok=True)

    print(f"[BotPerfeito] seed={args.seed}  n_eps={args.n_eps}  horizonte={args.passos_maximos}")
    print(f"[BotPerfeito] imperfeicao={args.imperfeicao}  lazy_steps={args.lazy_steps}")
    executor = ExecutorBot(env, dir_res, imperfeicao=args.imperfeicao,
                            lazy_steps=args.lazy_steps, seed=args.seed)
    executor.executar(num_episodios=args.n_eps, passos_maximos=args.passos_maximos,
                       salvar_trajetorias=True)
