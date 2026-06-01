import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, List, Any, Optional
import sys
import os

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
for subdir in ['plant', 'birds', 'pollinators', 'soil', 'weather', 'weeds', 'pest', 'cides-fertilizers', 'facilities']:
    sys.path.insert(0, os.path.join(diretorio_atual, subdir))

from plant import Planta, EstagioPlanta
from birds import Passaros
from pollinators import Polinizadores
from soil import Solo
from weather import Clima
from weeds import ErvasDaninhas
from pest import Pragas
from cides_fertilizers import DefensivosFertilizantes, TipoNutriente, TipoDefensivo
from facilities import Instalacoes, TipoInstalacao, TipoEspantalho

from weather import EventosClimaticosExtremos
from pest import SistemaDoencas
from birds import MigracaoSazonal

class AmbienteFazendaGym(gym.Env):
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4), passos_maximos: int = 365):
        self.tamanho_grade = tamanho_grade
        self.passos_maximos = passos_maximos
        self.passo_atual = 0

        self.planta = Planta(tamanho_grade)
        self.passaros = Passaros(tamanho_grade)
        self.polinizadores = Polinizadores(tamanho_grade)
        self.solo = Solo(tamanho_grade)
        self.clima = Clima(tamanho_grade)
        self.ervas_daninhas = ErvasDaninhas(tamanho_grade)
        self.pragas = Pragas(tamanho_grade)
        self.defensivos_fertilizantes = DefensivosFertilizantes(tamanho_grade)
        self.instalacoes = Instalacoes(tamanho_grade)

        self.eventos_extremos = EventosClimaticosExtremos()
        self.sistema_doencas = SistemaDoencas(tamanho_grade)
        self.migracao_sazonal = MigracaoSazonal(tamanho_grade)

        self._configurar_espaco_acoes()
        self._configurar_espaco_observacao()

        self.recompensa_episodio = 0
        self.historico_colheita = []

    def _configurar_espaco_acoes(self):
        self.nomes_acoes = [
            'plantar', 'colher', 'regar', 'fertilizar_N', 'fertilizar_P',
            'fertilizar_K', 'fertilizar_C', 'herbicida', 'pesticida',
            'colocar_espantalho_basico', 'colocar_espantalho_avancado', 'remover_espantalho',
            'colocar_cerca', 'observar', 'esperar'
        ]

        n_acoes = len(self.nomes_acoes)
        self.action_space = gym.spaces.Discrete(n_acoes)

    def _configurar_espaco_observacao(self):
        obs_base = 5
        obs_sistemas_amb = 15
        obs_grade_por_celula = 8
        obs_grade_total = self.tamanho_grade[0] * self.tamanho_grade[1] * obs_grade_por_celula

        tamanho_obs = obs_base + obs_sistemas_amb + obs_grade_total
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(tamanho_obs,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.planta.resetar()
        self.passaros.resetar()
        self.polinizadores.resetar()
        self.solo.resetar()
        self.clima.resetar()
        self.ervas_daninhas.resetar()
        self.pragas.resetar()
        self.defensivos_fertilizantes.resetar()
        self.instalacoes.resetar()

        self.eventos_extremos.resetar()
        self.sistema_doencas.resetar()
        self.migracao_sazonal.resetar()

        self.passo_atual = 0
        self.recompensa_episodio = 0
        self.historico_colheita = []

        return self._obter_observacao(), {}

    def step(self, action):
        nome_acao = self.nomes_acoes[action]

        recompensa = 0
        terminado = False
        truncado = False
        info = {}

        recompensa += self._executar_acao(nome_acao, action)

        self._atualizar_ambiente()

        recompensa += self._calcular_recompensa_passo()

        self.passo_atual += 1
        self.recompensa_episodio += recompensa

        # Coleta de KPIs Agrícolas para o artigo e análise de HPO
        info['kpis'] = {
            'produtividade_acumulada': float(sum(self.historico_colheita)),
            'qualidade_solo_media': float(np.mean([c.qualidade for c in self.solo.estado_solo.values()])),
            'umidade_solo_media': float(np.mean([c.umidade for c in self.solo.estado_solo.values()])),
            'densidade_polinizadores': float(self.polinizadores.obter_densidade_media()),
            'uso_quimicos_total': int(sum(len(apps) for apps in self.defensivos_fertilizantes.aplicacoes.values())),
            'num_plantas_ativas': int(len(self.planta.plantas)),
            'progresso_episodio': float(self.passo_atual / self.passos_maximos)
        }

        if self.passo_atual >= self.passos_maximos:
            truncado = True
            recompensa += self._calcular_recompensa_final()

        return self._obter_observacao(), recompensa, terminado, truncado, info

    def _executar_acao(self, nome_acao: str, id_acao: int) -> float:
        recompensa = 0

        if nome_acao == 'plantar':
            espacos_vazios = []
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    if (x, y) not in self.planta.plantas:
                        espacos_vazios.append((x, y))
            if espacos_vazios:
                local = espacos_vazios[np.random.randint(len(espacos_vazios))]
                self.planta.adicionar_planta(local[0], local[1])
                recompensa -= 1.0

        elif nome_acao == 'colher':
            for pos in list(self.planta.plantas.keys()):
                planta_obj = self.planta.plantas[pos]
                if planta_obj.estagio == EstagioPlanta.COLHEITA:
                    # Multiplicador de rendimento: combina dois sinais de qualidade
                    # de gestão com horizontes temporais diferentes —
                    #   (A) fator_medio: condições médias ao longo de TODO o ciclo
                    #   (B) qualidade do solo local na colheita: estado atual do solo
                    # Fórmula: mult = (mult_fator + mult_solo) / 2
                    #   mult_fator = max(0.3, 0.3 + fator_medio)  → range ≈ 0.6-1.8
                    #   mult_solo  = 0.5 + qualidade_solo         → range 0.5-1.5
                    # Média das duas: range ≈ 0.55-1.65
                    # Baseline sem gestão (fator≈0.70, qual≈0.40): mult≈0.95 (quase sem mudança)
                    # Bot fertilizando (fator≈0.90, qual≈0.65): mult≈1.23 (+23%)
                    # Ótimo (fator≈1.3, qual≈0.80): mult≈1.55 (+55%)
                    celula_solo = self.solo.obter_solo_em(pos[0], pos[1])
                    mult_fator = planta_obj.multiplicador_rendimento
                    mult_solo  = 0.5 + celula_solo.qualidade
                    multiplicador = (mult_fator + mult_solo) / 2.0
                    quantidade_colhida = planta_obj.peso_fruto * multiplicador
                    self.historico_colheita.append(quantidade_colhida)
                    self.planta.colher(pos)
                    recompensa += quantidade_colhida * 5.0

        elif nome_acao == 'regar':
            for pos in self.planta.plantas:
                celula_solo = self.solo.obter_solo_em(pos[0], pos[1])
                celula_solo.umidade = min(1.0, celula_solo.umidade + 0.2)
            recompensa -= 0.5

        elif nome_acao == 'fertilizar_N':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_fertilizante(
                        x, y, TipoNutriente.NITROGENIO, 2.0)
            recompensa -= 2.0

        elif nome_acao == 'fertilizar_P':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_fertilizante(
                        x, y, TipoNutriente.FOSFORO, 2.0)
            recompensa -= 2.0

        elif nome_acao == 'fertilizar_K':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_fertilizante(
                        x, y, TipoNutriente.POTASSIO, 2.0)
            recompensa -= 2.0

        elif nome_acao == 'fertilizar_C':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_fertilizante(
                        x, y, TipoNutriente.CARBONO, 2.0)
            recompensa -= 2.0

        elif nome_acao == 'herbicida':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_defensivo(
                        x, y, TipoDefensivo.HERBICIDA, 0.8, 2.0)
            recompensa -= 1.5

        elif nome_acao == 'pesticida':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.defensivos_fertilizantes.aplicar_defensivo(
                        x, y, TipoDefensivo.PESTICIDA, 0.8, 2.0)
            recompensa -= 1.5

        elif nome_acao == 'colocar_espantalho_basico':
            posicoes_com_planta = list(self.planta.plantas.keys())
            if posicoes_com_planta:
                pos = posicoes_com_planta[np.random.randint(len(posicoes_com_planta))]
                self.instalacoes.colocar_espantalho(pos[0], pos[1], TipoEspantalho.BASICO)
                recompensa -= 1.0

        elif nome_acao == 'colocar_espantalho_avancado':
            posicoes_com_planta = list(self.planta.plantas.keys())
            if posicoes_com_planta:
                pos = posicoes_com_planta[np.random.randint(len(posicoes_com_planta))]
                self.instalacoes.colocar_espantalho(pos[0], pos[1], TipoEspantalho.AVANCADO)
                recompensa -= 3.0

        elif nome_acao == 'remover_espantalho':
            posicoes_espantalhos = self.instalacoes.obter_todas_posicoes_por_tipo(
                TipoInstalacao.ESPANTALHO)
            if posicoes_espantalhos:
                pos = posicoes_espantalhos[np.random.randint(len(posicoes_espantalhos))]
                self.instalacoes.remover_espantalho(pos[0], pos[1])

        elif nome_acao == 'colocar_cerca':
            for x in range(self.tamanho_grade[0]):
                for y in range(self.tamanho_grade[1]):
                    self.instalacoes.colocar_cerca_viva(x, y)
            recompensa -= 5.0

        elif nome_acao == 'observar':
            recompensa -= 0.1

        elif nome_acao == 'esperar':
            pass

        return recompensa

    def _atualizar_ambiente(self):
        obs_clima = self.clima.obter_observacao()
        obs_solo = self.solo.obter_observacao()
        obs_instalacoes = self.instalacoes.obter_observacao()
        contrib_defensivos = self.defensivos_fertilizantes.obter_observacao()

        self.clima.atualizar()
        obs_clima = self.clima.obter_observacao()
        obs_clima['step'] = self.passo_atual

        self.eventos_extremos.verificar_condicoes_extremas(obs_clima)

        self.solo.atualizar(obs_clima, contrib_defensivos)
        self.defensivos_fertilizantes.atualizar()
        self.instalacoes.atualizar()

        self.passaros.atualizar(obs_instalacoes)
        contrib_solo_defensivos = self.defensivos_fertilizantes.obter_contribuicoes_solo()
        self.polinizadores.atualizar(obs_clima, obs_instalacoes, contrib_solo_defensivos)
        self.ervas_daninhas.atualizar(obs_clima, obs_solo, contrib_defensivos)
        self.pragas.atualizar(obs_clima, self.planta.obter_observacao(), contrib_defensivos,
                          self.passaros.obter_observacao())

        posicoes_plantas = list(self.planta.plantas.keys())
        self.sistema_doencas.espalhar_doenca(obs_clima, posicoes_plantas)
        self.sistema_doencas.progredir_doencas(obs_clima)

        disponibilidade_alimento = self._calcular_disponibilidade_alimento()
        gatilho_migracao = self.migracao_sazonal.verificar_gatilho_migracao(obs_clima, disponibilidade_alimento)
        if gatilho_migracao:
            self.migracao_sazonal.executar_migracao(gatilho_migracao, obs_clima)
        self.migracao_sazonal.atualizar_passaros_migrando(self.passo_atual, obs_clima)

        self.planta.atualizar(obs_clima, obs_solo, self.polinizadores.obter_observacao(),
                          self.ervas_daninhas.obter_observacao())

    def _calcular_disponibilidade_alimento(self) -> Dict:
        disponibilidade_alimento = {}

        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                pos = (x, y)
                disponibilidade = 0.0

                if pos in self.planta.plantas:
                    planta_obj = self.planta.plantas[pos]
                    if planta_obj.estagio in [EstagioPlanta.FLORESCIMENTO, EstagioPlanta.FRUTIFICACAO, EstagioPlanta.COLHEITA]:
                        disponibilidade += 0.8
                    elif planta_obj.estagio in [EstagioPlanta.CRESCIMENTO, EstagioPlanta.FLORESCIMENTO]:
                        disponibilidade += 0.4

                densidade_ervas = self.ervas_daninhas.obter_densidade_em(x, y)
                disponibilidade += densidade_ervas * 0.2

                densidade_polinizadores = self.polinizadores.obter_densidade_em(x, y)
                disponibilidade += densidade_polinizadores * 0.1

                disponibilidade_alimento[pos] = min(1.0, disponibilidade)

        return disponibilidade_alimento

    def _calcular_recompensa_passo(self) -> float:
        recompensa = 0.0
        num_plantas = len(self.planta.plantas)

        # Bônus de progresso de estágio: sinal denso para guiar o aprendizado
        # (proporcional à saúde, não por passo estático)
        for pos, planta_obj in self.planta.plantas.items():
            saude = planta_obj.saude
            if planta_obj.estagio == EstagioPlanta.GERMINACAO:
                recompensa += 0.05 * saude
            elif planta_obj.estagio == EstagioPlanta.CRESCIMENTO:
                recompensa += 0.10 * saude
            elif planta_obj.estagio == EstagioPlanta.FLORESCIMENTO:
                recompensa += 0.15 * saude
            elif planta_obj.estagio == EstagioPlanta.FRUTIFICACAO:
                recompensa += 0.20 * saude
            elif planta_obj.estagio == EstagioPlanta.COLHEITA:
                recompensa += 0.30 * saude  # incentivo para colher logo

        # Penalidade climática: uma penalidade global por passo, não multiplicada por planta
        if num_plantas > 0:
            obs_clima = self.clima.obter_observacao()
            temp = obs_clima.get('temperatura', 20)
            if temp > 35:
                recompensa -= 1.0
            elif temp < 5:
                recompensa -= 0.8

        # Penalidade por ervas daninhas (pressão média sobre o campo)
        if num_plantas > 0:
            densidade_media_ervas = 0.0
            for pos in self.planta.plantas.keys():
                densidade_media_ervas += self.ervas_daninhas.obter_densidade_em(pos[0], pos[1])
            densidade_media_ervas /= num_plantas
            recompensa -= densidade_media_ervas * 0.5

        # Penalidade por doenças (cap para evitar dominância)
        impacto_doenca_total = 0.0
        for pos in self.planta.plantas.keys():
            impacto_doenca_total += self.sistema_doencas.calcular_impacto_doenca(pos[0], pos[1])
        recompensa -= min(2.0, impacto_doenca_total * 0.5)

        # Bônus ecológico pelos pássaros residentes (pequeno, não dominante)
        resumo_migracao = self.migracao_sazonal.obter_resumo_migracao()
        if resumo_migracao['populacao_residente'] > 3:
            recompensa += 0.2

        return recompensa

    def _calcular_recompensa_final(self) -> float:
        recompensa = 0.0

        # Recompensa principal: colheita acumulada ao longo do episódio
        total_colheita = sum(self.historico_colheita)
        recompensa += total_colheita * 5.0

        # Bônus por frutas ainda na planta (não colhidas a tempo)
        for planta_obj in self.planta.plantas.values():
            if planta_obj.estagio in (EstagioPlanta.FRUTIFICACAO, EstagioPlanta.COLHEITA):
                recompensa += planta_obj.peso_fruto * 2.0  # penalização parcial (perdeu valor)

        # Bônus por qualidade do solo (investimento de longo prazo)
        qualidade_media_solo = np.mean(
            [celula.qualidade for celula in self.solo.estado_solo.values()])
        recompensa += qualidade_media_solo * 10.0

        # Bônus por polinizadores (biodiversidade sustentável)
        densidade_media_polinizadores = self.polinizadores.obter_densidade_media()
        recompensa += densidade_media_polinizadores * 5.0

        # Penalidade proporcional ao uso excessivo de defensivos (por célula)
        total_quimicos = sum(len(apps) for apps in self.defensivos_fertilizantes.aplicacoes.values())
        num_celulas = self.tamanho_grade[0] * self.tamanho_grade[1]
        quimicos_por_celula = total_quimicos / max(1, num_celulas)
        recompensa -= min(10.0, quimicos_por_celula * 2.0)

        return recompensa

    def _obter_observacao(self) -> np.ndarray:
        obs = []

        obs_clima = self.clima.obter_observacao()
        obs.extend([
            obs_clima.get('temperatura', 20) / 40,
            obs_clima.get('humidity', 0.5),
            obs_clima.get('rain', 0) / 50.0,
            obs_clima.get('wind', 0.5)
        ])

        obs.append(self.passo_atual / self.passos_maximos)

        impactos_extremos = self.eventos_extremos.obter_impactos_atuais()
        obs.extend(impactos_extremos)

        indicadores_doenca = self.sistema_doencas.obter_indicadores_doenca()
        obs.extend(indicadores_doenca)

        status_migracao = self.migracao_sazonal.obter_status_migracao()
        obs.extend(status_migracao)

        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                pos = (x, y)

                if pos in self.planta.plantas:
                    planta_obj = self.planta.plantas[pos]
                    obs.extend([
                        1.0,
                        planta_obj.saude,
                        list(EstagioPlanta).index(planta_obj.estagio) / len(EstagioPlanta)
                    ])
                else:
                    obs.extend([0.0, 0.0, 0.0])

                celula_solo = self.solo.obter_solo_em(x, y)
                obs.extend([
                    celula_solo.umidade,
                    celula_solo.qualidade
                ])

                densidade_ervas = self.ervas_daninhas.obter_densidade_em(x, y)
                obs.append(densidade_ervas)

                impacto_doenca = self.sistema_doencas.calcular_impacto_doenca(x, y)
                obs.append(impacto_doenca)

                pop_passaros = self.migracao_sazonal.obter_populacao_em(x, y)
                obs.append(pop_passaros / 10.0)

        return np.array(obs, dtype=np.float32)

    def render(self, mode='human'):
        if mode == 'human':
            grade = [['  ' for _ in range(self.tamanho_grade[1])] for _ in range(self.tamanho_grade[0])]

            for (x, y), planta_obj in self.planta.plantas.items():
                if 0 <= x < self.tamanho_grade[0] and 0 <= y < self.tamanho_grade[1]:
                    grade[x][y] = f'P{planta_obj.estagio.value}'

            for linha in grade:
                print(' '.join(linha))
            print(f"Passo: {self.passo_atual}/{self.passos_maximos}")

    def close(self):
        pass
