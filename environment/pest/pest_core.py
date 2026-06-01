import numpy as np
from typing import Dict, Tuple, List
import random

class Pragas:
    def __init__(self, tamanho_grade: Tuple[int, int] = (10, 10)):
        self.tamanho_grade = tamanho_grade

        self.populacao_base = 10
        self.populacao_maxima = 100
        self.taxa_reproducao = 0.1
        self.taxa_dispersao = 0.05

        self.resetar()

    def resetar(self):
        self.populacao_pragas = {}
        self.passo_tempo = 0
        self._inicializar_populacao_pragas()

    def _inicializar_populacao_pragas(self):
        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                populacao = np.random.randint(0, self.populacao_base)
                self.populacao_pragas[(x, y)] = populacao

    def atualizar(self, clima: Dict, plantas: Dict, defensivos: Dict, passaros: Dict = None):
        self.passo_tempo += 1

        if passaros is None:
            passaros = {}

        nova_populacao = {}

        for pos in list(self.populacao_pragas.keys()):
            pop_atual = self.populacao_pragas[pos]

            if pop_atual <= 0:
                nova_populacao[pos] = 0
                continue

            fator_crescimento = self._calcular_fator_crescimento(pos, clima, plantas, defensivos)

            mudanca_populacao = pop_atual * self.taxa_reproducao * fator_crescimento

            taxa_mortalidade = 0.05
            mudanca_populacao -= pop_atual * taxa_mortalidade

            pop_passaros = passaros.get(pos, 0)
            predacao_passaros = pop_passaros * 0.002 * pop_atual
            mudanca_populacao -= predacao_passaros

            fator_dispersao = self._calcular_fator_dispersao(pos)
            mudanca_populacao += fator_dispersao

            efeito_pesticida = self._calcular_efeito_pesticida(pos, defensivos)
            mudanca_populacao *= (1 - efeito_pesticida)

            nova_pop = max(0, pop_atual + int(mudanca_populacao))
            nova_pop = min(nova_pop, self.populacao_maxima)

            nova_populacao[pos] = nova_pop

        self.populacao_pragas = nova_populacao

    def _calcular_fator_crescimento(self, pos: Tuple[int, int], clima: Dict, plantas: Dict, defensivos: Dict) -> float:
        fator = 1.0

        temp = clima.get('temperatura', 20)
        umidade_ar = clima.get('umidade_ar', 0.5)

        if 20 <= temp <= 30:
            fator *= 1.3
        elif temp < 10 or temp > 35:
            fator *= 0.4

        if umidade_ar > 0.7:
            fator *= 1.2

        dados_planta = plantas.get(pos, {})
        if dados_planta:
            estagio_planta = dados_planta.get('estagio', '')
            saude_planta = dados_planta.get('saude', 0)

            if estagio_planta in ['crescimento', 'florescimento', 'frutificacao']:
                fator *= (0.5 + 0.5 * saude_planta)
            else:
                fator *= 0.3

        return fator

    def _calcular_fator_dispersao(self, pos: Tuple[int, int]) -> float:
        x, y = pos
        dispersao = 0.0

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                pos_vizinho = (x + dx, y + dy)
                if pos_vizinho in self.populacao_pragas:
                    pop_vizinho = self.populacao_pragas[pos_vizinho]
                    if pop_vizinho > 20:
                        dispersao += pop_vizinho * self.taxa_dispersao * 0.01

        return dispersao

    def _calcular_efeito_pesticida(self, pos: Tuple[int, int], defensivos: Dict) -> float:
        dados_defensivo = defensivos.get(pos, {})

        efeito_pesticida = dados_defensivo.get('efeito_pesticida', 0)

        quantidade_defensivo = dados_defensivo.get('quantidade_defensivo', 0)
        efeito_geral = quantidade_defensivo * 0.4

        efeito_total = efeito_pesticida + efeito_geral
        return min(0.95, efeito_total)

    def obter_observacao(self) -> Dict:
        return self.populacao_pragas.copy()

    def obter_populacao_em(self, x: int, y: int) -> int:
        pos = (x, y)
        return self.populacao_pragas.get(pos, 0)

    def obter_populacao_total(self) -> int:
        return sum(self.populacao_pragas.values())

    def esta_nivel_infestacao(self, x: int, y: int, limiar: int = 50) -> bool:
        return self.obter_populacao_em(x, y) >= limiar

    def aplicar_tratamento(self, x: int, y: int, efetividade: float = 0.8):
        pos = (x, y)
        if pos in self.populacao_pragas:
            reducao = int(self.populacao_pragas[pos] * efetividade)
            self.populacao_pragas[pos] = max(0, self.populacao_pragas[pos] - reducao)

    def definir_parametros_populacao(self, pop_base: int, pop_max: int, taxa_repro: float, taxa_dispersao: float):
        self.populacao_base = pop_base
        self.populacao_maxima = pop_max
        self.taxa_reproducao = taxa_repro
        self.taxa_dispersao = taxa_dispersao
