import numpy as np
from typing import Dict, Tuple, List
import random

_TAXA_NATALIDADE = 0.02
_TAXA_MORTALIDADE = 0.015

class Passaros:
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4)):
        self.tamanho_grade = tamanho_grade

        self.populacao_minima = 0
        self.populacao_maxima = 50
        self.media_populacao_base = 15

        self.resetar()

    def resetar(self):
        self.populacao_passaros = {}
        self.passo_tempo = 0
        self._gerar_populacao_inicial()

    def _gerar_populacao_inicial(self):
        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                populacao = np.random.randint(
                    self.populacao_minima,
                    min(self.populacao_maxima, self.media_populacao_base + 10)
                )
                self.populacao_passaros[(x, y)] = populacao

    def atualizar(self, instalacoes: Dict):
        self.passo_tempo += 1

        for pos in list(self.populacao_passaros.keys()):
            pop_atual = self.populacao_passaros[pos]

            nascimentos = int(pop_atual * _TAXA_NATALIDADE)
            mortes = int(pop_atual * _TAXA_MORTALIDADE)
            ruido = np.random.randint(-2, 3)
            nova_pop = pop_atual + nascimentos - mortes + ruido

            nova_pop = self._aplicar_efeito_espantalho(nova_pop, pos, instalacoes)

            self.populacao_passaros[pos] = int(np.clip(nova_pop, self.populacao_minima, self.populacao_maxima))

    def _aplicar_efeito_espantalho(self, populacao_base: int, pos: Tuple[int, int], instalacoes: Dict) -> int:
        forca_espantalho = instalacoes.get(pos, {}).get('forca_espantalho', 0)

        if forca_espantalho > 0:
            taxa_reducao = 1.0 - np.exp(-forca_espantalho * 0.5)
            taxa_reducao = min(0.9, taxa_reducao)
            populacao_final = int(populacao_base * (1.0 - taxa_reducao))
        else:
            populacao_final = populacao_base

        return max(0, populacao_final)

    def obter_observacao(self) -> Dict:
        return self.populacao_passaros.copy()

    def obter_populacao_em(self, x: int, y: int) -> int:
        pos = (x, y)
        return self.populacao_passaros.get(pos, 0)

    def obter_populacao_total(self) -> int:
        return sum(self.populacao_passaros.values())

    def definir_parametros_populacao(self, pop_min: int, pop_max: int, media_base: int):
        self.populacao_minima = pop_min
        self.populacao_maxima = pop_max
        self.media_populacao_base = media_base
