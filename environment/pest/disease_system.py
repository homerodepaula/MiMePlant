import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import random

class EstagioDoenca(Enum):
    SAUDAVEL = 0
    INICIAL = 1
    MODERADO = 2
    SEVERO = 3
    RECUPERANDO = 4

class TipoDoenca:
    def __init__(self, nome: str, taxa_dispersao: float, severidade: float,
                 temp_otima: Tuple[float, float], umidade_otima: Tuple[float, float]):
        self.nome = nome
        self.taxa_dispersao = taxa_dispersao
        self.severidade = severidade
        self.temp_otima = temp_otima
        self.umidade_otima = umidade_otima

class SistemaDoencas:
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4)):
        self.tamanho_grade = tamanho_grade
        self.doencas = {}
        self.ciclos_infeccao = {}
        self.doencas_plantas = {}

        self.tipos_doenca = {
            'infeccao_fungica': TipoDoenca(
                'infeccao_fungica',
                taxa_dispersao=0.15,
                severidade=0.7,
                temp_otima=(15, 25),
                umidade_otima=(0.7, 0.9)
            ),
            'praga_bacteriana': TipoDoenca(
                'praga_bacteriana',
                taxa_dispersao=0.20,
                severidade=0.8,
                temp_otima=(20, 30),
                umidade_otima=(0.6, 0.8)
            ),
            'mosaico_viral': TipoDoenca(
                'mosaico_viral',
                taxa_dispersao=0.10,
                severidade=0.5,
                temp_otima=(18, 28),
                umidade_otima=(0.4, 0.7)
            ),
            'podridao_raiz': TipoDoenca(
                'podridao_raiz',
                taxa_dispersao=0.12,
                severidade=0.9,
                temp_otima=(10, 20),
                umidade_otima=(0.8, 1.0)
            )
        }

        self.taxas_progressao = {
            EstagioDoenca.INICIAL: 0.1,
            EstagioDoenca.MODERADO: 0.15,
            EstagioDoenca.SEVERO: 0.05,
            EstagioDoenca.RECUPERANDO: 0.2
        }

        self.taxa_recuperacao_base = 0.05
        self.impulso_recuperacao_tratamento = 0.15

    def verificar_infeccao(self, x: int, y: int, nome_doenca: str) -> EstagioDoenca:
        pos = (x, y)
        if pos in self.doencas_plantas and nome_doenca in self.doencas_plantas[pos]:
            return self.doencas_plantas[pos][nome_doenca]
        return EstagioDoenca.SAUDAVEL

    def espalhar_doenca(self, dados_clima: Dict, posicoes_plantas: List[Tuple[int, int]]):
        temp = dados_clima.get('temperatura', 20)
        umidade_ar = dados_clima.get('umidade_ar', 0.5)

        for nome_doenca, tipo_doenca in self.tipos_doenca.items():
            if self._condicoes_favoraveis(temp, umidade_ar, tipo_doenca):
                novas_infeccoes = []

                for pos in posicoes_plantas:
                    if pos in self.doencas_plantas and nome_doenca in self.doencas_plantas[pos]:
                        estagio = self.doencas_plantas[pos][nome_doenca]
                        if estagio in [EstagioDoenca.MODERADO, EstagioDoenca.SEVERO]:
                            taxa_dispersao = tipo_doenca.taxa_dispersao
                            if estagio == EstagioDoenca.SEVERO:
                                taxa_dispersao *= 1.5

                            vizinhos = self._obter_vizinhos(pos)
                            for vizinho in vizinhos:
                                if vizinho in posicoes_plantas:
                                    chance_infeccao = taxa_dispersao * self._obter_susceptibilidade_planta(vizinho)
                                    if random.random() < chance_infeccao:
                                        if vizinho not in self.doencas_plantas:
                                            self.doencas_plantas[vizinho] = {}
                                        self.doencas_plantas[vizinho][nome_doenca] = EstagioDoenca.INICIAL
                                        novas_infeccoes.append(vizinho)

                if novas_infeccoes:
                    if nome_doenca not in self.ciclos_infeccao:
                        self.ciclos_infeccao[nome_doenca] = []
                    self.ciclos_infeccao[nome_doenca].append({
                        'passo': dados_clima.get('passo', 0),
                        'novas_infeccoes': len(novas_infeccoes),
                        'condicoes': {'temp': temp, 'umidade_ar': umidade_ar}
                    })

    def progredir_doencas(self, dados_clima: Dict, tratamento_aplicado: bool = False):
        temp = dados_clima.get('temperatura', 20)
        umidade_ar = dados_clima.get('umidade_ar', 0.5)

        for pos, doencas in list(self.doencas_plantas.items()):
            for nome_doenca, estagio in list(doencas.items()):
                tipo_doenca = self.tipos_doenca[nome_doenca]

                taxa_progressao = self.taxas_progressao.get(estagio, 0)

                if self._condicoes_favoraveis(temp, umidade_ar, tipo_doenca):
                    taxa_progressao *= 1.5
                else:
                    taxa_progressao *= 0.7

                taxa_recuperacao = self.taxa_recuperacao_base
                if tratamento_aplicado:
                    taxa_recuperacao += self.impulso_recuperacao_tratamento

                if estagio == EstagioDoenca.RECUPERANDO:
                    if random.random() < taxa_recuperacao:
                        del doencas[nome_doenca]
                        if not doencas:
                            del self.doencas_plantas[pos]
                elif estagio == EstagioDoenca.SEVERO:
                    if random.random() < taxa_recuperacao:
                        doencas[nome_doenca] = EstagioDoenca.RECUPERANDO
                else:
                    if random.random() < taxa_progressao:
                        proximo_estagio = self._obter_proximo_estagio(estagio)
                        doencas[nome_doenca] = proximo_estagio

    def aplicar_tratamento(self, x: int, y: int, tipo_tratamento: str = 'fungicida'):
        pos = (x, y)
        if pos in self.doencas_plantas:
            doencas = self.doencas_plantas[pos]

            for nome_doenca in list(doencas.keys()):
                if tipo_tratamento == 'fungicida' and 'fungica' in nome_doenca:
                    doencas[nome_doenca] = EstagioDoenca.RECUPERANDO
                elif tipo_tratamento == 'antibiotico' and 'bacteriana' in nome_doenca:
                    doencas[nome_doenca] = EstagioDoenca.RECUPERANDO
                elif tipo_tratamento == 'amplo_espectro':
                    if random.random() < 0.6:
                        doencas[nome_doenca] = EstagioDoenca.RECUPERANDO

    def calcular_impacto_doenca(self, x: int, y: int) -> float:
        pos = (x, y)
        if pos not in self.doencas_plantas:
            return 0.0

        impacto_total = 0.0
        for nome_doenca, estagio in self.doencas_plantas[pos].items():
            tipo_doenca = self.tipos_doenca[nome_doenca]

            impacto_estagio = {
                EstagioDoenca.INICIAL: 0.2,
                EstagioDoenca.MODERADO: 0.5,
                EstagioDoenca.SEVERO: 0.8,
                EstagioDoenca.RECUPERANDO: 0.1
            }.get(estagio, 0)

            impacto_total += tipo_doenca.severidade * impacto_estagio

        return min(1.0, impacto_total)

    def obter_indicadores_doenca(self) -> List[float]:
        indicadores = []

        total_posicoes = self.tamanho_grade[0] * self.tamanho_grade[1]
        for nome_doenca in self.tipos_doenca.keys():
            contagem_infectados = sum(
                1 for doencas in self.doencas_plantas.values()
                if nome_doenca in doencas and doencas[nome_doenca] != EstagioDoenca.SAUDAVEL
            )
            indicadores.append(contagem_infectados / total_posicoes)

        pressao_total = sum(
            self.calcular_impacto_doenca(pos[0], pos[1])
            for pos in self.doencas_plantas.keys()
        )
        indicadores.append(pressao_total / total_posicoes)

        return indicadores

    def calcular_epidemiologia(self) -> Dict:
        total_posicoes = self.tamanho_grade[0] * self.tamanho_grade[1]

        contagem_estagios = {estagio: 0 for estagio in EstagioDoenca}
        for doencas in self.doencas_plantas.values():
            for estagio in doencas.values():
                contagem_estagios[estagio] += 1

        total_infectados = sum(contagem_estagios[estagio] for estagio in [EstagioDoenca.INICIAL, EstagioDoenca.MODERADO, EstagioDoenca.SEVERO])
        prevalencia = total_infectados / total_posicoes if total_posicoes > 0 else 0

        pontuacao_severidade = (
            contagem_estagios[EstagioDoenca.MODERADO] * 1 +
            contagem_estagios[EstagioDoenca.SEVERO] * 2
        ) / total_posicoes if total_posicoes > 0 else 0

        return {
            'prevalencia': prevalencia,
            'pontuacao_severidade': pontuacao_severidade,
            'total_infectados': total_infectados,
            'distribuicao_estagios': {estagio.name: contagem for estagio, contagem in contagem_estagios.items()}
        }

    def _condicoes_favoraveis(self, temp: float, umidade_ar: float, tipo_doenca: TipoDoenca) -> bool:
        temp_favoravel = tipo_doenca.temp_otima[0] <= temp <= tipo_doenca.temp_otima[1]
        umidade_favoravel = tipo_doenca.umidade_otima[0] <= umidade_ar <= tipo_doenca.umidade_otima[1]
        return temp_favoravel and umidade_favoravel

    def _obter_vizinhos(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        vizinhos = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.tamanho_grade[0] and 0 <= ny < self.tamanho_grade[1]:
                    vizinhos.append((nx, ny))
        return vizinhos

    def _obter_susceptibilidade_planta(self, pos: Tuple[int, int]) -> float:
        susceptibilidade = 0.3

        if pos in self.doencas_plantas:
            doencas_existentes = len(self.doencas_plantas[pos])
            susceptibilidade += doencas_existentes * 0.1

        return min(1.0, susceptibilidade)

    def _obter_proximo_estagio(self, estagio_atual: EstagioDoenca) -> EstagioDoenca:
        progressao = {
            EstagioDoenca.SAUDAVEL: EstagioDoenca.INICIAL,
            EstagioDoenca.INICIAL: EstagioDoenca.MODERADO,
            EstagioDoenca.MODERADO: EstagioDoenca.SEVERO,
            EstagioDoenca.SEVERO: EstagioDoenca.SEVERO,
            EstagioDoenca.RECUPERANDO: EstagioDoenca.SAUDAVEL
        }
        return progressao.get(estagio_atual, estagio_atual)

    def resetar(self):
        self.doencas = {}
        self.ciclos_infeccao = {}
        self.doencas_plantas = {}
