import numpy as np
from typing import Dict, Tuple, List, Optional
from enum import Enum

class TipoInstalacao(Enum):
    ESPANTALHO = "espantalho"
    CERCA_VIVA = "cerca_viva"
    IRRIGACAO = "irrigacao"
    ESTUFA = "estufa"

class TipoEspantalho(Enum):
    BASICO = "basico"
    AVANCADO = "avancado"

_DECAIMENTO_ESPANTALHO_POR_PASSO = 0.001

class Instalacoes:
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4)):
        self.tamanho_grade = tamanho_grade

        self.forcas_espantalho = {
            TipoEspantalho.BASICO: 1.0,
            TipoEspantalho.AVANCADO: 2.0
        }

        self.resetar()

    def resetar(self):
        self.instalacoes = {}
        self.passo_tempo = 0

    def colocar_espantalho(self, x: int, y: int, tipo_espantalho: TipoEspantalho):
        pos = (x, y)
        if 0 <= x < self.tamanho_grade[0] and 0 <= y < self.tamanho_grade[1]:
            forca = self.forcas_espantalho[tipo_espantalho]
            instalacao = Instalacao(
                tipo=TipoInstalacao.ESPANTALHO,
                tipo_espantalho=tipo_espantalho,
                forca_espantalho=forca
            )
            self.instalacoes[pos] = instalacao

    def colocar_cerca_viva(self, x: int, y: int):
        pos = (x, y)
        if 0 <= x < self.tamanho_grade[0] and 0 <= y < self.tamanho_grade[1]:
            instalacao = Instalacao(tipo=TipoInstalacao.CERCA_VIVA)
            self.instalacoes[pos] = instalacao

    def colocar_irrigacao(self, x: int, y: int, eficiencia: float = 0.8):
        pos = (x, y)
        if 0 <= x < self.tamanho_grade[0] and 0 <= y < self.tamanho_grade[1]:
            instalacao = Instalacao(
                tipo=TipoInstalacao.IRRIGACAO,
                eficiencia_irrigacao=eficiencia
            )
            self.instalacoes[pos] = instalacao

    def remover_espantalho(self, x: int, y: int):
        pos = (x, y)
        if pos in self.instalacoes and self.instalacoes[pos].tipo == TipoInstalacao.ESPANTALHO:
            del self.instalacoes[pos]

    def remover_instalacao(self, x: int, y: int):
        pos = (x, y)
        if pos in self.instalacoes:
            del self.instalacoes[pos]

    def atualizar(self):
        self.passo_tempo += 1
        for instalacao in self.instalacoes.values():
            if instalacao.tipo == TipoInstalacao.ESPANTALHO:
                instalacao.forca_espantalho = max(
                    0.1,
                    instalacao.forca_espantalho * (1.0 - _DECAIMENTO_ESPANTALHO_POR_PASSO)
                )

    def obter_observacao(self) -> Dict:
        obs = {}

        for pos, instalacao in self.instalacoes.items():
            obs[pos] = {
                'tipo': instalacao.tipo.value,
                'forca_espantalho': getattr(instalacao, 'forca_espantalho', 0),
                'tipo_espantalho': getattr(instalacao, 'tipo_espantalho', None),
                'eficiencia_irrigacao': getattr(instalacao, 'eficiencia_irrigacao', None)
            }

        return obs

    def obter_instalacao_em(self, x: int, y: int) -> Optional['Instalacao']:
        pos = (x, y)
        return self.instalacoes.get(pos)

    def tem_espantalho(self, x: int, y: int) -> bool:
        instalacao = self.obter_instalacao_em(x, y)
        return instalacao is not None and instalacao.tipo == TipoInstalacao.ESPANTALHO

    def obter_forca_espantalho(self, x: int, y: int) -> float:
        instalacao = self.obter_instalacao_em(x, y)
        if instalacao and instalacao.tipo == TipoInstalacao.ESPANTALHO:
            return getattr(instalacao, 'forca_espantalho', 0)
        return 0.0

    def tem_cerca_viva(self, x: int, y: int) -> bool:
        instalacao = self.obter_instalacao_em(x, y)
        return instalacao is not None and instalacao.tipo == TipoInstalacao.CERCA_VIVA

    def tem_irrigacao(self, x: int, y: int) -> bool:
        instalacao = self.obter_instalacao_em(x, y)
        return instalacao is not None and instalacao.tipo == TipoInstalacao.IRRIGACAO

    def obter_eficiencia_irrigacao(self, x: int, y: int) -> float:
        instalacao = self.obter_instalacao_em(x, y)
        if instalacao and instalacao.tipo == TipoInstalacao.IRRIGACAO:
            return getattr(instalacao, 'eficiencia_irrigacao', 0.8)
        return 0.0

    def obter_todas_posicoes_por_tipo(self, tipo_instalacao: TipoInstalacao) -> List[Tuple[int, int]]:
        posicoes = []
        for pos, instalacao in self.instalacoes.items():
            if instalacao.tipo == tipo_instalacao:
                posicoes.append(pos)
        return posicoes

    def definir_forca_espantalho(self, tipo_espantalho: TipoEspantalho, forca: float):
        self.forcas_espantalho[tipo_espantalho] = forca

        for instalacao in self.instalacoes.values():
            if (instalacao.tipo == TipoInstalacao.ESPANTALHO and
                instalacao.tipo_espantalho == tipo_espantalho):
                instalacao.forca_espantalho = forca

class Instalacao:
    def __init__(self, tipo: TipoInstalacao, tipo_espantalho: Optional[TipoEspantalho] = None,
                 forca_espantalho: float = 0, eficiencia_irrigacao: float = 0.8):
        self.tipo = tipo
        self.tipo_espantalho = tipo_espantalho
        self.forca_espantalho = forca_espantalho
        self.eficiencia_irrigacao = eficiencia_irrigacao
        self.idade = 0
