import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional

class EstagioPlanta(Enum):
    SEMENTE = "semente"
    GERMINACAO = "germinacao"
    CRESCIMENTO = "crescimento"
    FLORESCIMENTO = "florescimento"
    FRUTIFICACAO = "frutificacao"
    COLHEITA = "colheita"
    MORTA = "morta"

class Planta:
    def __init__(self, tamanho_grade: Tuple[int, int] = (10, 10)):
        self.tamanho_grade = tamanho_grade
        self.resetar()

    def resetar(self):
        self.plantas = {}
        self.passo_tempo = 0

    def adicionar_planta(self, x: int, y: int, tipo_planta: str = "padrao"):
        if 0 <= x < self.tamanho_grade[0] and 0 <= y < self.tamanho_grade[1]:
            self.plantas[(x, y)] = EstadoPlanta(tipo_planta)

    def atualizar(self, clima: Dict, solo: Dict, polinizadores: Dict, ervas_daninhas: Dict):
        self.passo_tempo += 1
        plantas_para_remover = []

        for pos, planta in self.plantas.items():
            if planta.estagio == EstagioPlanta.MORTA:
                continue

            fator_crescimento = self._calcular_fator_crescimento(clima, solo, polinizadores, ervas_daninhas, pos)
            planta.atualizar(fator_crescimento)

            if planta.estagio == EstagioPlanta.MORTA:
                plantas_para_remover.append(pos)

        for pos in plantas_para_remover:
            del self.plantas[pos]

    def _calcular_fator_crescimento(self, clima: Dict, solo: Dict, polinizadores: Dict, ervas_daninhas: Dict, pos: Tuple[int, int]) -> float:
        fator = 1.0

        temp = clima.get('temperatura', 20)
        umidade_ar = clima.get('umidade_ar', 0.5)
        chuva = clima.get('chuva', 0)

        if temp < 10 or temp > 35:
            fator *= 0.7
        if umidade_ar < 0.3 or umidade_ar > 0.9:
            fator *= 0.8
        if chuva < 2.0:
            fator *= 0.9

        qualidade_solo = solo.get(pos, {}).get('qualidade', 0.5)
        fator *= (1.0 + qualidade_solo)

        if self.plantas[pos].estagio == EstagioPlanta.FLORESCIMENTO:
            val_pol = polinizadores.get(pos, 0)
            densidade_polinizadores = val_pol.get('densidade', 0) if isinstance(val_pol, dict) else val_pol
            fator *= (0.8 + 0.4 * densidade_polinizadores)

        densidade_ervas = ervas_daninhas.get(pos, {}).get('densidade', 0)
        fator *= (1.0 - 0.5 * densidade_ervas)

        return max(0.0, fator)

    def obter_observacao(self) -> Dict:
        obs = {}
        for pos, planta in self.plantas.items():
            obs[pos] = {
                'estagio': planta.estagio.value,
                'saude': planta.saude,
                'progresso_crescimento': planta.progresso_crescimento,
                'peso_fruto': planta.peso_fruto
            }
        return obs

    def pode_colher(self, pos: Tuple[int, int]) -> bool:
        if pos in self.plantas:
            return self.plantas[pos].estagio == EstagioPlanta.COLHEITA
        return False

    def colher(self, pos: Tuple[int, int]) -> float:
        if self.pode_colher(pos):
            quantidade_rendimento = self.plantas[pos].peso_fruto
            del self.plantas[pos]
            return quantidade_rendimento
        return 0.0

class EstadoPlanta:
    def __init__(self, tipo_planta: str = "padrao"):
        self.tipo_planta = tipo_planta
        self.estagio = EstagioPlanta.SEMENTE
        self.saude = 1.0
        self.progresso_crescimento = 0.0
        self.peso_fruto = 0.0
        self.idade = 0
        # Rastreio da qualidade de cuidado acumulada nas fases ativas.
        # Usado para calcular o multiplicador de rendimento na colheita:
        # yield_efetivo = peso_fruto × max(0.3, 0.3 + fator_medio)
        # Baseline fator≈0.70 (clima típico) → mult=1.0 (sem mudança)
        # Bot com solo fértil e sem ervas fator≈1.0+ → mult=1.3+ (+30%)
        # Negligência fator≈0.4 → mult=0.7 (-30%)
        self._fator_soma = 0.0
        self._fator_n    = 0

        self.limiares_estagio = {
            EstagioPlanta.SEMENTE: 5,
            EstagioPlanta.GERMINACAO: 10,
            EstagioPlanta.CRESCIMENTO: 20,
            EstagioPlanta.FLORESCIMENTO: 15,
            EstagioPlanta.FRUTIFICACAO: 25,
            EstagioPlanta.COLHEITA: 10
        }

    @property
    def fator_medio(self) -> float:
        """Fator de crescimento médio durante as fases ativas (CRESCIMENTO→COLHEITA)."""
        return self._fator_soma / self._fator_n if self._fator_n > 0 else 1.0

    @property
    def multiplicador_rendimento(self) -> float:
        """Multiplicador de rendimento na colheita baseado na qualidade do cuidado.

        Fórmula: max(0.3, 0.3 + fator_medio)
          fator_medio ≈ 0.70 (baseline climático) → mult = 1.00  (sem diferença)
          fator_medio ≈ 1.00 (bot fertilizando)   → mult = 1.30  (+30%)
          fator_medio ≈ 1.40 (condições ótimas)   → mult = 1.70  (+70%)
          fator_medio ≈ 0.40 (negligência)        → mult = 0.70  (-30%)
        """
        return max(0.3, 0.3 + self.fator_medio)

    def atualizar(self, fator_crescimento: float):
        self.idade += 1
        self.progresso_crescimento += fator_crescimento * 0.5

        limiar_atual = self.limiares_estagio.get(self.estagio, float('inf'))
        if self.progresso_crescimento >= limiar_atual:
            self._avancar_estagio()

        self._atualizar_saude(fator_crescimento)

        # Acumular fator nas fases ativas para calcular multiplicador_rendimento
        _FASES_ATIVAS = (EstagioPlanta.CRESCIMENTO, EstagioPlanta.FLORESCIMENTO,
                         EstagioPlanta.FRUTIFICACAO, EstagioPlanta.COLHEITA)
        if self.estagio in _FASES_ATIVAS:
            self._fator_soma += fator_crescimento
            self._fator_n    += 1

        if self.estagio == EstagioPlanta.FRUTIFICACAO:
            self.peso_fruto += fator_crescimento * 0.5

    def _avancar_estagio(self):
        ordem_estagios = [
            EstagioPlanta.SEMENTE,
            EstagioPlanta.GERMINACAO,
            EstagioPlanta.CRESCIMENTO,
            EstagioPlanta.FLORESCIMENTO,
            EstagioPlanta.FRUTIFICACAO,
            EstagioPlanta.COLHEITA
        ]

        indice_atual = ordem_estagios.index(self.estagio)
        if indice_atual < len(ordem_estagios) - 1:
            self.estagio = ordem_estagios[indice_atual + 1]
            self.progresso_crescimento = 0.0
        elif self.estagio == EstagioPlanta.COLHEITA:
            if self.progresso_crescimento >= self.limiares_estagio[self.estagio]:
                self.estagio = EstagioPlanta.MORTA

    def _atualizar_saude(self, fator_crescimento: float):
        if fator_crescimento < 0.3:
            self.saude -= 0.05
        elif fator_crescimento > 0.8:
            self.saude = min(1.0, self.saude + 0.03)
        elif fator_crescimento > 0.5:
            self.saude = min(1.0, self.saude + 0.01)

        self.saude = max(0.0, self.saude)

        if self.saude <= 0:
            self.estagio = EstagioPlanta.MORTA
