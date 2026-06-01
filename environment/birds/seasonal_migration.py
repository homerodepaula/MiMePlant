import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
import random

class EstadoMigracao(Enum):
    RESIDENTE = 0
    PREPARANDO = 1
    MIGRANDO = 2
    RETORNANDO = 3

class MigracaoSazonal:
    def __init__(self, tamanho_grade: Tuple[int, int] = (4, 4)):
        self.tamanho_grade = tamanho_grade
        self.estacoes_migracao = ['primavera', 'outono', 'inverno']
        self.estacao_atual = 'primavera'

        self.gatilhos_migracao = {
            'limiar_temperatura': 15,
            'limiar_umidade': 0.3,
            'limiar_duracao_dia': 10,
            'limiar_escassez_alimento': 0.2
        }

        self.passaros_residentes = {}
        self.passaros_migrando = []
        self.historico_migracao = []

        self.rotas_migracao = {
            'norte_sul': {
                'primavera': 'para_norte',
                'outono': 'para_sul',
                'inverno': 'para_sul'
            },
            'leste_oeste': {
                'primavera': 'para_leste',
                'outono': 'para_oeste',
                'inverno': 'para_oeste'
            }
        }

        self.taxa_migracao_base = 0.3
        self.taxa_sobrevivencia_migracao = 0.85
        self.taxa_retorno = 0.9

        self.resetar()

    def verificar_gatilho_migracao(self, dados_clima: Dict, disponibilidade_alimento: Dict) -> Optional[str]:
        temp = dados_clima.get('temperatura', 20)
        umidade_ar = dados_clima.get('umidade_ar', 0.5)
        duracao_dia = dados_clima.get('duracao_dia', 12)
        estacao = dados_clima.get('estacao', 'primavera')

        total_alimento = np.mean(list(disponibilidade_alimento.values())) if disponibilidade_alimento else 0.5

        gatilhos = []

        if temp < self.gatilhos_migracao['limiar_temperatura']:
            gatilhos.append('estresse_frio')

        if umidade_ar < self.gatilhos_migracao['limiar_umidade']:
            gatilhos.append('estresse_seca')

        if duracao_dia < self.gatilhos_migracao['limiar_duracao_dia']:
            gatilhos.append('dias_curtos')

        if total_alimento < self.gatilhos_migracao['limiar_escassez_alimento']:
            gatilhos.append('escassez_alimento')

        if estacao in ['outono', 'inverno'] and not gatilhos:
            gatilhos.append('sazonal')

        if gatilhos:
            return gatilhos[0]

        return None

    def executar_migracao(self, tipo_gatilho: str, dados_clima: Dict):
        estacao = dados_clima.get('estacao', 'primavera')

        if estacao in self.rotas_migracao['norte_sul']:
            direcao = self.rotas_migracao['norte_sul'][estacao]
        else:
            direcao = 'para_sul'

        grupos_migrando = []
        for pos, populacao in list(self.passaros_residentes.items()):
            if populacao > 0:
                taxa_migracao = self.taxa_migracao_base

                if tipo_gatilho == 'estresse_frio':
                    taxa_migracao *= 1.5
                elif tipo_gatilho == 'estresse_seca':
                    taxa_migracao *= 1.3
                elif tipo_gatilho == 'escassez_alimento':
                    taxa_migracao *= 1.4

                pop_migrando = int(populacao * taxa_migracao)
                if pop_migrando > 0:
                    grupo = {
                        'origem': pos,
                        'populacao': pop_migrando,
                        'direcao': direcao,
                        'passo_partida': dados_clima.get('passo', 0),
                        'gatilho': tipo_gatilho,
                        'estado': EstadoMigracao.MIGRANDO
                    }
                    grupos_migrando.append(grupo)

                    self.passaros_residentes[pos] = populacao - pop_migrando
                    if self.passaros_residentes[pos] <= 0:
                        del self.passaros_residentes[pos]

        self.passaros_migrando.extend(grupos_migrando)

        self.historico_migracao.append({
            'passo': dados_clima.get('passo', 0),
            'gatilho': tipo_gatilho,
            'estacao': estacao,
            'grupos_migrando': len(grupos_migrando),
            'total_migrando': sum(g['populacao'] for g in grupos_migrando),
            'direcao': direcao
        })

    def atualizar_passaros_migrando(self, passo_atual: int, dados_clima: Dict):
        estacao = dados_clima.get('estacao', 'primavera')

        for grupo in list(self.passaros_migrando):
            dias_desde_partida = passo_atual - grupo['passo_partida']

            duracao_migracao = random.randint(7, 14)

            if dias_desde_partida >= duracao_migracao:
                taxa_sobrevivencia = self.taxa_sobrevivencia_migracao

                if grupo['gatilho'] == 'estresse_frio':
                    taxa_sobrevivencia *= 0.9
                elif grupo['gatilho'] == 'estresse_seca':
                    taxa_sobrevivencia *= 0.85

                sobreviventes = int(grupo['populacao'] * taxa_sobrevivencia)

                if sobreviventes > 0:
                    if estacao in ['primavera', 'verao'] and random.random() < self.taxa_retorno:
                        pos_retorno = self._obter_posicao_retorno(grupo['origem'])
                        if pos_retorno:
                            if pos_retorno in self.passaros_residentes:
                                self.passaros_residentes[pos_retorno] += sobreviventes
                            else:
                                self.passaros_residentes[pos_retorno] = sobreviventes

                            self.passaros_migrando.remove(grupo)
                    else:
                        self.passaros_migrando.remove(grupo)
                else:
                    self.passaros_migrando.remove(grupo)

    def _obter_posicao_retorno(self, origem: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = origem

        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)

        novo_x = max(0, min(self.tamanho_grade[0] - 1, x + dx))
        novo_y = max(0, min(self.tamanho_grade[1] - 1, y + dy))

        return (novo_x, novo_y)

    def obter_status_migracao(self) -> List[float]:
        status = []

        status.append(len(self.passaros_migrando))

        total_migrando = sum(g['populacao'] for g in self.passaros_migrando)
        status.append(total_migrando)

        total_residentes = sum(self.passaros_residentes.values())
        status.append(total_residentes)

        max_migracao_possivel = total_residentes + total_migrando
        if max_migracao_possivel > 0:
            atividade_migracao = total_migrando / max_migracao_possivel
        else:
            atividade_migracao = 0.0
        status.append(atividade_migracao)

        if self.historico_migracao:
            passo_atual = self.historico_migracao[-1]['passo']
            migracoes_recentes = len([
                m for m in self.historico_migracao[-10:]
                if m['passo'] >= (passo_atual - 10)
            ])
        else:
            migracoes_recentes = 0
        status.append(min(1.0, migracoes_recentes / 5.0))

        return status

    def adicionar_passaros(self, x: int, y: int, populacao: int):
        pos = (x, y)
        if pos in self.passaros_residentes:
            self.passaros_residentes[pos] += populacao
        else:
            self.passaros_residentes[pos] = populacao

    def remover_passaros(self, x: int, y: int, populacao: int):
        pos = (x, y)
        if pos in self.passaros_residentes:
            self.passaros_residentes[pos] = max(0, self.passaros_residentes[pos] - populacao)
            if self.passaros_residentes[pos] <= 0:
                del self.passaros_residentes[pos]

    def obter_populacao_em(self, x: int, y: int) -> int:
        pos = (x, y)
        return self.passaros_residentes.get(pos, 0)

    def obter_populacao_total(self) -> int:
        total_residentes = sum(self.passaros_residentes.values())
        total_migrando = sum(g['populacao'] for g in self.passaros_migrando)
        return total_residentes + total_migrando

    def obter_resumo_migracao(self) -> Dict:
        return {
            'populacao_residente': sum(self.passaros_residentes.values()),
            'populacao_migrando': sum(g['populacao'] for g in self.passaros_migrando),
            'grupos_migracao': len(self.passaros_migrando),
            'total_eventos_migracao': len(self.historico_migracao),
            'migracoes_recentes': len([
                m for m in self.historico_migracao[-5:]
            ]) if self.historico_migracao else 0
        }

    def resetar(self):
        self.passaros_residentes = {}
        self.passaros_migrando = []
        self.historico_migracao = []

        for x in range(self.tamanho_grade[0]):
            for y in range(self.tamanho_grade[1]):
                if random.random() < 0.3:
                    populacao = random.randint(1, 5)
                    self.passaros_residentes[(x, y)] = populacao
