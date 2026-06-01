"""Wrapper de mascaramento de acoes invalidas para o ambiente FarmGym.

Injeta em info['mascara_acoes'] um vetor 0/1 indicando quais das 15 acoes sao
validas no estado atual. Usado por todos os agentes (BC, DAgger) e pelos
baselines (oraculo, aleatorio).
"""
import numpy as np
import gymnasium as gym


class InvolucroMascaraAcoes(gym.Wrapper):
    """Wrapper que injeta máscaras de ações válidas na info do ambiente."""

    def __init__(self, env):
        super().__init__(env)
        self.num_acoes = env.action_space.n

    def step(self, acao):
        obs, recompensa, terminado, truncado, info = self.env.step(acao)
        info['mascara_acoes'] = self._gerar_mascara(obs)
        return obs, recompensa, terminado, truncado, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['mascara_acoes'] = self._gerar_mascara(obs)
        return obs, info

    def _gerar_mascara(self, obs):
        # Índices de ação: 0=plantar 1=colher 2=regar 3=fertN 4=fertP 5=fertK 6=fertC
        #                  7=herbicida 8=pesticida 9=espant_basico 10=espant_avancado
        #                  11=remover_espant 12=colocar_cerca 13=observar 14=esperar
        mascara = np.ones(self.num_acoes, dtype=np.float32)

        deslocamento_base = 20
        tamanho_celula = 8
        num_celulas = 16

        tem_espaco_vazio = False
        tem_colheitavel = False
        tem_ervas = False
        tem_doenca = False
        tem_planta = False
        umidade_media = 0.0
        n_celulas_com_planta = 0

        for i in range(num_celulas):
            idx = deslocamento_base + (i * tamanho_celula)
            planta_presente = obs[idx + 0]
            estagio_planta  = obs[idx + 2]
            umidade_solo    = obs[idx + 3]
            densidade_ervas = obs[idx + 5]
            impacto_doenca  = obs[idx + 6]

            if planta_presente == 0:
                tem_espaco_vazio = True
            else:
                tem_planta = True
                n_celulas_com_planta += 1
                umidade_media += umidade_solo

            # COLHEITA = índice 5/6 nos enums; normalizado por 7 estágios → ≥5/7≈0.71
            if planta_presente > 0 and estagio_planta >= 0.71:
                tem_colheitavel = True
            if densidade_ervas > 0.1:
                tem_ervas = True
            if impacto_doenca > 0.1:
                tem_doenca = True

        if n_celulas_com_planta > 0:
            umidade_media /= n_celulas_com_planta

        if not tem_espaco_vazio:
            mascara[0] = 0.0   # plantar — só faz sentido se há espaço vazio
        if not tem_colheitavel:
            mascara[1] = 0.0   # colher — só faz sentido se há planta pronta
        if umidade_media > 0.8:
            mascara[2] = 0.0   # regar — solo já saturado
        if not tem_ervas:
            mascara[7] = 0.0   # herbicida — sem ervas daninhas
        if not tem_doenca:
            mascara[8] = 0.0   # pesticida — sem doença/praga
        if not tem_planta:
            mascara[9]  = 0.0  # espantalho básico — sem plantas para proteger
            mascara[10] = 0.0  # espantalho avançado — sem plantas para proteger

        mascara[13] = 1.0      # observar — sempre válido
        mascara[14] = 1.0      # esperar  — sempre válido
        return mascara
