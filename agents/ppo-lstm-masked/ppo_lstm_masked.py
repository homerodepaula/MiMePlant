import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import os
import json
import matplotlib.pyplot as plt
import time
from collections import deque


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


class RedePPOLSTMMascarada(nn.Module):
    """
    Rede PPO-LSTM-Masked com:
      - Extrator de características em duas camadas (LayerNorm + ReLU)
      - LSTM empilhada de N camadas para memória temporal profunda
      - LayerNorm pós-LSTM para estabilidade do gradiente
      - Mascaramento de ações inválidas (logits → -1e8 antes do softmax)
      - Cabeças separadas para política e valor
    """

    def __init__(self, dim_obs: int, dim_acoes: int,
                 dim_oculta: int = 256, num_camadas_lstm: int = 2):
        super().__init__()

        self.dim_oculta = dim_oculta
        self.num_camadas_lstm = num_camadas_lstm

        self.extrator = nn.Sequential(
            nn.Linear(dim_obs, dim_oculta),
            nn.LayerNorm(dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, dim_oculta),
            nn.LayerNorm(dim_oculta),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=dim_oculta,
            hidden_size=dim_oculta,
            num_layers=num_camadas_lstm,
            batch_first=True,
        )

        self.norm_lstm = nn.LayerNorm(dim_oculta)

        self.cabeca_politica = nn.Sequential(
            nn.Linear(dim_oculta, dim_oculta // 2),
            nn.LayerNorm(dim_oculta // 2),
            nn.ReLU(),
            nn.Linear(dim_oculta // 2, dim_acoes),
        )

        self.cabeca_valor = nn.Sequential(
            nn.Linear(dim_oculta, dim_oculta // 2),
            nn.LayerNorm(dim_oculta // 2),
            nn.ReLU(),
            nn.Linear(dim_oculta // 2, 1),
        )

    def forward(self, x, estado_oculto, mascara_acoes=None, finalizados=None):
        """
        x              : (batch, seq, dim_obs)
        estado_oculto  : tupla (h, c) cada (num_camadas, batch, dim_oculta)
        mascara_acoes  : (batch, seq, dim_acoes) ou None
        finalizados    : (batch, seq) float — 1.0 onde o episódio terminou
        """
        caracteristicas = self.extrator(x)

        if finalizados is not None:
            saidas = []
            h, c = estado_oculto
            for t in range(x.size(1)):
                saida, (h, c) = self.lstm(caracteristicas[:, t:t+1, :], (h, c))
                saidas.append(saida)
                # zerar estado oculto em transições terminais
                mascara_reset = (1.0 - finalizados[:, t]).view(1, -1, 1)
                h = h * mascara_reset
                c = c * mascara_reset
            saida_lstm = torch.cat(saidas, dim=1)
            novo_estado = (h, c)
        else:
            saida_lstm, novo_estado = self.lstm(caracteristicas, estado_oculto)

        saida_lstm = self.norm_lstm(saida_lstm)

        logits = self.cabeca_politica(saida_lstm)
        valor  = self.cabeca_valor(saida_lstm)

        if mascara_acoes is not None:
            NEG_INF = torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
            logits = torch.where(mascara_acoes > 0.5, logits, NEG_INF)

        return logits, valor, novo_estado


class BufferReproducaoLSTMMascarado:
    """Buffer de replay com suporte a estados LSTM e máscaras de ações."""

    def __init__(self, capacidade: int = 2048):
        self.capacidade = capacidade
        self.limpar()

    def armazenar(self, estado, acao, recompensa, prox_estado, finalizado,
                  log_prob, valor, h, c, mascara):
        self.estados.append(estado)
        self.acoes.append(acao)
        self.recompensas.append(recompensa)
        self.proximos_estados.append(prox_estado)
        self.finalizados.append(float(finalizado))
        self.log_probs.append(log_prob)
        self.valores.append(valor)
        self.estados_h.append(h.cpu().numpy())
        self.estados_c.append(c.cpu().numpy())
        self.mascaras.append(mascara)

    def calcular_vantagens_e_retornos(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        recompensas = list(self.recompensas)
        finalizados = list(self.finalizados)
        valores     = list(self.valores)

        n = len(recompensas)
        vantagens    = np.zeros(n, dtype=np.float32)
        retornos_arr = np.zeros(n, dtype=np.float32)
        prox_valor = 0.0
        vantagem   = 0.0

        for i in reversed(range(n)):
            if finalizados[i]:
                prox_valor = 0.0
                vantagem   = 0.0
            delta    = recompensas[i] + gamma * prox_valor - valores[i]
            vantagem = delta + gamma * gae_lambda * vantagem
            prox_valor = valores[i]
            vantagens[i]    = vantagem
            retornos_arr[i] = vantagem + valores[i]

        vantagens = (vantagens - vantagens.mean()) / (vantagens.std() + 1e-8)
        self.vantagens = vantagens
        self.retornos  = retornos_arr.tolist()

    def obter_lotes_sequencia(self, tamanho_lote: int, tam_sequencia: int = 64):
        """Gera lotes de sequências não sobrepostas para treinamento correto do LSTM."""
        # Materializa deques em listas para fatiamento eficiente
        estados    = list(self.estados)
        acoes      = list(self.acoes)
        log_probs  = list(self.log_probs)
        finalizados= list(self.finalizados)
        mascaras   = list(self.mascaras)
        valores    = list(self.valores)
        estados_h  = list(self.estados_h)
        estados_c  = list(self.estados_c)
        vantagens  = self.vantagens
        retornos   = self.retornos

        num_amostras  = len(estados)
        inicios_valid = np.arange(0, num_amostras - tam_sequencia + 1, tam_sequencia)
        np.random.shuffle(inicios_valid)

        seqs_por_lote = max(1, tamanho_lote // tam_sequencia)

        for i in range(0, len(inicios_valid), seqs_por_lote):
            lote_inicios = inicios_valid[i:i + seqs_por_lote]

            l_est, l_ac, l_lp, l_van, l_ret, l_fin = [], [], [], [], [], []
            l_mask, l_val_ant, l_h0, l_c0 = [], [], [], []

            for ini in lote_inicios:
                fim = ini + tam_sequencia
                l_est.append(estados[ini:fim])
                l_ac.append(acoes[ini:fim])
                l_lp.append(log_probs[ini:fim])
                l_van.append(vantagens[ini:fim])
                l_ret.append(retornos[ini:fim])
                l_fin.append(finalizados[ini:fim])
                l_mask.append(mascaras[ini:fim])
                l_val_ant.append(valores[ini:fim])
                l_h0.append(estados_h[ini])
                l_c0.append(estados_c[ini])

            yield (
                torch.FloatTensor(np.array(l_est)),
                torch.LongTensor(np.array(l_ac)),
                torch.FloatTensor(np.array(l_lp)),
                torch.FloatTensor(np.array(l_van)),
                torch.FloatTensor(np.array(l_ret)),
                torch.FloatTensor(np.array(l_fin)),
                torch.FloatTensor(np.array(l_mask)),
                torch.FloatTensor(np.array(l_val_ant)),
                torch.FloatTensor(np.concatenate(l_h0, axis=1)),
                torch.FloatTensor(np.concatenate(l_c0, axis=1)),
            )

    def limpar(self):
        self.estados          = deque(maxlen=self.capacidade)
        self.acoes            = deque(maxlen=self.capacidade)
        self.recompensas      = deque(maxlen=self.capacidade)
        self.proximos_estados = deque(maxlen=self.capacidade)
        self.finalizados      = deque(maxlen=self.capacidade)
        self.log_probs        = deque(maxlen=self.capacidade)
        self.valores          = deque(maxlen=self.capacidade)
        self.estados_h        = deque(maxlen=self.capacidade)
        self.estados_c        = deque(maxlen=self.capacidade)
        self.mascaras         = deque(maxlen=self.capacidade)
        self.vantagens        = []
        self.retornos         = []


class AgentePPOLSTMMascarado:
    """
    PPO-LSTM-Masked — combina memória temporal recorrente com mascaramento de ações.

    Melhorias sobre os agentes individuais:
      1. LSTM de 2 camadas — representação temporal mais profunda
      2. Mascaramento de ações durante coleta e atualização
      3. Value clipping (PPO v2) — estabiliza a estimativa de valor
      4. Parada antecipada por KL — evita atualizações destrutivas
      5. LayerNorm pós-LSTM — gradientes mais estáveis em sequências longas
      6. Monitoramento de eficiência da máscara por episódio
    """

    def __init__(self, espaco_observacao, espaco_acoes,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 faixa_corte: float = 0.2,
                 coef_valor: float = 0.5,
                 coef_entropia: float = 0.01,
                 norma_max_gradiente: float = 0.5,
                 epocas_ppo: int = 4,
                 tamanho_lote: int = 256,
                 tamanho_buffer: int = 2048,
                 tam_sequencia: int = 64,
                 dim_oculta: int = 256,
                 num_camadas_lstm: int = 2,
                 limite_kl: float = 0.02):

        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr                  = lr
        self.gamma               = gamma
        self.gae_lambda          = gae_lambda
        self.faixa_corte         = faixa_corte
        self.coef_valor          = coef_valor
        self.coef_entropia       = coef_entropia
        self.norma_max_gradiente = norma_max_gradiente
        self.epocas_ppo          = epocas_ppo
        self.tamanho_lote        = tamanho_lote
        self.tam_sequencia       = tam_sequencia
        self.dim_oculta          = dim_oculta
        self.num_camadas_lstm    = num_camadas_lstm
        self.limite_kl           = limite_kl
        self.espaco_acoes        = espaco_acoes

        dim_obs   = espaco_observacao.shape[0]
        dim_acoes = espaco_acoes.n

        self.rede = RedePPOLSTMMascarada(
            dim_obs=dim_obs,
            dim_acoes=dim_acoes,
            dim_oculta=dim_oculta,
            num_camadas_lstm=num_camadas_lstm,
        ).to(self.dispositivo)

        self.otimizador = optim.Adam(self.rede.parameters(), lr=lr, eps=1e-5)
        self.agendador  = None  # configurado pelo treinador após conhecer num_episodios
        self.buffer     = BufferReproducaoLSTMMascarado(tamanho_buffer)
        self.resetar_estado_oculto()

        self.estatisticas_treino = {
            'recompensas_episodios': [],
            'duracao_episodios':     [],
            'perdas_politica':       [],
            'perdas_valor':          [],
            'perdas_entropia':       [],
            'perdas_totais':         [],
            'kl_divergencias':       [],
            'taxas_aprendizado':     [],
            'eficiencia_mascara':    [],
        }

    def resetar_estado_oculto(self, tamanho_lote: int = 1):
        self.estado_oculto = (
            torch.zeros(self.num_camadas_lstm, tamanho_lote, self.dim_oculta).to(self.dispositivo),
            torch.zeros(self.num_camadas_lstm, tamanho_lote, self.dim_oculta).to(self.dispositivo),
        )

    def selecionar_acao(self, observacao, mascara, treinando: bool = True):
        obs_t  = torch.FloatTensor(observacao).unsqueeze(0).unsqueeze(0).to(self.dispositivo)
        mask_t = torch.FloatTensor(mascara).unsqueeze(0).unsqueeze(0).to(self.dispositivo)

        with torch.no_grad():
            logits, valor, novo_estado = self.rede(obs_t, self.estado_oculto, mask_t)

        logits = logits.squeeze(1)   # (1, dim_acoes)
        valor  = valor.squeeze()

        probs = F.softmax(logits, dim=-1)
        dist  = torch.distributions.Categorical(probs)
        acao  = dist.sample() if treinando else torch.argmax(probs, dim=-1)
        log_prob = dist.log_prob(acao)

        h_atual, c_atual = self.estado_oculto
        self.estado_oculto = novo_estado

        return acao.item(), log_prob.item(), valor.item(), (h_atual, c_atual)

    def atualizar(self):
        if len(self.buffer.estados) < self.tamanho_lote:
            return None, None, None

        self.buffer.calcular_vantagens_e_retornos(self.gamma, self.gae_lambda)

        soma_p = soma_v = soma_e = soma_kl = 0.0
        n_atualizacoes = 0
        parada_antecipada = False

        for _ in range(self.epocas_ppo):
            if parada_antecipada:
                break

            for lote in self.buffer.obter_lotes_sequencia(self.tamanho_lote, self.tam_sequencia):
                (estados, acoes, log_probs_ant, vantagens, retornos,
                 finalizados, mascaras, valores_antigos, h0, c0) = lote

                estados       = estados.to(self.dispositivo)
                acoes         = acoes.to(self.dispositivo)
                log_probs_ant = log_probs_ant.to(self.dispositivo)
                vantagens     = vantagens.to(self.dispositivo)
                retornos      = retornos.to(self.dispositivo)
                finalizados   = finalizados.to(self.dispositivo)
                mascaras      = mascaras.to(self.dispositivo)
                valores_antigos = valores_antigos.to(self.dispositivo)
                h0 = h0.to(self.dispositivo)
                c0 = c0.to(self.dispositivo)

                logits, valores, _ = self.rede(estados, (h0, c0), mascaras, finalizados)

                dim_a         = self.espaco_acoes.n
                logits_f      = logits.view(-1, dim_a)
                valores_f     = valores.view(-1)
                acoes_f       = acoes.view(-1)
                lp_ant_f      = log_probs_ant.view(-1)
                vant_f        = vantagens.view(-1)
                ret_f         = retornos.view(-1)
                val_ant_f     = valores_antigos.view(-1)

                probs = F.softmax(logits_f, dim=-1)
                dist  = torch.distributions.Categorical(probs)
                novos_lp = dist.log_prob(acoes_f)

                # Parada antecipada por KL
                kl = (lp_ant_f - novos_lp).mean().item()
                soma_kl += abs(kl)
                if abs(kl) > self.limite_kl * 1.5:
                    parada_antecipada = True
                    break

                # Perda da política (PPO clipped)
                razao  = torch.exp(novos_lp - lp_ant_f)
                surr1  = -vant_f * razao
                surr2  = -vant_f * torch.clamp(razao, 1 - self.faixa_corte, 1 + self.faixa_corte)
                perda_politica = torch.max(surr1, surr2).mean()

                # Perda do valor com clipping (PPO v2)
                valores_clip = val_ant_f + torch.clamp(
                    valores_f - val_ant_f, -self.faixa_corte, self.faixa_corte)
                perda_valor = torch.max(
                    F.mse_loss(valores_f, ret_f),
                    F.mse_loss(valores_clip, ret_f),
                )

                perda_entropia = -dist.entropy().mean()

                perda_total = (perda_politica
                               + self.coef_valor  * perda_valor
                               + self.coef_entropia * perda_entropia)

                self.otimizador.zero_grad()
                perda_total.backward()
                torch.nn.utils.clip_grad_norm_(self.rede.parameters(), self.norma_max_gradiente)
                self.otimizador.step()

                soma_p += perda_politica.item()
                soma_v += perda_valor.item()
                soma_e += perda_entropia.item()
                n_atualizacoes += 1

        self.buffer.limpar()

        if n_atualizacoes == 0:
            return None, None, None

        med_p  = soma_p  / n_atualizacoes
        med_v  = soma_v  / n_atualizacoes
        med_e  = soma_e  / n_atualizacoes
        med_kl = soma_kl / n_atualizacoes

        self.estatisticas_treino['perdas_politica'].append(med_p)
        self.estatisticas_treino['perdas_valor'].append(med_v)
        self.estatisticas_treino['perdas_entropia'].append(med_e)
        self.estatisticas_treino['perdas_totais'].append(med_p + med_v + med_e)
        self.estatisticas_treino['kl_divergencias'].append(med_kl)

        return med_p, med_v, med_e

    def salvar_modelo(self, caminho: str):
        torch.save({
            'network_state_dict':    self.rede.state_dict(),
            'optimizer_state_dict':  self.otimizador.state_dict(),
            'training_stats':        self.estatisticas_treino,
            'hyperparameters': {
                'lr':                  self.lr,
                'gamma':               self.gamma,
                'gae_lambda':          self.gae_lambda,
                'faixa_corte':         self.faixa_corte,
                'coef_valor':          self.coef_valor,
                'coef_entropia':       self.coef_entropia,
                'norma_max_gradiente': self.norma_max_gradiente,
                'epocas_ppo':          self.epocas_ppo,
                'tamanho_lote':        self.tamanho_lote,
                'tam_sequencia':       self.tam_sequencia,
                'num_camadas_lstm':    self.num_camadas_lstm,
                'limite_kl':           self.limite_kl,
            },
        }, caminho)
        print(f"Modelo salvo em {caminho}")

    def carregar_modelo(self, caminho: str):
        ckpt = torch.load(caminho, map_location=self.dispositivo, weights_only=False)
        self.rede.load_state_dict(ckpt['network_state_dict'])
        self.otimizador.load_state_dict(ckpt['optimizer_state_dict'])
        self.estatisticas_treino = ckpt['training_stats']
        print(f"Modelo carregado de {caminho}")


class TreinadorPPOLSTMMascarado:

    def __init__(self, env, agente, dir_resultados: str):
        self.env    = env
        self.agente = agente
        self.dir_execucao = os.path.join(
            dir_resultados, f"ppo_lstm_masked_exec_{int(time.time())}")
        os.makedirs(self.dir_execucao, exist_ok=True)
        print(f"Diretorio de execucao criado: {self.dir_execucao}")

    def treinar(self, num_episodios: int = 500, passos_maximos: int = 365,
                freq_salvamento: int = 100, freq_avaliacao: int = 50):

        print(f"Iniciando treinamento PPO-LSTM-Masked por {num_episodios} episodios...")
        print(f"Dispositivo:    {self.agente.dispositivo}")
        print(f"Camadas LSTM:   {self.agente.num_camadas_lstm}")
        print(f"Dim oculta:     {self.agente.dim_oculta}")
        print(f"Tamanho buffer: {self.agente.buffer.capacidade}")
        print()

        # Decaimento linear de LR: começa em lr, termina em 10% do lr
        self.agente.agendador = optim.lr_scheduler.LinearLR(
            self.agente.otimizador,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_episodios,
        )

        tempo_inicio = time.time()

        for episodio in range(num_episodios):
            obs, info = self.env.reset()
            mascara = info['mascara_acoes']
            self.agente.resetar_estado_oculto()

            rec_ep = dur_ep = 0
            acoes_validas = total_acoes = 0

            for _ in range(passos_maximos):
                acao, log_prob, valor, (h, c) = self.agente.selecionar_acao(
                    obs, mascara, treinando=True)

                total_acoes += 1
                if mascara[acao] > 0:
                    acoes_validas += 1

                prox_obs, recompensa, terminado, truncado, info = self.env.step(acao)
                prox_mascara = info['mascara_acoes']

                self.agente.buffer.armazenar(
                    obs, acao, recompensa, prox_obs, terminado or truncado,
                    log_prob, valor, h, c, mascara)

                obs, mascara = prox_obs, prox_mascara
                rec_ep += recompensa
                dur_ep += 1

                if terminado or truncado:
                    break

            eficiencia = acoes_validas / max(1, total_acoes)
            self.agente.estatisticas_treino['recompensas_episodios'].append(rec_ep)
            self.agente.estatisticas_treino['duracao_episodios'].append(dur_ep)
            self.agente.estatisticas_treino['eficiencia_mascara'].append(eficiencia)

            if len(self.agente.buffer.estados) >= self.agente.tamanho_lote:
                self.agente.atualizar()

            if self.agente.agendador is not None:
                self.agente.agendador.step()
                lr_atual = self.agente.otimizador.param_groups[0]['lr']
                self.agente.estatisticas_treino['taxas_aprendizado'].append(lr_atual)

            if (episodio + 1) % 50 == 0:
                ultimos = self.agente.estatisticas_treino['recompensas_episodios'][-50:]
                media = np.mean(ultimos)
                print(f"Ep {episodio+1:4d}/{num_episodios} | "
                      f"Recompensa Media (50): {media:8.2f} | "
                      f"Ultima: {rec_ep:8.2f} | "
                      f"Mascara: {eficiencia:.1%}")

            if (episodio + 1) % freq_salvamento == 0:
                self.agente.salvar_modelo(
                    os.path.join(self.dir_execucao, f"modelo_ep{episodio+1}.pt"))

            if (episodio + 1) % freq_avaliacao == 0:
                self._avaliar_agente(episodio + 1, passos_maximos)

        self.agente.salvar_modelo(os.path.join(self.dir_execucao, "modelo_final.pt"))
        self._salvar_estatisticas()
        self._gerar_graficos()

        tempo_total = time.time() - tempo_inicio
        print(f"\nTreinamento concluido em {tempo_total:.0f}s ({tempo_total/60:.1f} min)")
        return self.dir_execucao

    def _avaliar_agente(self, episodio: int, passos_maximos: int, num_ep_aval: int = 5):
        recompensas = []
        for _ in range(num_ep_aval):
            obs, info = self.env.reset()
            mascara = info['mascara_acoes']
            self.agente.resetar_estado_oculto()
            rec = 0
            for _ in range(passos_maximos):
                acao, _, _, _ = self.agente.selecionar_acao(obs, mascara, treinando=False)
                obs, r, feito, trunc, info = self.env.step(acao)
                mascara = info['mascara_acoes']
                rec += r
                if feito or trunc:
                    break
            recompensas.append(rec)

        media = np.mean(recompensas)
        desvio = np.std(recompensas)
        print(f"  Avaliacao ep {episodio}: {media:.2f} +/- {desvio:.2f}")

        with open(os.path.join(self.dir_execucao, f"avaliacao_ep{episodio}.json"), 'w') as f:
            json.dump({'episodio': episodio, 'media': media,
                       'desvio': desvio, 'recompensas': recompensas}, f, indent=2)

    def _salvar_estatisticas(self):
        caminho = os.path.join(self.dir_execucao, "estatisticas_treinamento.json")
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(self.agente.estatisticas_treino, f, indent=2)
        print(f"Estatisticas salvas em {caminho}")

    def _gerar_graficos(self):
        try:
            fig, eixos = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Desempenho PPO-LSTM-Masked', fontsize=16)
            stats = self.agente.estatisticas_treino

            def _plot(eixo, dados, titulo, xlabel, ylabel, janela=30):
                eixo.plot(dados, alpha=0.25, linewidth=0.6)
                if len(dados) >= janela:
                    mm = [np.mean(dados[max(0, i-janela):i+1]) for i in range(len(dados))]
                    eixo.plot(mm, color='red', linewidth=1.8, label=f'MM-{janela}')
                    eixo.legend(fontsize=8)
                eixo.set_title(titulo); eixo.set_xlabel(xlabel); eixo.set_ylabel(ylabel)
                eixo.grid(True, alpha=0.3)

            _plot(eixos[0, 0], stats['recompensas_episodios'],
                  'Recompensas por Episodio', 'Episodio', 'Recompensa')
            _plot(eixos[0, 1], stats['duracao_episodios'],
                  'Duracao dos Episodios', 'Episodio', 'Passos')
            _plot(eixos[0, 2], stats['eficiencia_mascara'],
                  'Eficiencia da Mascara', 'Episodio', 'Fracao Acoes Validas')

            if stats['perdas_politica']:
                _plot(eixos[1, 0], stats['perdas_politica'],
                      'Perda da Politica', 'Atualizacao', 'Perda')
            if stats['perdas_valor']:
                _plot(eixos[1, 1], stats['perdas_valor'],
                      'Perda do Valor', 'Atualizacao', 'Perda')
            if stats['kl_divergencias']:
                _plot(eixos[1, 2], stats['kl_divergencias'],
                      'Divergencia KL', 'Atualizacao', 'KL')

            plt.tight_layout()
            caminho = os.path.join(self.dir_execucao, "desempenho_treinamento.png")
            plt.savefig(caminho, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Graficos salvos em {caminho}")
        except Exception as e:
            print(f"Erro ao gerar graficos: {e}")
