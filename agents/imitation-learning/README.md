# Aprendizado por Imitação — BC + DAgger

Esta pasta contém o oráculo especialista e o agente de imitação.

## `bot_perfeito.py` — oráculo especialista (o "professor")
Política heurística com acesso ao estado interno do ambiente. Gera as
demonstrações usadas no treino e rotula os estados visitados pelo agente
durante o DAgger.

## `il_agent.py` — agente de Aprendizado por Imitação (LSTM)
Implementa **as duas fases em um único arquivo**, porque ambas compartilham a
mesma rede e o mesmo conjunto de dados (o DAgger parte do modelo já treinado por
BC e apenas agrega novos dados):

| Algoritmo | Onde está | Como rodar |
|---|---|---|
| **Behavioral Cloning (BC)** | método `treinar_bc()` | `python il_agent.py --so-bc` |
| **DAgger (Dataset Aggregation)** | método `treinar_dagger()` (usa `_coletar_dagger()`) | `python il_agent.py` (BC + DAgger) |

Classes principais:

- `RedeBCLSTM`     — a política (codificador + LSTM + cabeça de decisão).
- `ConjuntoDadosBC` — o conjunto de dados de treino (sequências para a LSTM).
- `AgenteBCLSTM`    — orquestra o BC (`treinar_bc`) e o DAgger (`treinar_dagger`).

No estudo de qualificação, os dois algoritmos são executados para cada uma das
5 configurações de hiperparâmetros pelo script `run_il_experiments.py` (na raiz
do repositório), que salva os resultados de BC em `results/experiments/il_bc/`
e os de DAgger em `results/experiments/il_dagger/`.
