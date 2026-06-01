# Aprendizado por Imitação — BC + DAgger

Esta pasta contém o oráculo especialista e o agente de imitação, com o
**Behavioral Cloning** e o **DAgger** em arquivos separados.

## `bot_perfeito.py` — oráculo especialista (o "professor")
Política heurística com acesso ao estado interno do ambiente. Gera as
demonstrações usadas no treino e rotula os estados visitados pelo agente
durante o DAgger.

## `rede_lstm.py` — rede e dataset compartilhados
`RedeBCLSTM` (a política: codificador FC + LSTM + cabeça de decisão) e
`ConjuntoDadosBC` (organiza os episódios em sequências para o LSTM). São usados
tanto pelo BC quanto pelo DAgger, por isso ficam em um módulo único.

## `bc.py` — Behavioral Cloning
`class AgenteBC`: aprendizado supervisionado direto nas demonstrações do oráculo
(método `treinar_bc`). Reúne a infraestrutura compartilhada do agente (rede,
normalização, avaliação, seleção de ação, persistência).

```
python bc.py            # treina BC e avalia
```

## `dagger.py` — DAgger (Dataset Aggregation)
`class AgenteDAgger(AgenteBC)`: **estende** o BC. Parte do modelo treinado por
Behavioral Cloning e, em rodadas iterativas, coleta dados nos estados que o
próprio agente visita (`_coletar_dagger`), rotulados pelo oráculo, e refina a
rede (`treinar_dagger`). Isso corrige o *covariate shift* que o BC puro não cobre.

```
python dagger.py            # BC + DAgger
python dagger.py --so-bc    # só Behavioral Cloning
```

---

No estudo de qualificação, BC e DAgger são executados para cada uma das 5
configurações de hiperparâmetros pelo `run_il_experiments.py` (na raiz do
repositório), que salva os resultados de BC em `results/experiments/il_bc/` e os
de DAgger em `results/experiments/il_dagger/`.
