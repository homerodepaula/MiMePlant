# MimePlant

Código e resultados da etapa de **qualificação de mestrado** sobre Aprendizado por
Imitação aplicado ao manejo agrícola simulado, no contexto da Aprendizagem por
Reforço Automatizada (AutoRL).

Um agente é treinado por imitação, a partir de um oráculo especialista, em um
simulador de fazenda. O algoritmo base é a **Clonagem Comportamental (BC)** e o
refinamento é o **DAgger**. O estudo mostra como a escolha dos hiperparâmetros de
treinamento afeta o desempenho e a robustez do agente, dentro e fora da
distribuição climática, motivando a calibração automática por Otimização de
Hiperparâmetros (HPO) prevista para a etapa final da dissertação.

> Este repositório contém apenas o código de experimentação e os resultados.
> O texto da dissertação não faz parte dele.

## Estrutura

```
environment/                Simulador da fazenda (interface Gymnasium):
                            clima, solo, plantas, pragas, ervas daninhas,
                            polinizadores, aves, instalacoes e o wrapper de
                            mascaramento de acoes (mascara_acoes.py).
agents/
  imitation-learning/       bot_perfeito.py  -> oraculo especialista (professor)
                            rede_lstm.py     -> rede LSTM + dataset (compartilhados)
                            bc.py            -> Behavioral Cloning (classe AgenteBC)
                            dagger.py        -> DAgger (AgenteDAgger, estende AgenteBC)
results/                    metricas, avaliacoes, analise consolidada e figuras

run_il_experiments.py       treina e avalia BC + DAgger de UMA configuracao
pipeline_qualificacao.py    orquestra as 5 configuracoes x 5 sementes
_bot_eval.py                avalia o oraculo (referencia)
_random_qualif.py           avalia um agente aleatorio (piso)
_bc_longo.py                controle: BC treinada por mais epocas
_analise_final.py           consolida tudo em results/analise_final.json
gerar_figuras_qualif.py     gera as figuras em results/figuras/
```

## Requisitos

Python 3.10+ e as dependências de `requirements.txt`:

```
pip install -r requirements.txt
```

Uma GPU com CUDA é opcional, mas acelera o treinamento.

## Reprodução

Execute os comandos a partir da raiz do repositório (os scripts usam caminhos
relativos). A ordem completa, do zero:

```
python agents/imitation-learning/bot_perfeito.py   # 1. demonstracoes do oraculo
python pipeline_qualificacao.py                    # 2. treina BC + DAgger (5 configs x 5 seeds)
python _bot_eval.py                                # 3. referencia: oraculo
python _random_qualif.py                           #    piso: agente aleatorio
python _bc_longo.py                                #    controle: BC-longo
python _analise_final.py                           # 4. analise consolidada
python gerar_figuras_qualif.py                     #    figuras
```

## Configurações de hiperparâmetros

Partindo de uma referência (`baseline`), cada configuração varia um eixo por vez:

| Configuração | Variação em relação ao baseline |
|---|---|
| baseline   | referência |
| lr_baixo   | taxa de aprendizado menor |
| lr_alto    | taxa de aprendizado maior |
| lstm_menor | memória recorrente menor |
| seq_longa  | janela temporal maior |

Cada configuração é treinada com as sementes `{42, 123, 2024, 7, 1337}` e avaliada
em regime normal (in-distribution) e em dois cenários fora da distribuição
(seco e úmido).

## Notas

- Os modelos treinados (`*.pt`) e o arquivo de demonstrações brutas
  (`trajetorias_especialista.json`) não são versionados por questão de tamanho;
  ambos são regenerados pelos scripts acima.
- O ambiente é uma adaptação do simulador Farm-gym (Maillard et al., 2023).
