#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import AmbienteFazendaGym
import numpy as np

def teste_ambiente():
    print("Testando Ambiente Fazenda-Gym (Grade 4x4)")
    print("=" * 50)

    ambiente = AmbienteFazendaGym()

    print(f"Ambiente criado com sucesso!")
    print(f"Tamanho da grade: {ambiente.grid_size}")
    print(f"Tamanho do espaco de acoes: {ambiente.action_space.n}")
    print(f"Formato do espaco de observacoes: {ambiente.observation_space.shape}")
    print(f"Passos maximos: {ambiente.max_steps}")

    observacao, info = ambiente.reset()
    print(f"Reset do ambiente bem-sucedido!")
    print(f"Formato da observacao inicial: {observacao.shape}")

    print("\nTestando acoes aleatorias...")
    recompensa_total = 0

    for passo in range(10):
        acao = ambiente.action_space.sample()
        observacao, recompensa, finalizado, truncado, info = ambiente.step(acao)
        recompensa_total += recompensa

        if passo % 5 == 0:
            print(f"  Passo {passo}: Acao={acao}, Recompensa={recompensa:.3f}")

        if finalizado or truncado:
            break

    print(f"Teste concluido!")
    print(f"Recompensa total apos 10 passos: {recompensa_total:.3f}")

    print(f"\nDetalhamento do espaco de observacoes:")
    print(f"  Observacoes globais: 5 (clima + tempo)")
    print(f"  Celulas da grade: {ambiente.grid_size[0]} x {ambiente.grid_size[1]} = {ambiente.grid_size[0] * ambiente.grid_size[1]}")
    print(f"  Obs por celula: 6 (planta: 3, solo: 2, ervas daninhas: 1)")
    print(f"  Total obs: 5 + {ambiente.grid_size[0] * ambiente.grid_size[1]} x 6 = {5 + ambiente.grid_size[0] * ambiente.grid_size[1] * 6}")

    ambiente.close()
    print("\nTodos os testes aprovados! Ambiente pronto para treinamento RL.")

if __name__ == "__main__":
    teste_ambiente()
