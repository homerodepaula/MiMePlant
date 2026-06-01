#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import AmbienteFazendaGym
import numpy as np

def teste_integracao_componentes():
    print("Testando Integracao de Componentes")
    print("-" * 30)

    ambiente = AmbienteFazendaGym()
    observacao, _ = ambiente.reset()

    print(f"Plantas iniciais: {len(ambiente.plant.plants)}")
    print(f"Clima: {ambiente.weather.get_observation()}")
    print(f"Qualidade media do solo: {np.mean([solo.quality for solo in ambiente.soil.soil_state.values()]):.3f}")

    print("\nTestando ciclo de plantio...")

    posicoes_plantio = [(0, 0), (1, 1), (2, 2)]
    for x, y in posicoes_plantio:
        acao = list(ambiente.actions.keys()).index('plant')
        observacao, recompensa, _, _, _ = ambiente.step(acao)
        print(f"Plantado em ({x},{y}), recompensa: {recompensa:.1f}")

    print(f"Plantas apos plantio: {len(ambiente.plant.plants)}")

    print("\nTestando fertilizacao...")
    for _ in range(3):
        acao = list(ambiente.actions.keys()).index('fertilize_N')
        observacao, recompensa, _, _, _ = ambiente.step(acao)
        print(f"Fertilizado, recompensa: {recompensa:.1f}")

    print("\nTestando irrigacao...")
    acao = list(ambiente.actions.keys()).index('water')
    observacao, recompensa, _, _, _ = ambiente.step(acao)
    print(f"Irrigado, recompensa: {recompensa:.1f}")

    print("\nTestando instalacoes...")
    acao = list(ambiente.actions.keys()).index('put_scarecrow_basic')
    observacao, recompensa, _, _, _ = ambiente.step(acao)
    print(f"Espantalho instalado, recompensa: {recompensa:.1f}")

    print("\nExecutando simulacao de crescimento...")
    for passo in range(10):
        acao = ambiente.action_space.sample()
        observacao, recompensa, finalizado, truncado, info = ambiente.step(acao)

        if passo % 5 == 0:
            print(f"Passo {passo}: Total de plantas: {len(ambiente.plant.plants)}")

        if finalizado or truncado:
            break

    print("\nTestando colheita...")
    contagem_colheita = 0
    for x, y in posicoes_plantio:
        if (x, y) in ambiente.plant.plants:
            acao = list(ambiente.actions.keys()).index('harvest')
            observacao, recompensa, _, _, _ = ambiente.step(acao)
            if recompensa > 0:
                contagem_colheita += 1
                print(f"Colhido em ({x},{y}), recompensa: {recompensa:.1f}")

    print(f"Colheitas bem-sucedidas: {contagem_colheita}/{len(posicoes_plantio)} culturas")

    print(f"\nEstado final do ambiente:")
    print(f"- Total de plantas: {len(ambiente.plant.plants)}")
    print(f"- Total colhido: {sum(ambiente.harvest_history):.1f}")
    print(f"- Recompensa do episodio: {ambiente.episode_reward:.1f}")
    print(f"- Qualidade media do solo: {np.mean([solo.quality for solo in ambiente.soil.soil_state.values()]):.3f}")

    ambiente.close()
    return True

def teste_casos_limite():
    print("\nTestando Casos Limite")
    print("-" * 20)

    ambiente = AmbienteFazendaGym()

    print("Testando multiplos resets...")
    for i in range(3):
        observacao, _ = ambiente.reset()
        assert observacao.shape == (101,)
        print(f"Reset {i+1}: OK")

    print("Testando limites de acoes...")
    observacao, _ = ambiente.reset()

    for acao in range(15):
        try:
            observacao, recompensa, finalizado, truncado, info = ambiente.step(acao)
            assert observacao.shape == (101,)
        except Exception as e:
            print(f"ERRO com acao {acao}: {e}")
            return False

    print("Todas as acoes tratadas corretamente")

    print("Testando episodio longo...")
    observacao, _ = ambiente.reset()
    passos = 0
    while passos < 100:
        acao = ambiente.action_space.sample()
        observacao, recompensa, finalizado, truncado, info = ambiente.step(acao)
        passos += 1

        if finalizado or truncado:
            print(f"Episodio encerrado no passo {passos}")
            break

    print(f"Teste de episodio longo concluido")

    ambiente.close()
    return True

def executar_testes_integracao():
    print("Testes de Integracao Fazenda-Gym")
    print("=" * 40)

    testes = [
        teste_integracao_componentes,
        teste_casos_limite
    ]

    aprovados = 0
    for teste in testes:
        try:
            if teste():
                aprovados += 1
                print("APROVADO")
            else:
                print("REPROVADO")
        except Exception as e:
            print(f"ERRO: {e}")

    print(f"\nTestes de integracao: {aprovados}/{len(testes)} aprovados")
    return aprovados == len(testes)

if __name__ == "__main__":
    sucesso = executar_testes_integracao()
    print(f"\nResultado geral: {'SUCESSO' if sucesso else 'FALHA'}")
    sys.exit(0 if sucesso else 1)
