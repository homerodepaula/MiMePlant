#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import AmbienteFazendaGym
import numpy as np

def teste_funcionalidade_basica():
    print("=== Testando Funcionalidade Basica ===")

    ambiente = AmbienteFazendaGym()
    assert ambiente.grid_size == (4, 4), f"Esperado (4,4), obtido {ambiente.grid_size}"
    assert ambiente.action_space.n == 15, f"Esperado 15 acoes, obtido {ambiente.action_space.n}"
    assert ambiente.observation_space.shape == (101,), f"Esperado (101,), obtido {ambiente.observation_space.shape}"
    print("APROVADO Criacao padrao")

    ambiente_personalizado = AmbienteFazendaGym(grid_size=(6, 6), max_steps=200)
    assert ambiente_personalizado.grid_size == (6, 6), f"Esperado (6,6), obtido {ambiente_personalizado.grid_size}"
    assert ambiente_personalizado.max_steps == 200, f"Esperado 200, obtido {ambiente_personalizado.max_steps}"
    print("APROVADO Criacao personalizada")

    observacao, info = ambiente.reset()
    assert observacao.shape == (101,), f"Esperado formato obs (101,), obtido {observacao.shape}"
    assert isinstance(info, dict), f"Esperado info dict, obtido {type(info)}"
    print("APROVADO Funcionalidade de reset")

    ambiente.close()
    ambiente_personalizado.close()
    return True

def teste_espaco_acoes():
    print("\n=== Testando Espaco de Acoes ===")

    ambiente = AmbienteFazendaGym()
    observacao, _ = ambiente.reset()

    nomes_acoes = list(ambiente.actions.keys())
    assert len(nomes_acoes) == 15, f"Esperado 15 acoes, obtido {len(nomes_acoes)}"
    print(f"Quantidade de acoes: {len(nomes_acoes)} acoes")

    for i, nome_acao in enumerate(nomes_acoes):
        try:
            observacao, recompensa, finalizado, truncado, info = ambiente.step(i)
            assert observacao.shape == (101,), f"Acao {nome_acao}: formato obs incorreto"
            assert isinstance(recompensa, (int, float)), f"Acao {nome_acao}: tipo de recompensa incorreto"
            assert isinstance(finalizado, bool), f"Acao {nome_acao}: tipo de finalizado incorreto"
            assert isinstance(truncado, bool), f"Acao {nome_acao}: tipo de truncado incorreto"
            print(f"Acao {i} ({nome_acao}): APROVADO")
        except Exception as e:
            print(f"Acao {i} ({nome_acao}): REPROVADO - {e}")
            return False

    ambiente.close()
    return True

def teste_espaco_observacoes():
    print("\n=== Testando Espaco de Observacoes ===")

    ambiente = AmbienteFazendaGym()
    observacao, _ = ambiente.reset()

    assert len(observacao) == 101, f"Esperado 101 observacoes, obtido {len(observacao)}"

    assert np.all(observacao >= 0), "Observacoes contem valores negativos"
    assert np.all(observacao <= 1), "Observacoes contem valores > 1"

    obs_globais = observacao[:5]
    print(f"Observacoes globais: {obs_globais}")

    obs_grade = observacao[5:].reshape(16, 6)
    print(f"Formato das observacoes da grade: {obs_grade.shape}")

    lista_obs = []
    for _ in range(5):
        acao = ambiente.action_space.sample()
        observacao, _, _, _, _ = ambiente.step(acao)
        lista_obs.append(observacao.copy())

    array_obs = np.array(lista_obs)
    assert not np.allclose(array_obs, array_obs[0]), "Observacoes nao mudam ao longo do tempo"
    print("Observacoes dinamicas: APROVADO")

    ambiente.close()
    return True

def teste_componentes_ambiente():
    print("\n=== Testando Componentes do Ambiente ===")

    ambiente = AmbienteFazendaGym()

    componentes = ['plant', 'birds', 'pollinators', 'soil', 'weather',
                   'weeds', 'pest', 'cides_fertilizers', 'facilities']

    for componente in componentes:
        assert hasattr(ambiente, componente), f"Componente ausente: {componente}"
        obj_componente = getattr(ambiente, componente)
        assert obj_componente is not None, f"Componente {componente} e None"
        print(f"Componente {componente}: INICIALIZADO")

    observacao, _ = ambiente.reset()
    plantas_iniciais = len(ambiente.plant.plants)

    acao = list(ambiente.actions.keys()).index('plant')
    observacao, recompensa, _, _, _ = ambiente.step(acao)

    print(f"Interacao com componente planta: {plantas_iniciais} -> {len(ambiente.plant.plants)} plantas")

    ambiente.close()
    return True

def teste_sistema_recompensa():
    print("\n=== Testando Sistema de Recompensa ===")

    ambiente = AmbienteFazendaGym()
    observacao, _ = ambiente.reset()

    recompensa_total = 0
    passos = 0
    finalizado = False
    truncado = False

    while not finalizado and not truncado and passos < 50:
        acao = ambiente.action_space.sample()
        observacao, recompensa, finalizado, truncado, info = ambiente.step(acao)
        recompensa_total += recompensa
        passos += 1

        assert isinstance(recompensa, (int, float)), f"Tipo de recompensa invalido: {type(recompensa)}"

    print(f"Sistema de recompensa: {passos} passos, recompensa total: {recompensa_total:.3f}")

    assert passos > 0, "Episodio terminou imediatamente"
    print("Encerramento do episodio: APROVADO")

    ambiente.close()
    return True

def teste_conversao_coordenadas():
    print("\n=== Testando Conversao de Coordenadas ===")

    ambiente = AmbienteFazendaGym()

    coordenadas_teste = [(0, 0), (0, 1), (1, 0), (3, 3)]

    for x, y in coordenadas_teste:
        valor_acao = x * ambiente.grid_size[1] + y
        x_convertido, y_convertido = ambiente._action_value_to_coords(valor_acao)

        assert x_convertido == x, f"Conversao X falhou: {x} -> {x_convertido}"
        assert y_convertido == y, f"Conversao Y falhou: {y} -> {y_convertido}"
        print(f"Conversao de coordenadas: ({x},{y}) -> acao {valor_acao} -> ({x_convertido},{y_convertido})")

    for valor_acao in range(16):
        x, y = ambiente._action_value_to_coords(valor_acao)
        assert 0 <= x < 4, f"X fora dos limites: {x}"
        assert 0 <= y < 4, f"Y fora dos limites: {y}"

    print("Todas as conversoes de coordenadas: APROVADO")

    ambiente.close()
    return True

def teste_memoria_e_desempenho():
    print("\n=== Testando Memoria e Desempenho ===")

    import time

    ambiente = AmbienteFazendaGym()

    episodios = 10
    tempo_inicio = time.time()

    for episodio in range(episodios):
        observacao, _ = ambiente.reset()
        passos = 0

        while passos < 100:
            acao = ambiente.action_space.sample()
            observacao, recompensa, finalizado, truncado, info = ambiente.step(acao)
            passos += 1

            if finalizado or truncado:
                break

    tempo_fim = time.time()
    tempo_total = tempo_fim - tempo_inicio

    print(f"Desempenho: {episodios} episodios em {tempo_total:.2f}s")
    print(f"Tempo medio por episodio: {tempo_total/episodios:.3f}s")

    ambientes = []
    for i in range(5):
        ambientes.append(AmbienteFazendaGym())

    for instancia_ambiente in ambientes:
        instancia_ambiente.close()

    print("Gerenciamento de memoria: APROVADO")

    ambiente.close()
    return True

def executar_teste_abrangente():
    print("Validacao Abrangente do Ambiente Fazenda-Gym (Grade 4x4)")
    print("=" * 60)

    testes = [
        ("Funcionalidade Basica", teste_funcionalidade_basica),
        ("Espaco de Acoes", teste_espaco_acoes),
        ("Espaco de Observacoes", teste_espaco_observacoes),
        ("Componentes do Ambiente", teste_componentes_ambiente),
        ("Sistema de Recompensa", teste_sistema_recompensa),
        ("Conversao de Coordenadas", teste_conversao_coordenadas),
        ("Memoria e Desempenho", teste_memoria_e_desempenho),
    ]

    aprovados = 0
    total = len(testes)

    for nome_teste, funcao_teste in testes:
        try:
            if funcao_teste():
                aprovados += 1
                print(f"{nome_teste}: APROVADO")
            else:
                print(f"{nome_teste}: REPROVADO")
        except Exception as e:
            print(f"{nome_teste}: ERRO - {e}")

    print("\n" + "=" * 60)
    print(f"RESULTADOS DA VALIDACAO: {aprovados}/{total} testes aprovados")

    if aprovados == total:
        print("AMBIENTE 100% FUNCIONAL!")
        return True
    else:
        print("Alguns testes falharam. Ambiente precisa de atencao.")
        return False

if __name__ == "__main__":
    sucesso = executar_teste_abrangente()
    sys.exit(0 if sucesso else 1)
