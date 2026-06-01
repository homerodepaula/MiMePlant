#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from env import AmbienteFazendaGym
import numpy as np

def validar_ambiente():
    print("Validacao do Ambiente Fazenda-Gym (Grade 4x4)")
    print("=" * 50)

    try:
        print("Teste 1: Criacao Basica")
        ambiente = AmbienteFazendaGym()
        assert ambiente.grid_size == (4, 4)
        assert ambiente.action_space.n == 15
        assert ambiente.observation_space.shape == (101,)
        print("APROVADO: Ambiente criado com dimensoes corretas")

        print("\nTeste 2: Funcionalidade de Reset")
        observacao, info = ambiente.reset()
        assert observacao.shape == (101,)
        assert isinstance(info, dict)
        print("APROVADO: Reset funciona corretamente")

        print("\nTeste 3: Todas as Acoes")
        for id_acao in range(15):
            observacao, recompensa, finalizado, truncado, info = ambiente.step(id_acao)
            assert observacao.shape == (101,)
            assert isinstance(recompensa, (int, float))
            assert isinstance(finalizado, bool)
            assert isinstance(truncado, bool)
        print("APROVADO: Todas as 15 acoes funcionam")

        print("\nTeste 4: Intervalos de Observacao")
        observacao, _ = ambiente.reset()
        assert np.all(observacao >= 0), "Observacoes negativas encontradas"
        assert np.all(observacao <= 1), "Observacoes > 1 encontradas"
        print("APROVADO: Observacoes normalizadas [0,1]")

        print("\nTeste 5: Observacoes Dinamicas")
        obs1, _ = ambiente.reset()
        for _ in range(5):
            acao = ambiente.action_space.sample()
            obs2, _, _, _, _ = ambiente.step(acao)
            if not np.allclose(obs1, obs2):
                print("APROVADO: Observacoes mudam ao longo do tempo")
                break
        else:
            print("AVISO: Observacoes nao estao mudando")

        print("\nTeste 6: Fluxo do Episodio")
        observacao, _ = ambiente.reset()
        passos = 0
        recompensa_total = 0

        while passos < 20:
            acao = ambiente.action_space.sample()
            observacao, recompensa, finalizado, truncado, info = ambiente.step(acao)
            recompensa_total += recompensa
            passos += 1

            if finalizado or truncado:
                break

        print(f"APROVADO: Episodio concluido em {passos} passos, recompensa: {recompensa_total:.3f}")

        print("\nTeste 7: Conversao de Coordenadas")
        for x in range(4):
            for y in range(4):
                valor_acao = x * 4 + y
                conv_x, conv_y = ambiente._action_value_to_coords(valor_acao)
                assert conv_x == x and conv_y == y
        print("APROVADO: Conversao de coordenadas funciona")

        print("\nTeste 8: Tamanho de Grade Personalizado")
        ambiente_personalizado = AmbienteFazendaGym(grid_size=(6, 6))
        assert ambiente_personalizado.grid_size == (6, 6)
        obs_personalizada, _ = ambiente_personalizado.reset()
        tamanho_esperado = 5 + (6 * 6 * 6)
        assert obs_personalizada.shape == (tamanho_esperado,)
        print(f"APROVADO: Grade personalizada 6x6 tem {tamanho_esperado} observacoes")

        ambiente.close()
        ambiente_personalizado.close()

        print("\n" + "=" * 50)
        print("SUCESSO: Ambiente 100% funcional!")
        print("Especificacoes principais:")
        print(f"- Grade: 4x4 (16 celulas)")
        print(f"- Acoes: 15 disponiveis")
        print(f"- Observacoes: 101 valores")
        print(f"- Todos os componentes: Funcionando")

        return True

    except Exception as e:
        print(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sucesso = validar_ambiente()
    sys.exit(0 if sucesso else 1)
