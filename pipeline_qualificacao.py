"""Pipeline da qualificação — estudo de sensibilidade a hiperparâmetros.

FOCO: apenas BC e DAgger, 5 conjuntos de hiperparâmetros, demonstrando a
importância da configuração. SEM estratégias de exploração (exp-categoria=det).

5 configurações (definidas em run_il_experiments.py -> IL_CONFIGS):
  - baseline   : lr=3e-4, dim_lstm=512, len_seq=32   (referência)
  - lr_baixo   : lr=5e-5                              (taxa de aprendizado baixa)
  - lr_alto    : lr=1e-3                              (taxa de aprendizado alta)
  - lstm_menor : dim_lstm=256                         (rede com metade da memória)
  - seq_longa  : len_seq=64                           (contexto temporal dobrado)

Cada configuração roda BC + DAgger × 5 sementes {42,123,2024,7,1337}, avaliados
in-distribution e em dois cenários OOD (dry/humid) — totalizando 25 runs.

Uso:
  python pipeline_qualificacao.py                # todas as 5 configs
  python pipeline_qualificacao.py --retomar      # pula runs já concluídos
"""
import os, sys, time, subprocess, argparse

RAIZ = os.path.dirname(os.path.abspath(__file__))
CONFIGS = ['baseline', 'lr_baixo', 'lr_alto', 'lstm_menor', 'seq_longa']
SEEDS = [42, 123, 2024, 7, 1337]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retomar', action='store_true', help='Pular runs já concluídos')
    args = ap.parse_args()

    t0 = time.time()
    n = len(CONFIGS)
    print('=' * 78)
    print('  QUALIFICAÇÃO — Estudo de Sensibilidade a Hiperparâmetros (BC + DAgger)')
    print(f'  {n} configs × {len(SEEDS)} seeds = {n * len(SEEDS)} runs (BC + DAgger cada)')
    print(f'  Configs: {CONFIGS}')
    print(f'  Seeds:   {SEEDS}')
    print('  Exploração: nenhuma (det) | Avaliação: in-dist + OOD dry + OOD humid')
    print('=' * 78)

    for i, cfg in enumerate(CONFIGS, 1):
        print(f'\n{"#"*78}\n#  CONFIG {i}/{n}: {cfg}\n{"#"*78}')
        cmd = [sys.executable, '-u', 'run_il_experiments.py',
               '--config', cfg,
               '--exp-categoria', 'det',
               '--seeds', *[str(s) for s in SEEDS]]
        if args.retomar:
            cmd.append('--retomar')
        proc = subprocess.run(cmd, cwd=RAIZ)
        if proc.returncode != 0:
            print(f'\n[ERRO] Config {cfg} falhou (código {proc.returncode}). Continuando...')
            continue
        print(f'\n  Config {cfg} concluída.')

    dt = time.time() - t0
    print('\n' + '=' * 78)
    print(f'  PIPELINE QUALIFICAÇÃO COMPLETO em {dt/3600:.2f}h ({dt/60:.0f} min)')
    print('=' * 78)
    print('  Próximo passo: python _analise_final.py')


if __name__ == '__main__':
    main()
