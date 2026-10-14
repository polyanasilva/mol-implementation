"""
Experimento do MOL v2 com controles.

Se colocar features correlacionadas em pixels vizinhos ajuda a CNN, o arranjo otimizado
tem de superar as mesmas features em posições aleatórias e no pior arranjo possível.

  30 features (5x6): arranjo por correlação, ordem do Map original, aleatório, pior
                     arranjo e rede comum no vetor
  16 features mRMR (4x4): arranjo por correlação, aleatório e rede comum

Mesmo protocolo do experimento principal; cada grupo roda com 3 sementes de rede
(5 permutações no aleatório de 30 features).

    python experimentos/mol_v2.py            # ~25 min com 4 processos
    python experimentos/mol_v2.py --rapido
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / 'results' / 'mol_v2'

SEMENTE = 42
GRID_LR = [0.1, 0.01, 0.001, 0.0001]


def avaliar(config):
    import warnings
    warnings.filterwarnings('ignore')
    import contextlib
    import io

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from sklearn.metrics import balanced_accuracy_score, recall_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    from mol.data import load_breast_cancer_data, split_data
    from mol.v2 import MOLv2Classifier

    with contextlib.redirect_stdout(io.StringIO()):
        X, y, _ = load_breast_cancer_data()
        X_tr, X_te, y_tr, y_te = split_data(X, y, 0.2, SEMENTE)

    est = MOLv2Classifier(n_features=config['n_features'], arranjo=config['arranjo'], modelo=config['modelo'],
                          semente_arranjo=config['semente_arranjo'], random_state=config['semente_rede'],
                          epochs=config['epochs'])
    cv = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=SEMENTE)
    t0 = time.time()
    gs = GridSearchCV(est, {'learning_rate': config['grid_lr']}, cv=cv, scoring='balanced_accuracy', n_jobs=1)
    gs.fit(X_tr, y_tr)
    folds = [float(gs.cv_results_[f'split{k}_test_score'][gs.best_index_]) for k in range(config['n_splits'])]
    pred = gs.predict(X_te)
    m = gs.best_estimator_
    return {
        'nome': config['nome'], 'grupo': config['grupo'], 'n_features': m.forma_[0] * m.forma_[1],
        'forma': f"{m.forma_[0]}x{m.forma_[1]}", 'arranjo': config['arranjo'], 'modelo': config['modelo'],
        'cv_media': float(np.mean(folds)), 'cv_desvio': float(np.std(folds)),
        'teste_bal_acc': float(balanced_accuracy_score(y_te, pred)),
        'teste_sensibilidade': float(recall_score(y_te, pred, pos_label=1)),
        'erro_arranjo': m.erro_arranjo_, 'correlacao_vizinhos': m.correlacao_vizinhos_,
        'melhor_lr': gs.best_params_['learning_rate'], 'tempo_s': round(time.time() - t0, 1), 'folds': folds,
    }


def montar_configuracoes(rapido):
    base = {'epochs': 5 if rapido else 100, 'n_splits': 3 if rapido else 10,
            'grid_lr': [0.01] if rapido else GRID_LR}
    n = 1 if rapido else 3
    configs = []

    def add(grupo, n_features, arranjo, modelo='cnn', semente_arranjo=0, semente_rede=SEMENTE, sufixo=''):
        configs.append({**base, 'nome': f'{grupo}{sufixo}', 'grupo': grupo, 'n_features': n_features,
                        'arranjo': arranjo, 'modelo': modelo, 'semente_arranjo': semente_arranjo,
                        'semente_rede': semente_rede})

    for s in range(n):
        add('30-v2-correlacao', None, 'correlacao', semente_rede=SEMENTE + s, sufixo=f'-s{s + 1}')
        add('30-ranking', None, 'ranking', semente_rede=SEMENTE + s, sufixo=f'-s{s + 1}')
        add('30-pior', None, 'pior', semente_rede=SEMENTE + s, sufixo=f'-s{s + 1}')
        add('30-vetor-MLP', None, 'ranking', modelo='mlp', semente_rede=SEMENTE + s, sufixo=f'-s{s + 1}')
        add('16-v2-correlacao', 16, 'correlacao', semente_rede=SEMENTE + s, sufixo=f'-s{s + 1}')
    for p in range(1 if rapido else 5):
        add('30-aleatorio', None, 'aleatorio', semente_arranjo=100 + p, sufixo=f'-p{p + 1}')
    for p in range(n):
        add('16-aleatorio', 16, 'aleatorio', semente_arranjo=200 + p, sufixo=f'-p{p + 1}')
    add('16-vetor-MLP', 16, 'ranking', modelo='mlp', sufixo='-s1')
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rapido', action='store_true')
    ap.add_argument('--processos', type=int, default=4)
    args = ap.parse_args()

    configs = montar_configuracoes(args.rapido)
    sufixo = '_rapido' if args.rapido else ''
    print(f"{len(configs)} configurações, {args.processos} processos", flush=True)
    resultados = []
    with ProcessPoolExecutor(max_workers=args.processos) as pool:
        for r in pool.map(avaliar, configs):
            resultados.append(r)
            print(f"  {r['nome']:<24} {r['forma']:>4}  CV {r['cv_media']:.4f} ± {r['cv_desvio']:.4f}  "
                  f"teste {r['teste_bal_acc']:.4f}  |corr| viz {r['correlacao_vizinhos']:.2f}  ({r['tempo_s']}s)",
                  flush=True)
            pd.DataFrame(resultados).drop(columns=['folds']).to_csv(PASTA / f'resultados{sufixo}.csv', index=False)
            pd.DataFrame({x['nome']: x['folds'] for x in resultados}).to_csv(
                PASTA / f'fold_scores{sufixo}.csv', index_label='fold')
    print("concluído")


if __name__ == '__main__':
    main()
