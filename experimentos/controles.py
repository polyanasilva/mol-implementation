"""
Testes de controle: cada parte do MOL contribui para a classificação?

  1. Optimize: 15 soluções sorteadas (formato e máscara) contra PSO, EPSO e ABC.
  2. Formato e ordem: as mesmas 16 features em 16x1, 1x16, 2x8, 4x4 e 4x4 embaralhado.
  3. Imagem: as mesmas features numa rede comum, sem convolução.

Mesmo protocolo do experimento principal (split, 10 folds, grid da taxa de aprendizado,
100 épocas, acurácia balanceada). A escala e o ranking do Map são refeitos em cada fold.

    python experimentos/controles.py            # ~35 min com 4 processos
    python experimentos/controles.py --rapido
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
PASTA = RAIZ / 'results' / 'controles'

SEMENTE = 42
GRID_LR = [0.1, 0.01, 0.001, 0.0001]
N_ALEATORIAS = 15


# Classificador com solução FIXA (sem a etapa Optimize)
def criar_classificador():
    """Definido dentro de uma função para o Keras só ser importado nos processos filhos."""
    import keras
    from keras import layers
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.preprocessing import StandardScaler

    from mol.algorithm import MapOptimizeLearn
    from mol.cnn import build_cnn

    class MOLFixo(BaseEstimator, ClassifierMixin):
        """
        Map (ranking dentro do fold) + solução fixa + rede.

        posicoes  : posições do ranking que entram (ex.: (0, 1, ..., 15) = 16 primeiras)
        forma     : (h, w) da imagem
        permutacao: se dada, embaralha a ordem dos pixels (mesma para todos os pacientes)
        modelo    : 'cnn' (imagem h x w) ou 'mlp' (vetor, sem convolução)
        """

        def __init__(self, posicoes=(0, 1), forma=(1, 2), permutacao=None, modelo='cnn',
                     learning_rate=1e-3, epochs=100, batch_size=32, random_state=SEMENTE):
            self.posicoes = posicoes
            self.forma = forma
            self.permutacao = permutacao
            self.modelo = modelo
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.random_state = random_state

        def _entrada(self, X):
            Z = self.scaler_.transform(np.asarray(X, dtype=float))[:, self.colunas_]
            if self.modelo == 'mlp':
                return Z
            h, w = self.forma
            return Z.reshape(-1, h, w, 1)

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            self.scaler_ = StandardScaler().fit(X)
            Z = self.scaler_.transform(X)
            mol = MapOptimizeLearn(map_strategy='euclidean', random_state=self.random_state)
            ordem = np.argsort(-mol._feature_importance(Z, y), kind='stable')
            colunas = ordem[list(self.posicoes)]
            if self.permutacao is not None:
                colunas = colunas[list(self.permutacao)]
            self.colunas_ = colunas

            keras.backend.clear_session()
            keras.utils.set_random_seed(self.random_state)
            k = len(colunas)
            n_oculta = int(np.ceil((k + 1) / 2))
            if self.modelo == 'cnn':
                h, w = self.forma
                self.model_ = build_cnn((h, w, 1), n_oculta, 2, self.learning_rate)
            else:
                self.model_ = keras.Sequential([
                    keras.Input(shape=(k,)),
                    layers.Dense(n_oculta, activation='relu'),
                    layers.Dense(1, activation='sigmoid'),
                ])
                self.model_.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                                    loss='binary_crossentropy')
            self.model_.fit(self._entrada(X), y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
            return self

        def predict_proba(self, X):
            p = self.model_.predict(self._entrada(X), verbose=0)
            return np.hstack([1 - p, p])

        def predict(self, X):
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    return MOLFixo


def avaliar(config):
    """Grid search de uma configuração."""
    import warnings
    warnings.filterwarnings('ignore')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    from mol.data import load_breast_cancer_data, split_data

    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        X, y, _ = load_breast_cancer_data()
        X_tr, X_te, y_tr, y_te = split_data(X, y, 0.2, SEMENTE)

    MOLFixo = criar_classificador()
    est = MOLFixo(posicoes=tuple(config['posicoes']), forma=tuple(config['forma']),
                  permutacao=tuple(config['permutacao']) if config.get('permutacao') else None,
                  modelo=config['modelo'], epochs=config['epochs'])
    cv = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=SEMENTE)
    t0 = time.time()
    gs = GridSearchCV(est, {'learning_rate': config['grid_lr']}, cv=cv, scoring='balanced_accuracy', n_jobs=1)
    gs.fit(X_tr, y_tr)
    folds = [float(gs.cv_results_[f'split{k}_test_score'][gs.best_index_]) for k in range(config['n_splits'])]
    return {
        'nome': config['nome'], 'teste': config['teste'], 'modelo': config['modelo'],
        'forma': f"{config['forma'][0]}x{config['forma'][1]}", 'n_features': len(config['posicoes']),
        'cv_media': float(np.mean(folds)), 'cv_desvio': float(np.std(folds)),
        'teste_bal_acc': float(balanced_accuracy_score(y_te, gs.predict(X_te))),
        'melhor_lr': gs.best_params_['learning_rate'], 'folds': folds,
        'posicoes': list(config['posicoes']), 'tempo_s': round(time.time() - t0, 1),
    }


# Definição das configurações
def montar_configuracoes(rapido):
    import contextlib
    import io

    from mol.algorithm import MapOptimizeLearn
    from mol.data import prepare_data_pipeline

    M = 30
    base = {'epochs': 5 if rapido else 100, 'n_splits': 3 if rapido else 10,
            'grid_lr': [0.01] if rapido else GRID_LR}
    configs = []

    def add(nome, teste, posicoes, forma, modelo='cnn', permutacao=None):
        configs.append({**base, 'nome': nome, 'teste': teste, 'posicoes': [int(p) for p in posicoes],
                        'forma': list(forma), 'modelo': modelo,
                        'permutacao': [int(p) for p in permutacao] if permutacao is not None else None})

    # Teste 1: soluções aleatórias (mesma distribuição da população inicial dos enxames)
    mol = MapOptimizeLearn()
    mol.n_features_in_ = M
    rng = np.random.default_rng(2026)
    lower = np.concatenate([[1, 1], np.zeros(M)])
    upper = np.concatenate([[M, M], np.ones(M)])
    for i in range(3 if rapido else N_ALEATORIAS):
        (h, w), mask = mol._decode(rng.uniform(lower, upper))
        add(f'aleatoria-{i + 1:02d}', 1, np.where(mask)[0], (h, w))

    # Teste 2: as mesmas 16 features em formatos e ordens diferentes
    top16 = list(range(16))
    for h, w in [(16, 1), (1, 16), (2, 8), (4, 4)]:
        add(f'top16-{h}x{w}', 2, top16, (h, w))
    rng_p = np.random.default_rng(7)
    for i in range(1 if rapido else 5):
        add(f'top16-4x4-embaralhado-{i + 1}', 2, top16, (4, 4), permutacao=rng_p.permutation(16))

    # as 16 posições escolhidas pelo ABC (melhor MOL), ajustado no treino inteiro
    with contextlib.redirect_stdout(io.StringIO()):
        d = prepare_data_pipeline()
    abc = MapOptimizeLearn(map_strategy='euclidean', optimizer='abc').fit(d['X_train'], d['y_train'])
    abc16 = [int(k) for k in np.where(abc.selected_mask_)[0]]
    add('abc16-16x1', 2, abc16, (16, 1))
    add('abc16-4x4', 2, abc16, (4, 4))
    for i in range(1 if rapido else 3):
        add(f'abc16-4x4-embaralhado-{i + 1}', 2, abc16, (4, 4), permutacao=rng_p.permutation(16))

    # Teste 3: vetor numa rede comum (sem imagem)
    add('top16-vetor-MLP', 3, top16, (16, 1), modelo='mlp')
    add('abc16-vetor-MLP', 3, abc16, (16, 1), modelo='mlp')
    add('todas30-vetor-MLP', 3, range(30), (30, 1), modelo='mlp')
    add('todas30-5x6', 3, range(30), (5, 6))
    add('todas30-30x1', 3, range(30), (30, 1))
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rapido', action='store_true')
    ap.add_argument('--processos', type=int, default=4)
    args = ap.parse_args()

    configs = montar_configuracoes(args.rapido)
    print(f"{len(configs)} configurações, {args.processos} processos", flush=True)

    resultados = []
    with ProcessPoolExecutor(max_workers=args.processos) as pool:
        for r in pool.map(avaliar, configs):
            resultados.append(r)
            print(f"  [teste {r['teste']}] {r['nome']:<30} {r['forma']:>5} {r['n_features']:>2} feat  "
                  f"CV {r['cv_media']:.4f} ± {r['cv_desvio']:.4f}  ({r['tempo_s']}s)", flush=True)
            sufixo = '_rapido' if args.rapido else ''
            df = pd.DataFrame(resultados)
            df.drop(columns=['folds']).to_csv(PASTA / f'resultados{sufixo}.csv', index=False)
            pd.DataFrame({r_['nome']: r_['folds'] for r_ in resultados}).to_csv(
                PASTA / f'fold_scores{sufixo}.csv', index_label='fold')
    print("concluído")


if __name__ == '__main__':
    main()
