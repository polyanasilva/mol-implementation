"""
Treina no conjunto de treino e salva em models/ os modelos finais, com os hiperparâmetros
escolhidos pelo grid search dos experimentos:

  mol_original/   melhor MOL do artigo (Map euclidiano + ABC + CNN)
  mol_v2/         MOL v2 (arranjo por correlação, imagem 5x6)
  lr.joblib       regressão logística (LR-S)
  ann.joblib      rede neural tabular (ANN-S)

Também grava models/metricas_teste.json com o desempenho de cada um no teste.

    python experimentos/salvar_modelos.py
"""
import ast
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import joblib
import pandas as pd

from mol.baselines import build_baselines
from mol.cnn import MOLCNNClassifier
from mol.data import prepare_data_pipeline
from mol.utils import calculate_metrics
from mol.v2 import MOLv2Classifier

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / 'models'
SEMENTE = 42


def metricas(modelo, dados):
    proba = modelo.predict_proba(dados['X_test'])[:, 1]
    m = calculate_metrics(dados['y_test'], modelo.predict(dados['X_test']), proba)
    return {k: round(float(v), 4) for k, v in m.items()}


def main():
    dados = prepare_data_pipeline()
    X, y = dados['X_train'], dados['y_train']
    principal = pd.read_csv(RAIZ / 'results' / 'principal' / 'results.csv').set_index('model')
    saida = {}

    # MOL original: o transformer (escala, ranking, formato, máscara) vai em joblib e a rede em .keras
    lr = json.loads(principal.loc['MOL-euclidean-abc', 'best_params'])['learning_rate']
    mol = MOLCNNClassifier(map_strategy='euclidean', optimizer='abc', learning_rate=lr,
                           random_state=SEMENTE).fit(X, y)
    pasta = MODELOS / 'mol_original'
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(mol.mol_, pasta / 'map_optimize.joblib')
    mol.model_.save(pasta / 'cnn.keras')
    saida['mol_original'] = {'forma': list(mol.mol_.image_shape_), 'learning_rate': lr, **metricas(mol, dados)}

    # MOL v2: guarda o que é preciso para montar a imagem (escala, colunas na ordem dos pixels, formato)
    v2_res = pd.read_csv(RAIZ / 'results' / 'mol_v2' / 'resultados.csv').set_index('nome')
    lr = float(v2_res.loc['30-v2-correlacao-s1', 'melhor_lr'])
    v2 = MOLv2Classifier(arranjo='correlacao', learning_rate=lr, random_state=SEMENTE).fit(X, y)
    pasta = MODELOS / 'mol_v2'
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump({'scaler': v2.scaler_, 'colunas': v2.colunas_, 'forma': v2.forma_}, pasta / 'arranjo.joblib')
    v2.model_.save(pasta / 'cnn.keras')
    saida['mol_v2'] = {'forma': list(v2.forma_), 'learning_rate': lr, **metricas(v2, dados)}

    # Baselines: o pipeline inteiro (escala + modelo) em um arquivo
    pipelines = {(nome, fs): pipe for nome, fs, pipe, _ in
                 build_baselines({}, ('S',), random_state=SEMENTE, n_features=X.shape[1])}
    for arquivo, nome in [('lr', 'LR'), ('ann', 'ANN')]:
        params = json.loads(principal.loc[f'{nome}-S', 'best_params'])
        params = {k: ast.literal_eval(v) if isinstance(v, str) and v.startswith('(') else v
                  for k, v in params.items()}
        pipe = pipelines[(nome, 'S')].set_params(**params).fit(X, y)
        joblib.dump(pipe, MODELOS / f'{arquivo}.joblib')
        saida[arquivo] = {'params': {k: str(v) for k, v in params.items()}, **metricas(pipe, dados)}

    (MODELOS / 'metricas_teste.json').write_text(json.dumps(saida, indent=2, ensure_ascii=False))
    for nome, m in saida.items():
        print(f"{nome:<14} acurácia balanceada no teste: {m['balanced_accuracy']:.4f}")


if __name__ == '__main__':
    main()
