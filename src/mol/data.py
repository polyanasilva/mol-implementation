"""
Leitura e divisão da base Breast Cancer Wisconsin (CSV do Kaggle).

A normalização não é feita aqui: cada modelo ajusta a sua escala só com o treino de cada fold.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[2]
CSV_PADRAO = RAIZ / 'data' / 'breast-cancer.csv'
POSITIVE_LABEL = 'M'  # maligno = 1, benigno = 0
CLASS_NAMES = ['Benigno', 'Maligno']


def load_breast_cancer_data(path=CSV_PADRAO):
    """Devolve X (N x 30), y (1 = maligno) e os nomes das features."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c == 'id' or c.startswith('Unnamed')])

    y = (df.pop('diagnosis') == POSITIVE_LABEL).astype(int).values
    X = df.values.astype(float)
    feature_names = list(df.columns)

    if np.isnan(X).any():
        raise ValueError(f"{int(np.isnan(X).sum())} valores faltantes no dataset")

    print(f"Dataset: {X.shape[0]} amostras, {X.shape[1]} features "
          f"({np.sum(y == 0)} benignos, {np.sum(y == 1)} malignos)")
    return X, y, feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"Treino: {len(y_train)} ({y_train.sum()} malignos) | "
          f"Teste: {len(y_test)} ({y_test.sum()} malignos)")
    return X_train, X_test, y_train, y_test


def prepare_data_pipeline(path=CSV_PADRAO, test_size=0.2, random_state=42):
    X, y, feature_names = load_breast_cancer_data(path)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
    }
