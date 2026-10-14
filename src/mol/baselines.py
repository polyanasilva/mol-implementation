"""
Baselines do artigo: ANN, LR, KNN, RF, XGB e LGBM, cada um em três estratégias
de seleção de features (FS):
    S - sem seleção
    E - embedded: SelectFromModel usando os pesos/importâncias do próprio modelo
    W - wrapper : RFE (Recursive Feature Elimination) com o próprio modelo

KNN não tem pesos nem importâncias, então só roda em S.

Cada combinação é um Pipeline (scaler -> seleção -> modelo), de forma que todo
ajuste acontece só no treino de cada fold.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def mlp_importance(estimator):
    """Importância de cada entrada da ANN = soma |pesos| da camada de entrada."""
    return np.abs(estimator.coefs_[0]).sum(axis=1)


def _base_models(random_state, n_features):
    """Modelos com os parâmetros padrão descritos no artigo."""
    models = {
        'ANN': MLPClassifier(hidden_layer_sizes=(max(1, (n_features + 1) // 2),),
                             solver='adam', learning_rate_init=1e-3, max_iter=100,
                             random_state=random_state),
        # liblinear aceita penalidade L1 e L2 (o grid do artigo testa as duas)
        'LR': LogisticRegression(solver='liblinear', max_iter=1000),
        'KNN': KNeighborsClassifier(algorithm='kd_tree', metric='euclidean'),
        'RF': RandomForestClassifier(n_estimators=100, bootstrap=True,
                                     random_state=random_state, n_jobs=1),
    }
    if XGBClassifier is not None:
        models['XGB'] = XGBClassifier(learning_rate=0.1, n_estimators=100,
                                      eval_metric='logloss', random_state=random_state,
                                      n_jobs=1, verbosity=0)
    if LGBMClassifier is not None:
        models['LGBM'] = LGBMClassifier(learning_rate=0.1, n_estimators=100,
                                        random_state=random_state, n_jobs=1, verbose=-1)
    return models


IMPORTANCE_GETTER = {'ANN': mlp_importance}
SUPPORTS_FS = {'ANN', 'LR', 'RF', 'XGB', 'LGBM'}


def build_baselines(grids, fs_strategies=('S', 'E', 'W'), rfe_step=0.1,
                    random_state=42, n_features=30):
    """
    Returns:
        lista de (nome, fs, pipeline, param_grid)
    """
    from sklearn.base import clone

    out = []
    for name, model in _base_models(random_state, n_features).items():
        grid = {f'clf__{k}': v for k, v in grids.get(name, {}).items()}

        for fs in fs_strategies:
            if fs != 'S' and name not in SUPPORTS_FS:
                continue

            getter = IMPORTANCE_GETTER.get(name, 'auto')
            sel_model = clone(model)
            if fs == 'S':
                selector = 'passthrough'
                fs_grid = {}
            elif fs == 'E':
                selector = SelectFromModel(sel_model, threshold='mean', importance_getter=getter)
                fs_grid = {}
            elif fs == 'W':
                selector = RFE(sel_model, step=rfe_step, importance_getter=getter)
                fs_grid = {'fs__n_features_to_select': grids.get('rfe_n_features', [5, 10, 15])}
            else:
                raise ValueError(f"Estratégia de FS desconhecida: {fs}")

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('fs', selector),
                ('clf', clone(model)),
            ])
            out.append((name, fs, pipe, {**grid, **fs_grid}))
    return out


def n_selected_features(pipeline, n_features):
    fs = pipeline.named_steps['fs']
    if fs == 'passthrough' or fs is None:
        return n_features
    return int(fs.get_support().sum())
