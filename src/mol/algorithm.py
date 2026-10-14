"""
Estágios Map e Optimize do Map-Optimize-Learn (Serra Neto, Dutra & Mollinetti, IJCNN 2022).

Cada amostra tabular com M features vira uma imagem h x w. O transformer segue a
interface do scikit-learn: tudo o que é aprendido (escala, ranking, formato e máscara)
vem só dos dados passados a fit().
"""
import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler

from .optimizers import OPTIMIZERS

MAP_STRATEGIES = ('euclidean', 'anova', 'pearson', 'fisher', 'gain_ratio',
                  'mutual_info', 'random_forest')


class MapOptimizeLearn(BaseEstimator, TransformerMixin):
    """
    map_strategy    : critério de ranking das features (ver MAP_STRATEGIES)
    optimizer       : 'pso', 'epso', 'abc' ou 'random'
    n_population    : tamanho da população (50 no artigo)
    max_evaluations : avaliações da função objetivo (500 no artigo)
    sample_order    : ordem das amostras na Eq. 1, que compara a amostra i com a i+1.
                      O artigo não especifica. 'class' ordena por classe (os pares ficam
                      quase todos dentro da mesma classe); 'original' mantém a ordem recebida.
    min_pixels      : menor h*w permitido (2 no artigo)
    scaler          : 'standard' ou 'minmax'
    """

    def __init__(self, map_strategy='euclidean', optimizer='pso',
                 n_population=50, max_evaluations=500, sample_order='class',
                 min_pixels=2, scaler='standard', random_state=42, verbose=0):
        self.map_strategy = map_strategy
        self.optimizer = optimizer
        self.n_population = n_population
        self.max_evaluations = max_evaluations
        self.sample_order = sample_order
        self.min_pixels = min_pixels
        self.scaler = scaler
        self.random_state = random_state
        self.verbose = verbose

    def _feature_importance(self, X, y):
        n_features = X.shape[1]
        classes = np.unique(y)

        if self.map_strategy == 'anova':
            scores, _ = f_classif(X, y)

        elif self.map_strategy == 'pearson':
            Xc = X - X.mean(axis=0)
            yc = y - y.mean()
            scores = np.abs(Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc) + 1e-12)

        elif self.map_strategy == 'euclidean':
            # distância entre os centróides das classes, média sobre os pares de classes
            centroids = np.array([X[y == c].mean(axis=0) for c in classes])
            scores = np.zeros(n_features)
            n_pairs = 0
            for a in range(len(classes)):
                for b in range(a + 1, len(classes)):
                    scores += np.abs(centroids[a] - centroids[b])
                    n_pairs += 1
            scores /= max(n_pairs, 1)

        elif self.map_strategy == 'fisher':
            mu = X.mean(axis=0)
            num = np.zeros(n_features)
            den = np.zeros(n_features)
            for c in classes:
                Xc = X[y == c]
                num += len(Xc) * (Xc.mean(axis=0) - mu) ** 2
                den += len(Xc) * Xc.var(axis=0)
            scores = num / (den + 1e-12)

        elif self.map_strategy == 'gain_ratio':
            # ganho de informação / split info, com cada feature em 10 faixas por quantil
            Xb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile',
                                  quantile_method='averaged_inverted_cdf').fit_transform(X)
            h_y = entropy(np.bincount(np.searchsorted(classes, y)))
            scores = np.zeros(n_features)
            for j in range(n_features):
                bins, counts = np.unique(Xb[:, j], return_counts=True)
                h_y_given_x = sum(
                    (n / len(y)) * entropy(np.unique(y[Xb[:, j] == b], return_counts=True)[1])
                    for b, n in zip(bins, counts)
                )
                split_info = entropy(counts)
                scores[j] = (h_y - h_y_given_x) / split_info if split_info > 0 else 0.0

        elif self.map_strategy == 'mutual_info':
            scores = mutual_info_classif(X, y, random_state=self.random_state)

        elif self.map_strategy == 'random_forest':
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            scores = rf.fit(X, y).feature_importances_

        else:
            raise ValueError(f"map_strategy desconhecida: {self.map_strategy}. "
                             f"Opções: {MAP_STRATEGIES}")

        return np.nan_to_num(np.asarray(scores, dtype=float))

    def _decode(self, x):
        """
        Converte o vetor do otimizador x = [h, w, s_1..s_M] numa solução viável
        da Eq. (1): h*w = número de features selecionadas, min_pixels <= h*w <= M.

        s_i > 0.5 indica feature selecionada. Se o número de selecionadas não
        couber exatamente em h*w, removemos as de menor s_i ou adicionamos as de
        maior s_i entre as não selecionadas (reparo guloso).
        """
        M = self.n_features_in_
        h = int(np.clip(np.rint(x[0]), 1, M))
        w = int(np.clip(np.rint(x[1]), 1, M))
        if h * w > M:
            w = max(1, M // h)
        if h * w < self.min_pixels:
            w = int(np.ceil(self.min_pixels / h))
            if h * w > M:
                h, w = 1, self.min_pixels
        k = h * w

        s = x[2:]
        selected = s > 0.5
        n_sel = selected.sum()
        if n_sel > k:
            keep = np.where(selected)[0]
            keep = keep[np.argsort(-s[keep], kind='stable')[:k]]
        elif n_sel < k:
            extra = np.where(~selected)[0]
            extra = extra[np.argsort(-s[extra], kind='stable')[:k - n_sel]]
            keep = np.concatenate([np.where(selected)[0], extra])
        else:
            keep = np.where(selected)[0]

        mask = np.zeros(M, dtype=bool)
        mask[keep] = True
        return (h, w), mask

    @staticmethod
    def _objective(X_pairs_a, X_pairs_b, mask, n_samples):
        """
        Eq. (1) do artigo:
            (1/N) * sum_{i=1}^{N-1} (1/SM) * sum_{j=1}^{SM} (X[i,j] - X[i+1,j])^2
        onde SM = h*w = número de features selecionadas.
        """
        diff = X_pairs_a[:, mask] - X_pairs_b[:, mask]
        return (diff ** 2).mean(axis=1).sum() / n_samples

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.min_pixels < 2:
            raise ValueError("min_pixels deve ser >= 2 (restrição da Eq. 1)")
        self.n_features_in_ = X.shape[1]
        if self.min_pixels > self.n_features_in_:
            raise ValueError("min_pixels maior que o número de features")

        self.scaler_ = StandardScaler() if self.scaler == 'standard' else MinMaxScaler()
        Xs = self.scaler_.fit_transform(X)

        self.feature_scores_ = self._feature_importance(Xs, y)
        self.feature_order_ = np.argsort(-self.feature_scores_, kind='stable')
        X_ordered = Xs[:, self.feature_order_]

        if self.sample_order == 'class':
            idx = np.argsort(y, kind='stable')
        elif self.sample_order == 'original':
            idx = np.arange(len(y))
        else:
            raise ValueError("sample_order deve ser 'class' ou 'original'")
        Xo = X_ordered[idx]
        a, b, n = Xo[:-1], Xo[1:], len(Xo)

        def func(x):
            _, mask = self._decode(x)
            return self._objective(a, b, mask, n)

        M = self.n_features_in_
        lower = np.concatenate([[1, 1], np.zeros(M)])
        upper = np.concatenate([[M, M], np.ones(M)])
        rng = np.random.default_rng(self.random_state)

        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer desconhecido: {self.optimizer}. Opções: {list(OPTIMIZERS)}")
        result = OPTIMIZERS[self.optimizer](func, lower, upper, self.n_population,
                                            self.max_evaluations, rng)

        self.image_shape_, self.selected_mask_ = self._decode(result.best_x)
        self.objective_value_ = result.best_f
        self.convergence_ = np.array(result.history)
        # colunas originais que formam a imagem, na ordem em que preenchem os pixels
        self.selected_features_ = self.feature_order_[self.selected_mask_]

        if self.verbose:
            print(f"    MOL[{self.map_strategy}/{self.optimizer}] shape={self.image_shape_} "
                  f"features={self.selected_mask_.sum()}/{M} objetivo={self.objective_value_:.4f}")
        return self

    def transform(self, X):
        if not hasattr(self, 'selected_features_'):
            raise ValueError("MOL não foi ajustado. Chame fit() primeiro.")
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        h, w = self.image_shape_
        return Xs[:, self.selected_features_].reshape(-1, h, w)
