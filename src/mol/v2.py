"""
MOL v2: versão corrigida do Map-Optimize-Learn.

Diferenças em relação ao método do artigo:
  - o objetivo da otimização é a posição de cada feature: features correlacionadas
    devem ocupar pixels vizinhos (objetivo do IGTD, Zhu et al. 2021);
  - o formato da imagem é fixo, o mais quadrado possível;
  - nenhuma feature é descartada pelo arranjo. A seleção, se usada, é separada (mRMR).

O erro minimizado é
    soma_{i<j} (R[i, j] - Q[pixel(i), pixel(j)])^2
com R = ranking das dissimilaridades 1 - |correlação| entre features e
Q = ranking das distâncias euclidianas entre pixels.
"""
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler



# Formato
def formato_quadrado(k):
    """Formato h x w com h*w = k, o mais próximo possível de um quadrado (h <= w)."""
    h = max(d for d in range(1, int(np.sqrt(k)) + 1) if k % d == 0)
    if h == 1 and k > 3:
        raise ValueError(f"{k} features não formam uma grade 2D (número primo). "
                         f"Use n_features={k - 1} para selecionar {k - 1} features por mRMR.")
    return h, k // h


# MAP: dissimilaridade entre features e seleção
def _ranking_normalizado(D):
    """Troca cada valor do triângulo superior pelo seu posto, normalizado para [0, 1]."""
    k = D.shape[0]
    iu = np.triu_indices(k, 1)
    postos = rankdata(D[iu], method='average')
    postos = (postos - 1) / max(len(postos) - 1, 1)
    R = np.zeros_like(D, dtype=float)
    R[iu] = postos
    return R + R.T


def dissimilaridade_features(Z):
    """d(i, j) = 1 - |correlação de Pearson entre as features i e j|."""
    return 1 - np.abs(np.corrcoef(Z, rowvar=False))


def distancia_pixels(h, w):
    """Distância euclidiana entre cada par de pixels de uma grade h x w."""
    coords = np.array([(r, c) for r in range(h) for c in range(w)], dtype=float)
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


def selecionar_mrmr(Z, y, k):
    """
    Seleção gulosa por relevância menos redundância (mRMR).
      relevância(j)  = |média da classe 1 - média da classe 0| (mesmo critério do Map euclidiano),
                       normalizada para [0, 1]
      redundância(j) = média de |correlação| entre j e as features já escolhidas
    """
    classes = np.unique(y)
    rel = np.zeros(Z.shape[1])
    n_pares = 0
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            rel += np.abs(Z[y == classes[a]].mean(0) - Z[y == classes[b]].mean(0))
            n_pares += 1
    rel = rel / max(n_pares, 1)
    rel = rel / (rel.max() + 1e-12)
    corr = np.abs(np.corrcoef(Z, rowvar=False))

    escolhidas = [int(np.argmax(rel))]
    while len(escolhidas) < k:
        resto = [j for j in range(Z.shape[1]) if j not in escolhidas]
        pontuacao = [rel[j] - corr[j, escolhidas].mean() for j in resto]
        escolhidas.append(resto[int(np.argmax(pontuacao))])
    return np.array(escolhidas)


# OPTIMIZE: arranjo das features nos pixels
def erro_arranjo(R, Q, pixel_de):
    """erro = soma_{i<j} (R[i, j] - Q[pixel(i), pixel(j)])^2"""
    Qp = Q[np.ix_(pixel_de, pixel_de)]
    return float(((R - Qp) ** 2).sum() / 2)


def otimizar_trocas(R, Q, n_iter=4000, n_reinicios=4, rng=None, maximizar=False):
    """
    Busca local: troca os pixels de duas features e mantém a troca se o erro melhorar.
    Repete a partir de vários arranjos iniciais aleatórios e devolve o melhor.
    maximizar=True procura o PIOR arranjo (usado só como controle).
    """
    rng = np.random.default_rng() if rng is None else rng
    k = R.shape[0]
    sinal = -1.0 if maximizar else 1.0
    melhor, melhor_erro, historico = None, np.inf, []
    for _ in range(n_reinicios):
        p = rng.permutation(k)
        e = sinal * erro_arranjo(R, Q, p)
        hist = [sinal * e]
        for _ in range(n_iter):
            i, j = rng.choice(k, 2, replace=False)
            p[i], p[j] = p[j], p[i]
            e_novo = sinal * erro_arranjo(R, Q, p)
            if e_novo < e:
                e = e_novo
            else:
                p[i], p[j] = p[j], p[i]
            hist.append(sinal * e)
        if e < melhor_erro:
            melhor, melhor_erro, historico = p.copy(), e, hist
    return melhor, sinal * melhor_erro, historico


def otimizar_pso(R, Q, n_pop=50, max_evals=5000, rng=None):
    """
    O mesmo problema resolvido com o PSO do projeto (optimizers.py).
    Cada partícula é um vetor de k números; a ordem deles (argsort) define a permutação.
    """
    from .optimizers import pso
    rng = np.random.default_rng() if rng is None else rng
    k = R.shape[0]
    res = pso(lambda x: erro_arranjo(R, Q, np.argsort(x)), np.zeros(k), np.ones(k), n_pop, max_evals, rng)
    return np.argsort(res.best_x), float(res.best_f), res.history


def correlacao_vizinhos(Z_img_cols, h, w):
    """Média de |correlação| entre pixels vizinhos (horizontal e vertical). Mede a qualidade do arranjo."""
    corr = np.abs(np.corrcoef(Z_img_cols, rowvar=False))
    vals = []
    for r in range(h):
        for c in range(w):
            a = r * w + c
            if c + 1 < w:
                vals.append(corr[a, a + 1])
            if r + 1 < h:
                vals.append(corr[a, a + w])
    return float(np.mean(vals))


# Classificador completo
class MOLv2Classifier(BaseEstimator, ClassifierMixin):
    """
    Parâmetros
    ----------
    n_features : None (usa todas) ou k (seleciona k features por mRMR).
    arranjo    : 'correlacao' - otimiza o arranjo (o método proposto)
                 'aleatorio'  - permutação aleatória fixa (controle)
                 'pior'       - maximiza o erro do arranjo (controle)
                 'ranking'    - ordem do Map original, linha a linha (como no MOL do artigo)
    otimizador : 'trocas' (busca local) ou 'pso' (enxame do projeto).
    modelo     : 'cnn' (imagem) ou 'mlp' (vetor, sem convolução; controle).
    semente_arranjo : semente da permutação aleatória e da busca do arranjo.
    random_state    : semente da inicialização da rede.
    """

    def __init__(self, n_features=None, arranjo='correlacao', otimizador='trocas', n_iter=4000,
                 modelo='cnn', learning_rate=1e-3, epochs=100, batch_size=32,
                 semente_arranjo=0, random_state=42):
        self.n_features = n_features
        self.arranjo = arranjo
        self.otimizador = otimizador
        self.n_iter = n_iter
        self.modelo = modelo
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.semente_arranjo = semente_arranjo
        self.random_state = random_state

    # Map + Optimize
    def ajustar_arranjo(self, X, y):
        """Ajusta escalonamento, seleção e arranjo. Não treina a rede."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        M = Z.shape[1]
        k = M if self.n_features is None else int(self.n_features)
        self.forma_ = formato_quadrado(k)
        h, w = self.forma_
        rng = np.random.default_rng(self.semente_arranjo)

        # MAP: quais features entram
        if k == M:
            selecionadas = np.arange(M)
        else:
            selecionadas = selecionar_mrmr(Z, y, k)
        Zs = Z[:, selecionadas]

        R = _ranking_normalizado(dissimilaridade_features(Zs))
        Q = _ranking_normalizado(distancia_pixels(h, w))

        # OPTIMIZE: em que pixel cada feature fica
        if self.arranjo == 'correlacao':
            if self.otimizador == 'pso':
                pixel_de, erro, hist = otimizar_pso(R, Q, rng=rng)
            else:
                pixel_de, erro, hist = otimizar_trocas(R, Q, self.n_iter, rng=rng)
        elif self.arranjo == 'pior':
            pixel_de, erro, hist = otimizar_trocas(R, Q, self.n_iter, rng=rng, maximizar=True)
        elif self.arranjo == 'aleatorio':
            pixel_de = rng.permutation(k)
            erro, hist = erro_arranjo(R, Q, pixel_de), []
        elif self.arranjo == 'ranking':
            from .algorithm import MapOptimizeLearn
            notas = MapOptimizeLearn(map_strategy='euclidean')._feature_importance(Zs, y)
            ordem = np.argsort(-notas, kind='stable')          # ordem[p] = feature no pixel p
            pixel_de = np.argsort(ordem)
            erro, hist = erro_arranjo(R, Q, pixel_de), []
        else:
            raise ValueError(f"arranjo desconhecido: {self.arranjo}")

        # colunas_[p] = índice (na tabela original) da feature que ocupa o pixel p
        feature_no_pixel = np.argsort(pixel_de)
        self.colunas_ = selecionadas[feature_no_pixel]
        self.erro_arranjo_ = erro
        self.historico_arranjo_ = hist
        self.correlacao_vizinhos_ = correlacao_vizinhos(Z[:, self.colunas_], h, w)
        return self

    def para_imagens(self, X):
        Z = self.scaler_.transform(np.asarray(X, dtype=float))[:, self.colunas_]
        h, w = self.forma_
        return Z.reshape(-1, h, w)

    # Learn
    def fit(self, X, y):
        import keras
        from keras import layers

        from .cnn import build_cnn

        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        self.ajustar_arranjo(X, y_idx)
        h, w = self.forma_
        k = h * w
        n_oculta = int(np.ceil((k + 1) / 2))

        keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)
        if self.modelo == 'cnn':
            entrada = self.para_imagens(X)[..., np.newaxis]
            self.model_ = build_cnn((h, w, 1), n_oculta, len(self.classes_), self.learning_rate)
        else:
            entrada = self.para_imagens(X).reshape(len(X), -1)
            self.model_ = keras.Sequential([
                keras.Input(shape=(k,)),
                layers.Dense(n_oculta, activation='relu'),
                layers.Dense(1, activation='sigmoid'),
            ])
            self.model_.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                                loss='binary_crossentropy')
        self.model_.fit(entrada, y_idx, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X):
        imgs = self.para_imagens(X)
        entrada = imgs[..., np.newaxis] if self.modelo == 'cnn' else imgs.reshape(len(imgs), -1)
        p = self.model_.predict(entrada, verbose=0)
        return np.hstack([1 - p, p]) if p.shape[1] == 1 else p

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
