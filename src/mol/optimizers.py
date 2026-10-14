"""
Otimizadores sem derivada usados no estágio Optimize do MOL.

Todos minimizam uma função f(x) numa caixa [lower, upper] e respeitam um
orçamento máximo de avaliações (no artigo: população 50, 500 avaliações).

- PSO   : Particle Swarm Optimization (w=0.8, c1=c2=1.8, como no artigo)
- EPSO  : Evolutionary PSO (Miranda & Fonseca, 2002), tau=0.8, comunicação 0.9
- ABC   : Artificial Bee Colony (Karaboga & Basturk, 2008), limit = (M + pop) / 2
- random: busca aleatória (referência mínima, não faz parte do artigo)
"""
import numpy as np


class _Budget:
    """Conta avaliações e guarda o melhor ponto já visto."""

    def __init__(self, func, max_evals):
        self.func = func
        self.max_evals = max_evals
        self.n_evals = 0
        self.best_x = None
        self.best_f = np.inf
        self.history = []
        self.snapshots = []  # posição da população a cada iteração (usado para visualização)

    def snap(self, X):
        self.snapshots.append(np.array(X, copy=True))

    @property
    def exhausted(self):
        return self.n_evals >= self.max_evals

    def __call__(self, x):
        f = self.func(x)
        self.n_evals += 1
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        self.history.append(self.best_f)
        return f


def random_search(func, lower, upper, n_pop, max_evals, rng, **_):
    budget = _Budget(func, max_evals)
    while not budget.exhausted:
        budget(rng.uniform(lower, upper))
    return budget


def pso(func, lower, upper, n_pop, max_evals, rng, w=0.8, c1=1.8, c2=1.8, **_):
    budget = _Budget(func, max_evals)
    span = upper - lower
    v_max = span

    X = rng.uniform(lower, upper, size=(n_pop, len(lower)))
    V = rng.uniform(-span, span, size=X.shape) * 0.1
    F = np.array([budget(x) for x in X])
    P, P_f = X.copy(), F.copy()
    budget.snap(X)

    while not budget.exhausted:
        g = P[np.argmin(P_f)]
        for i in range(n_pop):
            if budget.exhausted:
                break
            r1, r2 = rng.random(len(lower)), rng.random(len(lower))
            V[i] = w * V[i] + c1 * r1 * (P[i] - X[i]) + c2 * r2 * (g - X[i])
            V[i] = np.clip(V[i], -v_max, v_max)
            X[i] = np.clip(X[i] + V[i], lower, upper)
            f = budget(X[i])
            if f < P_f[i]:
                P[i], P_f[i] = X[i].copy(), f
        budget.snap(X)
    return budget


def epso(func, lower, upper, n_pop, max_evals, rng, tau=0.8, communication=0.9, **_):
    """
    EPSO: cada partícula carrega seus próprios pesos estratégicos
    (inércia, memória, cooperação, perturbação do global best), que são
    mutados a cada iteração. A partícula original e uma réplica com pesos
    mutados se movem; sobrevive a melhor (seleção evolutiva).
    """
    budget = _Budget(func, max_evals)
    dim = len(lower)
    span = upper - lower

    X = rng.uniform(lower, upper, size=(n_pop, dim))
    V = rng.uniform(-span, span, size=X.shape) * 0.1
    W = rng.random((n_pop, 4))  # [inércia, memória, cooperação, ruído no gbest]
    F = np.array([budget(x) for x in X])
    P, P_f = X.copy(), F.copy()
    budget.snap(X)

    def move(x, v, wts, p, g):
        g_star = g + wts[3] * rng.standard_normal(dim) * span
        comm = rng.random(dim) < communication
        v_new = wts[0] * v + wts[1] * (p - x) + wts[2] * (g_star - x) * comm
        v_new = np.clip(v_new, -span, span)
        return np.clip(x + v_new, lower, upper), v_new

    while not budget.exhausted:
        g = P[np.argmin(P_f)]
        for i in range(n_pop):
            if budget.exhausted:
                break
            W_rep = np.clip(W[i] + tau * rng.standard_normal(4), 0, 1)

            x_o, v_o = move(X[i], V[i], W[i], P[i], g)
            f_o = budget(x_o)
            if budget.exhausted:
                x_r, v_r, f_r = x_o, v_o, np.inf
            else:
                x_r, v_r = move(X[i], V[i], W_rep, P[i], g)
                f_r = budget(x_r)

            if f_r < f_o:
                X[i], V[i], W[i], f_new = x_r, v_r, W_rep, f_r
            else:
                X[i], V[i], f_new = x_o, v_o, f_o
            if f_new < P_f[i]:
                P[i], P_f[i] = X[i].copy(), f_new
        budget.snap(X)
    return budget


def abc(func, lower, upper, n_pop, max_evals, rng, limit=None, **_):
    """
    ABC com n_pop/2 fontes de alimento (abelhas empregadas = observadoras).
    limit: tentativas sem melhora antes da fonte ser abandonada (scout).
    """
    budget = _Budget(func, max_evals)
    dim = len(lower)
    n_sources = max(2, n_pop // 2)
    if limit is None:
        limit = (dim + n_pop) / 2

    X = rng.uniform(lower, upper, size=(n_sources, dim))
    F = np.array([budget(x) for x in X])
    trials = np.zeros(n_sources)
    budget.snap(X)

    def fitness(f):
        return 1 / (1 + f) if f >= 0 else 1 + abs(f)

    def try_neighbour(i):
        k = rng.choice([s for s in range(n_sources) if s != i])
        j = rng.integers(dim)
        v = X[i].copy()
        v[j] = np.clip(v[j] + rng.uniform(-1, 1) * (v[j] - X[k, j]), lower[j], upper[j])
        f = budget(v)
        if f < F[i]:
            X[i], F[i], trials[i] = v, f, 0
        else:
            trials[i] += 1

    while not budget.exhausted:
        # Abelhas empregadas
        for i in range(n_sources):
            if budget.exhausted:
                break
            try_neighbour(i)

        # Abelhas observadoras (seleção proporcional ao fitness)
        fit = np.array([fitness(f) for f in F])
        probs = fit / fit.sum()
        for _ in range(n_sources):
            if budget.exhausted:
                break
            try_neighbour(rng.choice(n_sources, p=probs))

        # Abelha exploradora
        worst = np.argmax(trials)
        if trials[worst] > limit and not budget.exhausted:
            X[worst] = rng.uniform(lower, upper)
            F[worst] = budget(X[worst])
            trials[worst] = 0
        budget.snap(X)
    return budget


OPTIMIZERS = {
    'random': random_search,
    'pso': pso,
    'epso': epso,
    'abc': abc,
}
