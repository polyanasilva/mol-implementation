"""
Gera apresentacao/dados/mol_dados.json com os números usados nas cenas do Manim.

Roda no ambiente dos EXPERIMENTOS (usa o pacote mol e o TensorFlow), não no da apresentação.
As cenas só leem o JSON, então o ambiente do Manim não precisa dessas dependências.

    python apresentacao/gerar_dados.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer

from mol.algorithm import MapOptimizeLearn
from mol.data import load_breast_cancer_data, split_data

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / 'results'

SAIDA = Path(__file__).parent / 'dados'
SAIDA.mkdir(exist_ok=True)

CRITERIOS = ['euclidean', 'anova', 'pearson', 'fisher', 'gain_ratio', 'mutual_info', 'random_forest']


def pacientes_tipicos(Xs, y, classe, n):
    """Os n pacientes da classe mais próximos do centróide (evita outliers)."""
    idx = np.where(y == classe)[0]
    centro = Xs[idx].mean(axis=0)
    dist = np.linalg.norm(Xs[idx] - centro, axis=1)
    return idx[np.argsort(dist)[:n]]


def r(v, casas=4):
    return np.round(np.asarray(v, dtype=float), casas).tolist()


def dados_optimize(Z, y_tr, ordem, ex, benignos, malignos):
    """
    Dados do estágio Optimize, refazendo exatamente o que MapOptimizeLearn.fit faz:
    amostras ordenadas por classe, Eq. (1) com decodificação, mesmos otimizadores e sementes.
    """
    import csv
    from mol.optimizers import OPTIMIZERS

    M = Z.shape[1]
    mol = MapOptimizeLearn(map_strategy='euclidean')
    mol.n_features_in_ = M
    Zo = Z[:, ordem]                                  # colunas na ordem do Map
    idx = np.argsort(y_tr, kind='stable')             # benignos primeiro, depois malignos
    Xo, y_ord = Zo[idx], y_tr[idx]
    a, b, n = Xo[:-1], Xo[1:], len(Xo)

    def f_mask(mask):
        return mol._objective(a, b, mask, n)

    def func(x):
        return f_mask(mol._decode(x)[1])

    lower = np.concatenate([[1, 1], np.zeros(M)])
    upper = np.concatenate([[M, M], np.ones(M)])

    # Contribuição de cada feature (na ordem do ranking): D_j
    D = ((a - b) ** 2).sum(axis=0) / n
    D_ordenado = np.sort(D)
    f_por_k = [float(D_ordenado[:k].mean()) for k in range(2, M + 1)]          # k menores D_j
    f_top_k = [float(D[:k].mean()) for k in range(2, M + 1)]                    # k primeiras do ranking

    otimizadores = {}
    for nome in ['pso', 'epso', 'abc']:
        res = OPTIMIZERS[nome](func, lower, upper, 50, 500, np.random.default_rng(42))
        (h, w), mask = mol._decode(res.best_x)
        snaps = []
        for X in res.snapshots:
            snaps.append({'hw': r(X[:, :2], 3), 'f': r([func(x) for x in X], 4)})
        otimizadores[nome] = {
            'snapshots': snaps,
            'historico': r(res.history, 4),
            'forma': [h, w],
            'mascara': mask.astype(int).tolist(),
            'f': float(res.best_f),
            'n_avaliacoes': res.n_evals,
        }

    # Um exemplo real de decodificação: uma partícula inicial do PSO que precisa de reparo
    rng = np.random.default_rng(42)
    X0 = rng.uniform(lower, upper, size=(50, len(lower)))
    exemplo = None
    for x in X0:
        h0, w0 = int(np.clip(np.rint(x[0]), 1, M)), int(np.clip(np.rint(x[1]), 1, M))
        n_sel = int((x[2:] > 0.5).sum())
        if 6 <= h0 * w0 <= 20 and h0 > 1 and w0 > 1 and n_sel != h0 * w0:
            (h, w), mask = mol._decode(x)
            exemplo = {'x': r(x, 3), 'h_arred': h0, 'w_arred': w0, 'forma': [h, w],
                       'n_marcadas': n_sel, 'mascara': mask.astype(int).tolist()}
            break

    # Pares de imagens consecutivas (formato didático 4x4, 16 primeiras do ranking)
    m16 = np.zeros(M, dtype=bool)
    m16[:16] = True
    fronteira = int((y_ord == 0).sum())               # primeira posição maligna
    sequencia = list(range(fronteira - 3, fronteira + 3))

    def mse(i):
        return float(((Xo[i, :16] - Xo[i + 1, :16]) ** 2).mean())

    # Resultados da validação cruzada (do experimento completo)
    cv = {}
    with open(RESULTADOS / 'principal' / 'results.csv') as fcsv:
        for linha in csv.DictReader(fcsv):
            if linha['model'].startswith('MOL-euclidean-'):
                cv[linha['model'].split('-')[-1]] = {
                    'cv': float(linha['cv_balanced_accuracy_mean']),
                    'cv_std': float(linha['cv_balanced_accuracy_std']),
                    'teste': float(linha['test_balanced_accuracy'])}

    return {
        'otimizadores': otimizadores,
        'exemplo_decodificacao': exemplo,
        'D': r(D, 4),
        'f_por_k_menores': r(f_por_k, 4),
        'f_por_k_ranking': r(f_top_k, 4),
        'f_formas': {'5x6': float(f_mask(np.ones(M, bool))), '4x4': float(f_mask(m16)),
                     'otimo': float(D_ordenado[:2].mean())},
        'otimo_features': [int(k) for k in np.argsort(D)[:2]],     # posições no ranking
        'sequencia': {'z': r(Xo[sequencia], 3), 'y': y_ord[sequencia].tolist(),
                      'mse': [mse(i) for i in sequencia[:-1]], 'posicoes': sequencia},
        'cv': cv,
        'pacientes_ordenados': {'benigno': r(Z[benignos[0]][ordem], 3),
                                'maligno': r(Z[malignos[0]][ordem], 3),
                                'exemplo': r(Z[ex][ordem], 3)},
    }


def dados_learn(Z, y_tr, Z_te, y_te, ordem, ex, benignos):
    """
    Dados do estágio Learn: treina a CNN do artigo nas imagens 5x6 (30 features na ordem do Map)
    e guarda pesos, mapas de ativação de um paciente, curva de treino e previsões no teste.
    """
    import keras
    from sklearn.metrics import confusion_matrix

    from mol.cnn import build_cnn

    h, w = 5, 6
    k = h * w
    n_oculta = int(np.ceil((k + 1) / 2))
    imgs_tr = Z[:, ordem].reshape(-1, h, w, 1)
    imgs_te = Z_te[:, ordem].reshape(-1, h, w, 1)

    keras.utils.set_random_seed(42)
    modelo = build_cnn((h, w, 1), n_oculta, 2, 1e-3)
    hist = modelo.fit(imgs_tr, y_tr, epochs=100, batch_size=32, verbose=0).history

    conv, pool, flat, densa, saida = modelo.layers
    W, bias = conv.get_weights()                       # W: (3, 3, 1, 32)
    W1, b1 = densa.get_weights()
    w2, b2 = saida.get_weights()

    def passo_a_passo(img):
        """Refaz a rede à mão para um paciente e confere com o Keras."""
        x = img.reshape(h, w)
        xp = np.pad(x, 1)                              # padding 'same': borda de zeros
        pre = np.zeros((h, w, 32))
        for r in range(h):
            for c in range(w):
                pre[r, c] = (xp[r:r + 3, c:c + 3, None] * W[:, :, 0, :]).sum(axis=(0, 1)) + bias
        relu = np.maximum(pre, 0)
        # max-pooling 2x2 'same': 5x6 -> 3x3 (a última linha fica sozinha)
        pooled = np.zeros((3, 3, 32))
        for r in range(3):
            for c in range(3):
                pooled[r, c] = relu[2 * r:2 * r + 2, 2 * c:2 * c + 2].max(axis=(0, 1))
        oculta = np.maximum(pooled.reshape(-1) @ W1 + b1, 0)
        logit = float(oculta @ w2[:, 0] + b2[0])
        prob = 1 / (1 + np.exp(-logit))
        p_keras = float(modelo.predict(img.reshape(1, h, w, 1), verbose=0)[0, 0])
        assert abs(prob - p_keras) < 1e-4, (prob, p_keras)
        return x, pre, relu, pooled, oculta, logit, prob

    x, pre, relu, pooled, oculta, logit, prob = passo_a_passo(imgs_tr[ex])
    # Filtro para o exemplo: o que mais responde a este paciente (maior ativação após o ReLU)
    f = int(np.argmax(relu.max(axis=(0, 1))))
    # Posição do exemplo de conta: a de maior ativação desse filtro
    r0, c0 = [int(v) for v in np.unravel_index(np.argmax(relu[:, :, f]), (h, w))]
    outros = [int(i) for i in np.argsort(-relu.max(axis=(0, 1)))[:6]]

    xb, _, _, _, _, logit_b, prob_b = passo_a_passo(imgs_tr[benignos[0]])

    p_te = modelo.predict(imgs_te, verbose=0)[:, 0]
    pred = (p_te > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()

    return {
        'forma': [h, w], 'n_oculta': n_oculta,
        'parametros': {'conv': int(W.size + bias.size), 'densa': int(W1.size + b1.size),
                       'saida': int(w2.size + b2.size)},
        'paciente': {'imagem': r(x, 3), 'y': int(y_tr[ex]), 'logit': logit, 'prob': float(prob),
                     'oculta': r(oculta, 3)},
        'benigno': {'imagem': r(xb, 3), 'logit': logit_b, 'prob': float(prob_b)},
        'filtro': {'indice': f, 'pesos': r(W[:, :, 0, f], 3), 'bias': float(bias[f]),
                   'pre': r(pre[:, :, f], 3), 'relu': r(relu[:, :, f], 3), 'pool': r(pooled[:, :, f], 3),
                   'posicao_exemplo': [r0, c0]},
        'mapas_relu': {str(i): r(relu[:, :, i], 3) for i in outros},
        'historico_loss': r(hist['loss'], 4),
        'teste': {'prob': r(p_te, 4), 'y': y_te.tolist(), 'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
    }


def dados_avaliacao():
    """Resultados do experimento principal (results/) e um exemplo passo a passo do Wilcoxon."""
    import pandas as pd
    from scipy.stats import rankdata

    from mol.utils import wilcoxon_test

    res = pd.read_csv(RESULTADOS / 'principal' / 'results.csv')
    folds = pd.read_csv(RESULTADOS / 'principal' / 'fold_scores.csv', index_col='fold')

    modelos = []
    for _, l in res.sort_values('cv_balanced_accuracy_mean', ascending=False).iterrows():
        modelos.append({
            'nome': l['model'], 'familia': l['family'], 'fs': l['fs_strategy'],
            'cv_media': float(l['cv_balanced_accuracy_mean']), 'cv_desvio': float(l['cv_balanced_accuracy_std']),
            'teste': float(l['test_balanced_accuracy']), 'sensibilidade': float(l['test_sensitivity']),
            'especificidade': float(l['test_specificity']), 'n_features': int(l['n_features']),
            'tp': int(l['test_tp']), 'fp': int(l['test_fp']), 'tn': int(l['test_tn']), 'fn': int(l['test_fn']),
            'melhor_lr': json.loads(l['best_params']).get('learning_rate'),
        })

    melhor_mol = res[res.family == 'MOL'].sort_values('cv_balanced_accuracy_mean', ascending=False).iloc[0]['model']
    melhor_base = res[res.family != 'MOL'].sort_values('cv_balanced_accuracy_mean', ascending=False).iloc[0]['model']

    # Wilcoxon passo a passo: melhor MOL (A) contra o melhor baseline (B)
    a, b = folds[melhor_mol].values, folds[melhor_base].values
    d = np.round(a - b, 10)                                 # arredonda para preservar empates
    postos = rankdata(np.abs(d), method='average')          # zeros entram no ranking (método zsplit)
    w_mais = float(postos[d > 0].sum() + postos[d == 0].sum() / 2)
    w_menos = float(postos[d < 0].sum() + postos[d == 0].sum() / 2)

    comparacoes = []
    for outro in [melhor_base, 'LR-S', 'KNN-S', 'RF-S', 'XGB-S', 'LGBM-S', 'MOL-euclidean-pso', 'MOL-euclidean-epso']:
        comparacoes.append({'nome': outro, 'cv_media': float(folds[outro].mean()),
                            'diferenca': float(a.mean() - folds[outro].mean()),
                            'p': wilcoxon_test(a, folds[outro].values)})

    return {
        'modelos': modelos,
        'melhor_mol': melhor_mol, 'melhor_baseline': melhor_base,
        'folds_melhor_mol': r(a, 4),
        'wilcoxon_exemplo': {'A': melhor_mol, 'B': melhor_base, 'a': r(a, 4), 'b': r(b, 4), 'd': r(d, 4),
                             'postos': r(postos, 1), 'w_mais': w_mais, 'w_menos': w_menos,
                             'p': wilcoxon_test(a, b)},
        'comparacoes': comparacoes,
    }


def dados_critica(D, otimizadores, cv_otim, X_tr, y_tr):
    """Dados da análise crítica: testes de controle e MOL v2."""
    import ast

    import pandas as pd

    from mol.v2 import MOLv2Classifier

    D = np.asarray(D)
    ctrl = pd.read_csv(RESULTADOS / 'controles' / 'resultados.csv')
    comp_c = pd.read_csv(RESULTADOS / 'controles' / 'comparacoes.csv')
    resumo_c = pd.read_csv(RESULTADOS / 'controles' / 'resumo.csv').set_index('nome')

    # Teste 1: para cada solução (aleatória ou otimizada), nº de features, acurácia e o f da Eq. 1
    aleatorias = []
    for _, l in ctrl[ctrl.teste == 1].iterrows():
        pos = ast.literal_eval(l['posicoes'])
        aleatorias.append({'n': len(pos), 'cv': float(l['cv_media']), 'f': float(D[pos].mean()), 'forma': l['forma']})
    otimizadas = {}
    for nome, d in otimizadores.items():
        pos = [k for k, b in enumerate(d['mascara']) if b]
        otimizadas[nome] = {'n': len(pos), 'cv': cv_otim[nome]['cv'], 'f': float(d['f'])}
    todos_f = [a['f'] for a in aleatorias] + [o['f'] for o in otimizadas.values()]
    todos_cv = [a['cv'] for a in aleatorias] + [o['cv'] for o in otimizadas.values()]

    def media(nome):
        return float(resumo_c.loc[nome, 'cv_media'])

    def p_c(a, b):
        l = comp_c[(comp_c.A == a) & (comp_c.B == b)].iloc[0]
        return float(l['p_valor'])

    # MOL v2
    resumo_v = pd.read_csv(RESULTADOS / 'mol_v2' / 'resumo.csv').set_index('grupo')
    comp_v = pd.read_csv(RESULTADOS / 'mol_v2' / 'comparacoes.csv')

    def p_v(a, b):
        l = comp_v[(comp_v.A == a) & (comp_v.B == b)].iloc[0]
        return float(l['p_valor'])

    arranjos = {}
    for arr in ['correlacao', 'aleatorio']:
        m = MOLv2Classifier(arranjo=arr, semente_arranjo=100).ajustar_arranjo(X_tr, y_tr)
        arranjos[arr] = {'colunas': [int(c) for c in m.colunas_], 'correlacao_vizinhos': m.correlacao_vizinhos_}

    return {
        'teste1': {'aleatorias': aleatorias, 'otimizadas': otimizadas,
                   'media_aleatorias': media('média das aleatórias'),
                   'p': {n: p_c(f'{n.upper()} (otimizado)', 'média das aleatórias') for n in ['pso', 'epso', 'abc']},
                   'correlacao_f_cv': float(np.corrcoef(todos_f, todos_cv)[0, 1])},
        'teste2': {'barras': [['4×4', media('top16-4x4')], ['2×8', media('top16-2x8')], ['16×1', media('top16-16x1')],
                              ['1×16', media('top16-1x16')],
                              ['4×4 embaralhado', media('top16-4x4 embaralhado (média)')]],
                   'p_formato': p_c('top16-4x4', 'top16-16x1'),
                   'p_ordem': p_c('top16-4x4', 'top16-4x4 embaralhado (média)')},
        'teste3': {'pares': [['16 features', media('top16-4x4'), media('top16-vetor-MLP'), p_c('top16-4x4', 'top16-vetor-MLP')],
                             ['30 features', media('todas30-5x6'), media('todas30-vetor-MLP'),
                              p_c('todas30-5x6', 'todas30-vetor-MLP')]],
                   'lr': media('Regressão logística')},
        'v2': {'barras': [['MOL v2: arranjo por correlação', float(resumo_v.loc['30-v2-correlacao', 'cv_media']), 'v2'],
                          ['pixels em posição aleatória', float(resumo_v.loc['30-aleatorio', 'cv_media']), 'controle'],
                          ['pior arranjo possível', float(resumo_v.loc['30-pior', 'cv_media']), 'controle'],
                          ['rede comum, sem imagem', float(resumo_v.loc['30-vetor-MLP', 'cv_media']), 'vetor'],
                          ['regressão logística', float(resumo_v.loc['Regressão logística', 'cv_media']), 'baseline'],
                          ['MOL original (ABC)', float(resumo_v.loc['MOL original (ABC)', 'cv_media']), 'mol']],
               'p_aleatorio': p_v('30-v2-correlacao', '30-aleatorio'), 'p_pior': p_v('30-v2-correlacao', '30-pior'),
               'p_lr': p_v('30-v2-correlacao', 'Regressão logística'),
               'p_mol': p_v('30-v2-correlacao', 'MOL original (ABC)'),
               'arranjos': arranjos},
    }


def main():
    X, y, nomes = load_breast_cancer_data()
    X_tr, X_te, y_tr, y_te = split_data(X, y, test_size=0.2, random_state=42)

    # Padronização ajustada só no treino (igual ao pipeline)
    mu, sigma = X_tr.mean(axis=0), X_tr.std(axis=0)
    Z = (X_tr - mu) / sigma

    # Pontuações de cada critério do Map
    scores = {}
    for c in CRITERIOS:
        mol = MapOptimizeLearn(map_strategy=c)
        mol.n_features_in_ = X.shape[1]
        scores[c] = mol._feature_importance(Z, y_tr)
    ordem_euclid = np.argsort(-scores['euclidean'], kind='stable')

    # Features usadas nos exemplos de cálculo
    f_top = nomes.index('concave points_worst')    # feature 28
    f_low = nomes.index('fractal_dimension_mean')  # feature 10

    def estat_classe(j):
        out = {}
        for c in [0, 1]:
            zc = Z[y_tr == c, j]
            out[str(c)] = {'n': int(len(zc)), 'media': float(zc.mean()), 'var': float(zc.var())}
        return out

    # Detalhes do Gain Ratio para a feature top: 10 faixas por quantil
    disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile',
                            quantile_method='averaged_inverted_cdf')
    B = disc.fit_transform(Z[:, [f_top]])[:, 0].astype(int)
    faixas = []
    for b in range(10):
        m = B == b
        faixas.append({'benignos': int(((y_tr == 0) & m).sum()), 'malignos': int(((y_tr == 1) & m).sum()),
                       'limite_sup': float(disc.bin_edges_[0][b + 1])})
    h_y = float(entropy(np.bincount(y_tr)))
    h_y_dado_b = float(sum((f['benignos'] + f['malignos']) / len(y_tr) *
                           entropy([f['benignos'], f['malignos']]) for f in faixas))
    split_info = float(entropy([f['benignos'] + f['malignos'] for f in faixas]))

    # Uma divisão de árvore na feature top: o limiar que mais reduz o Gini
    def gini(nb, nm):
        n = nb + nm
        return 1 - (nb / n) ** 2 - (nm / n) ** 2 if n else 0.0
    col, n_tot = Z[:, f_top], len(y_tr)
    nb_tot, nm_tot = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    melhor = None
    valores = np.sort(np.unique(col))
    for lim in (valores[:-1] + valores[1:]) / 2:
        esq = col <= lim
        nb_e, nm_e = int(((y_tr == 0) & esq).sum()), int(((y_tr == 1) & esq).sum())
        nb_d, nm_d = nb_tot - nb_e, nm_tot - nm_e
        filhos = ((nb_e + nm_e) * gini(nb_e, nm_e) + (nb_d + nm_d) * gini(nb_d, nm_d)) / n_tot
        if melhor is None or filhos < melhor['gini_filhos']:
            melhor = {'limiar': float(lim), 'esq': [nb_e, nm_e], 'dir': [nb_d, nm_d],
                      'gini_pai': gini(nb_tot, nm_tot), 'gini_esq': gini(nb_e, nm_e),
                      'gini_dir': gini(nb_d, nm_d), 'gini_filhos': float(filhos)}
    melhor['delta'] = melhor['gini_pai'] - melhor['gini_filhos']

    # Pacientes de exemplo (do treino)
    ex = int(pacientes_tipicos(Z, y_tr, 1, 1)[0])
    benignos = pacientes_tipicos(Z, y_tr, 0, 6)
    malignos = pacientes_tipicos(Z, y_tr, 1, 6)

    # 8 linhas da tabela original para exibir: as 4 primeiras malignas e as 4 primeiras benignas
    linhas_tab = np.sort(np.concatenate([np.where(y == 1)[0][:4], np.where(y == 0)[0][:4]]))

    opt = dados_optimize(Z, y_tr, ordem_euclid, ex, benignos, malignos)

    dados = {
        'feature_names': nomes,
        'contagens': {
            'total': [int((y == 0).sum()), int((y == 1).sum())],
            'treino': [int((y_tr == 0).sum()), int((y_tr == 1).sum())],
            'teste': [int((y_te == 0).sum()), int((y_te == 1).sum())],
        },
        # primeiras linhas da tabela original (valores brutos) + diagnóstico
        'tabela': {'linhas': linhas_tab.tolist(), 'X': r(X[linhas_tab], 4), 'y': y[linhas_tab].tolist()},
        'mu': r(mu, 5),
        'sigma': r(sigma, 5),
        'paciente': {'x': r(X_tr[ex], 5), 'z': r(Z[ex], 3), 'y': int(y_tr[ex])},
        'imagens': {'benignos': r(Z[benignos], 3), 'malignos': r(Z[malignos], 3)},
        # colunas completas (z) das duas features do exemplo, com os rótulos
        'colunas': {'y': y_tr.tolist(), 'top': r(Z[:, f_top], 3), 'low': r(Z[:, f_low], 3)},
        'f_top': f_top,
        'f_low': f_low,
        'estat_top': estat_classe(f_top),
        'estat_low': estat_classe(f_low),
        'scores': {c: r(v, 4) for c, v in scores.items()},
        'top3': {c: [int(k) for k in np.argsort(-v, kind='stable')[:3]] for c, v in scores.items()},
        'ordem': ordem_euclid.tolist(),
        'gini_split': melhor,
        'gain_ratio': {'faixas': faixas, 'H_Y': h_y, 'H_Y_dado_B': h_y_dado_b,
                       'IG': h_y - h_y_dado_b, 'split_info': split_info},
        'optimize': opt,
        'critica': dados_critica(opt['D'], opt['otimizadores'], opt['cv'], X_tr, y_tr),
        'avaliacao': dados_avaliacao(),
        'learn': dados_learn(Z, y_tr, (X_te - mu) / sigma, y_te, ordem_euclid, ex, benignos),
    }
    (SAIDA / 'mol_dados.json').write_text(json.dumps(dados, indent=1, ensure_ascii=False))
    print(f"Dados salvos em {SAIDA / 'mol_dados.json'}")
    print(f"  paciente exemplo (treino #{ex}), GR: IG={h_y - h_y_dado_b:.4f}, split={split_info:.4f}")


if __name__ == '__main__':
    main()
