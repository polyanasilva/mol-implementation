"""
Resume o experimento do MOL v2: resumo.csv, comparacoes.csv, grafico.png e arranjo.png
(onde cada feature ficou na imagem) em results/mol_v2/.

    python experimentos/analisar_mol_v2.py
"""
import contextlib
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mol.data import prepare_data_pipeline
from mol.utils import wilcoxon_test
from mol.v2 import MOLv2Classifier

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / 'results' / 'mol_v2'

AZUL, LARANJA, CINZA, AMARELO, VERDE, VERMELHO = '#4C78A8', '#F58518', '#9A9A9A', '#E4B400', '#54A24B', '#D64545'

ROTULOS = {
    '30-v2-correlacao': 'MOL v2 · arranjo por correlação (5×6)',
    '30-ranking': 'ordem do Map original (5×6)',
    '30-aleatorio': 'pixels em posição aleatória (5×6)',
    '30-pior': 'pior arranjo possível (5×6)',
    '30-vetor-MLP': 'rede comum, sem imagem (30 features)',
    '16-v2-correlacao': 'MOL v2 · 16 features mRMR (4×4)',
    '16-aleatorio': '16 features mRMR, posição aleatória (4×4)',
    '16-vetor-MLP': 'rede comum, sem imagem (16 features mRMR)',
}
CORES = {'30-v2-correlacao': VERDE, '30-ranking': AZUL, '30-aleatorio': CINZA, '30-pior': VERMELHO,
         '30-vetor-MLP': AMARELO, '16-v2-correlacao': VERDE, '16-aleatorio': CINZA, '16-vetor-MLP': AMARELO}
BASELINES = {'Regressão logística': 'LR-S', 'ANN tabular (sklearn)': 'ANN-S', 'XGBoost': 'XGB-S',
             'MOL original (ABC)': 'MOL-euclidean-abc'}

# Tipo de medida de cada feature, para colorir o mapa do arranjo
TIPOS = {'radius': 'tamanho', 'perimeter': 'tamanho', 'area': 'tamanho', 'texture': 'textura',
         'smoothness': 'suavidade', 'compactness': 'forma', 'concavity': 'forma', 'concave points': 'forma',
         'symmetry': 'simetria', 'fractal_dimension': 'dim. fractal'}
COR_TIPO = {'tamanho': '#F58518', 'textura': '#54A24B', 'suavidade': '#4C78A8', 'forma': '#B279A2',
            'simetria': '#E4B400', 'dim. fractal': '#9A9A9A'}


def main():
    res = pd.read_csv(PASTA / 'resultados.csv')
    folds = pd.read_csv(PASTA / 'fold_scores.csv', index_col='fold')
    principal = pd.read_csv(RAIZ / 'results' / 'principal' / 'fold_scores.csv', index_col='fold')

    # Média de cada grupo, fold a fold (agrega sementes de rede e permutações)
    grupo_folds = {g: folds[list(d.nome)].mean(axis=1).values for g, d in res.groupby('grupo')}
    for nome, col in BASELINES.items():
        grupo_folds[nome] = principal[col].values

    linhas = []
    for g, d in res.groupby('grupo'):
        linhas.append({'grupo': g, 'descricao': ROTULOS[g], 'n_execucoes': len(d),
                       'cv_media': grupo_folds[g].mean(), 'cv_desvio_folds': grupo_folds[g].std(),
                       'cv_min_execucao': d.cv_media.min(), 'cv_max_execucao': d.cv_media.max(),
                       'teste_bal_acc_media': d.teste_bal_acc.mean(),
                       'teste_sensibilidade_media': d.teste_sensibilidade.mean(),
                       'correlacao_vizinhos': d.correlacao_vizinhos.mean(), 'erro_arranjo': d.erro_arranjo.mean()})
    for nome in BASELINES:
        linhas.append({'grupo': nome, 'descricao': nome + ' (experimento principal)', 'n_execucoes': 1,
                       'cv_media': grupo_folds[nome].mean(), 'cv_desvio_folds': grupo_folds[nome].std()})
    resumo = pd.DataFrame(linhas).sort_values('cv_media', ascending=False)
    resumo.to_csv(PASTA / 'resumo.csv', index=False)

    comparacoes = []
    for a, b, pergunta in [
        ('30-v2-correlacao', '30-aleatorio', 'arranjo otimizado é melhor que pixels aleatórios?'),
        ('30-v2-correlacao', '30-pior', 'arranjo otimizado é melhor que o pior arranjo?'),
        ('30-v2-correlacao', '30-ranking', 'arranjo otimizado é melhor que a ordem do Map original?'),
        ('30-v2-correlacao', '30-vetor-MLP', 'a imagem é melhor que o vetor numa rede comum?'),
        ('30-v2-correlacao', 'Regressão logística', 'MOL v2 é melhor que a regressão logística?'),
        ('30-v2-correlacao', 'ANN tabular (sklearn)', 'MOL v2 é melhor que a ANN tabular?'),
        ('30-v2-correlacao', 'XGBoost', 'MOL v2 é melhor que o XGBoost?'),
        ('30-v2-correlacao', 'MOL original (ABC)', 'MOL v2 é melhor que o MOL original?'),
        ('16-v2-correlacao', '16-aleatorio', '(16 features) arranjo otimizado é melhor que aleatório?'),
        ('16-v2-correlacao', '16-vetor-MLP', '(16 features) a imagem é melhor que o vetor?'),
        ('30-v2-correlacao', '16-v2-correlacao', 'usar as 30 é melhor que selecionar 16 por mRMR?'),
    ]:
        fa, fb = grupo_folds[a], grupo_folds[b]
        comparacoes.append({'pergunta': pergunta, 'A': a, 'B': b, 'media_A': fa.mean(), 'media_B': fb.mean(),
                            'diferenca': fa.mean() - fb.mean(), 'p_valor': wilcoxon_test(fa, fb)})
    comp = pd.DataFrame(comparacoes)
    comp['significativo_0.05'] = comp.p_valor < 0.05
    comp.to_csv(PASTA / 'comparacoes.csv', index=False)

    # Gráfico de resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1.5, 1]})
    ordem = ['30-v2-correlacao', '30-ranking', '30-aleatorio', '30-pior', '30-vetor-MLP',
             '16-v2-correlacao', '16-aleatorio', '16-vetor-MLP'] + list(BASELINES)
    y = np.arange(len(ordem))[::-1]
    medias = [grupo_folds[g].mean() for g in ordem]
    desvios = [grupo_folds[g].std() for g in ordem]
    cores = [CORES.get(g, LARANJA) for g in ordem]
    ax1.barh(y, medias, xerr=desvios, color=cores, capsize=3)
    for yi, m in zip(y, medias):
        ax1.text(0.882, yi, f'{m:.3f}', va='center', color='white', fontweight='bold')
    ax1.set_yticks(y)
    ax1.set_yticklabels([ROTULOS.get(g, g + ' (baseline)') for g in ordem])
    ax1.set_xlim(0.88, 1.0)
    ax1.set_xlabel('acurácia balanceada (CV de 10 folds; barra de erro = desvio entre folds)')
    ax1.set_title('MOL v2 contra os controles e os baselines')
    ax1.grid(axis='x', alpha=0.3)

    cnn30 = res[(res.n_features == 30) & (res.modelo == 'cnn')]
    for arr, cor, rot in [('correlacao', VERDE, 'otimizado'), ('ranking', AZUL, 'ordem do Map'),
                          ('aleatorio', CINZA, 'aleatório'), ('pior', VERMELHO, 'pior')]:
        d = cnn30[cnn30.arranjo == arr]
        ax2.scatter(d.correlacao_vizinhos, d.cv_media, s=90, color=cor, label=rot, zorder=3)
    r = np.corrcoef(cnn30.correlacao_vizinhos, cnn30.cv_media)[0, 1]
    ax2.set_xlabel('|correlação| média entre pixels vizinhos (qualidade do arranjo)')
    ax2.set_ylabel('acurácia balanceada (CV)')
    ax2.set_title(f'Arranjo melhor dá acurácia melhor? (r = {r:.2f}, CNN 5×6)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PASTA / 'grafico.png', dpi=160)

    # Mapa do arranjo
    with contextlib.redirect_stdout(io.StringIO()):
        d = prepare_data_pipeline()
    nomes = d['feature_names']
    fig, eixos = plt.subplots(1, 3, figsize=(19, 6.4))
    for ax, arr, titulo in zip(eixos, ['correlacao', 'ranking', 'aleatorio'],
                               ['MOL v2: arranjo por correlação', 'Ordem do Map original (linha a linha)',
                                'Posição aleatória (controle)']):
        m = MOLv2Classifier(arranjo=arr, semente_arranjo=100).ajustar_arranjo(d['X_train'], d['y_train'])
        h, w = m.forma_
        for p, col in enumerate(m.colunas_):
            r_, c_ = divmod(p, w)
            base = nomes[col].rsplit('_', 1)[0]
            ax.add_patch(plt.Rectangle((c_, h - 1 - r_), 1, 1, color=COR_TIPO[TIPOS[base]], ec='white', lw=2))
            ax.text(c_ + 0.5, h - 0.5 - r_, nomes[col].replace('_', '\n'), ha='center', va='center', fontsize=8.5)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{titulo}\n|correlação| entre vizinhos = {m.correlacao_vizinhos_:.2f}', fontsize=12)
    legenda = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COR_TIPO.values()]
    fig.legend(legenda, list(COR_TIPO), loc='lower center', ncol=6, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(PASTA / 'arranjo.png', dpi=160)

    pd.set_option('display.width', 250, 'display.max_colwidth', 70, 'display.float_format', '{:.4f}'.format)
    print(resumo[['descricao', 'n_execucoes', 'cv_media', 'cv_desvio_folds', 'cv_min_execucao', 'cv_max_execucao',
                  'teste_bal_acc_media', 'teste_sensibilidade_media', 'correlacao_vizinhos']].to_string(index=False))
    print()
    print(comp.to_string(index=False))
    print(f"\ncorrelação entre qualidade do arranjo e acurácia (CNN 5x6, {len(cnn30)} execuções): r = {r:.3f}")


if __name__ == '__main__':
    main()
