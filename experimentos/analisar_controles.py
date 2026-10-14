"""
Resume os testes de controle: junta os resultados com os folds do experimento principal,
aplica o Wilcoxon pareado e gera resumo.csv, comparacoes.csv e grafico.png em
results/controles/.

    python experimentos/analisar_controles.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mol.utils import wilcoxon_test

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / 'results' / 'controles'

AZUL, LARANJA, CINZA, AMARELO, VERDE = '#4C78A8', '#F58518', '#9A9A9A', '#E4B400', '#54A24B'


def main():
    res = pd.read_csv(PASTA / 'resultados.csv')
    folds = pd.read_csv(PASTA / 'fold_scores.csv', index_col='fold')
    principal = pd.read_csv(RAIZ / 'results' / 'principal' / 'fold_scores.csv', index_col='fold')

    # Referências do experimento principal (mesmo split e mesmos folds)
    refs = {'PSO (otimizado)': 'MOL-euclidean-pso', 'EPSO (otimizado)': 'MOL-euclidean-epso',
            'ABC (otimizado)': 'MOL-euclidean-abc', 'Regressão logística': 'LR-S', 'ANN (tabular)': 'ANN-S'}
    ref_folds = {nome: principal[col].values for nome, col in refs.items()}

    def f(nome):
        return folds[nome].values if nome in folds else ref_folds[nome]

    comparacoes = []

    def comparar(teste, a, b, pergunta):
        fa, fb = f(a), f(b)
        comparacoes.append({'teste': teste, 'pergunta': pergunta, 'A': a, 'B': b,
                            'media_A': fa.mean(), 'media_B': fb.mean(), 'diferenca_A_menos_B': fa.mean() - fb.mean(),
                            'p_valor': wilcoxon_test(fa, fb)})

    # Teste 1
    aleat = res[res.teste == 1].sort_values('cv_media', ascending=False)
    media_aleat = folds[aleat.nome].mean(axis=1).values          # média das aleatórias, fold a fold
    ref_folds['média das aleatórias'] = media_aleat
    for nome in ['PSO (otimizado)', 'EPSO (otimizado)', 'ABC (otimizado)']:
        comparar(1, nome, 'média das aleatórias', 'otimizar é melhor que sortear?')

    # Teste 2
    emb_top = [n for n in res.nome if n.startswith('top16-4x4-embaralhado')]
    emb_abc = [n for n in res.nome if n.startswith('abc16-4x4-embaralhado')]
    ref_folds['top16-4x4 embaralhado (média)'] = folds[emb_top].mean(axis=1).values
    ref_folds['abc16-4x4 embaralhado (média)'] = folds[emb_abc].mean(axis=1).values
    comparar(2, 'top16-4x4', 'top16-16x1', 'o formato 2D é melhor que o vetor?')
    comparar(2, 'top16-4x4', 'top16-2x8', 'o formato quadrado é melhor que 2x8?')
    comparar(2, 'top16-4x4', 'top16-4x4 embaralhado (média)', 'a ordem do Map nos pixels importa?')
    comparar(2, 'abc16-4x4', 'abc16-16x1', 'o formato 2D é melhor que o vetor? (máscara do ABC)')
    comparar(2, 'abc16-4x4', 'abc16-4x4 embaralhado (média)', 'a ordem do Map nos pixels importa? (máscara do ABC)')

    # Teste 3
    comparar(3, 'top16-16x1', 'top16-vetor-MLP', 'a convolução ajuda? (16 primeiras)')
    comparar(3, 'top16-4x4', 'top16-vetor-MLP', 'a imagem 4x4 é melhor que o vetor numa rede comum?')
    comparar(3, 'abc16-16x1', 'abc16-vetor-MLP', 'a convolução ajuda? (máscara do ABC)')
    comparar(3, 'todas30-5x6', 'todas30-vetor-MLP', 'a imagem 5x6 é melhor que o vetor? (30 features)')
    comparar(3, 'ABC (otimizado)', 'todas30-vetor-MLP', 'o melhor MOL é melhor que uma rede comum com tudo?')
    comparar(3, 'ABC (otimizado)', 'Regressão logística', 'o melhor MOL é melhor que a regressão logística?')

    comp = pd.DataFrame(comparacoes)
    comp['significativo_0.05'] = comp.p_valor < 0.05
    comp.to_csv(PASTA / 'comparacoes.csv', index=False)

    resumo = res.drop(columns=['posicoes']).copy()
    for nome, fs in ref_folds.items():
        resumo.loc[len(resumo)] = {'nome': nome, 'teste': 0, 'modelo': 'referência', 'forma': '', 'n_features': np.nan,
                                   'cv_media': fs.mean(), 'cv_desvio': fs.std(), 'teste_bal_acc': np.nan,
                                   'melhor_lr': np.nan, 'tempo_s': np.nan}
    resumo.to_csv(PASTA / 'resumo.csv', index=False)

    # Gráfico
    fig, eixos = plt.subplots(1, 3, figsize=(17, 6.2))

    # Teste 1: aleatórias (pontos) vs otimizados (linhas)
    ax = eixos[0]
    ax.scatter(aleat.n_features, aleat.cv_media, s=70, color=CINZA, label=f'{len(aleat)} soluções aleatórias', zorder=3)
    for nome, cor in [('PSO (otimizado)', AZUL), ('EPSO (otimizado)', LARANJA), ('ABC (otimizado)', VERDE)]:
        ax.axhline(ref_folds[nome].mean(), color=cor, lw=2, label=f'{nome}: {ref_folds[nome].mean():.3f}')
    ax.axhline(media_aleat.mean(), color='black', ls='--', lw=1.5, label=f'média das aleatórias: {media_aleat.mean():.3f}')
    ax.set_xlabel('número de features na imagem')
    ax.set_ylabel('acurácia balanceada (CV de 10 folds)')
    ax.set_title('Teste 1: otimizar é melhor que sortear?')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)

    def barras(ax, nomes, rotulos, cores, titulo):
        medias = [f(n).mean() for n in nomes]
        desvios = [f(n).std() for n in nomes]
        y = np.arange(len(nomes))[::-1]
        ax.barh(y, medias, xerr=desvios, color=cores, capsize=3)
        for yi, m in zip(y, medias):
            ax.text(0.862, yi, f'{m:.3f}', va='center', fontsize=10, color='white', fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(rotulos, fontsize=10)
        ax.set_xlim(0.86, 1.0)
        ax.set_xlabel('acurácia balanceada (CV de 10 folds)')
        ax.set_title(titulo)
        ax.grid(axis='x', alpha=0.3)

    barras(eixos[1],
           ['top16-4x4', 'top16-2x8', 'top16-16x1', 'top16-1x16', 'top16-4x4 embaralhado (média)',
            'abc16-4x4', 'abc16-16x1', 'abc16-4x4 embaralhado (média)'],
           ['16 primeiras · 4×4', '16 primeiras · 2×8', '16 primeiras · 16×1', '16 primeiras · 1×16',
            '16 primeiras · 4×4 embaralhado', 'máscara ABC · 4×4', 'máscara ABC · 16×1', 'máscara ABC · 4×4 embaralhado'],
           [AZUL] * 4 + [CINZA] + [VERDE] * 2 + [CINZA],
           'Teste 2: formato e ordem dos pixels importam?')

    barras(eixos[2],
           ['top16-4x4', 'top16-16x1', 'top16-vetor-MLP', 'todas30-5x6', 'todas30-30x1', 'todas30-vetor-MLP',
            'ABC (otimizado)', 'Regressão logística'],
           ['16 primeiras · CNN 4×4', '16 primeiras · CNN 16×1', '16 primeiras · rede comum (vetor)',
            '30 features · CNN 5×6', '30 features · CNN 30×1', '30 features · rede comum (vetor)',
            'Melhor MOL (ABC)', 'Regressão logística'],
           [AZUL, AZUL, AMARELO, AZUL, AZUL, AMARELO, VERDE, LARANJA],
           'Teste 3: virar imagem ajuda?')

    fig.tight_layout()
    fig.savefig(PASTA / 'grafico.png', dpi=160)

    # Saída no terminal
    pd.set_option('display.width', 250, 'display.max_colwidth', 60, 'display.float_format', '{:.4f}'.format)
    print('\n=== TESTE 1: soluções aleatórias ===')
    print(aleat[['nome', 'forma', 'n_features', 'cv_media', 'cv_desvio', 'teste_bal_acc']].to_string(index=False))
    print(f"\nmédia {aleat.cv_media.mean():.4f} | mín {aleat.cv_media.min():.4f} | máx {aleat.cv_media.max():.4f}")
    for nome in ['PSO (otimizado)', 'EPSO (otimizado)', 'ABC (otimizado)']:
        m = ref_folds[nome].mean()
        print(f"{nome}: {m:.4f}  -> {int((aleat.cv_media > m).sum())} de {len(aleat)} aleatórias são melhores")
    print(f"correlação (nº de features, acurácia) nas aleatórias: {np.corrcoef(aleat.n_features, aleat.cv_media)[0, 1]:.3f}")
    print('\n=== TESTES 2 e 3 ===')
    print(res[res.teste > 1][['nome', 'teste', 'forma', 'cv_media', 'cv_desvio', 'teste_bal_acc', 'melhor_lr']].to_string(index=False))
    print('\n=== COMPARAÇÕES (Wilcoxon pareado) ===')
    print(comp[['teste', 'pergunta', 'A', 'B', 'media_A', 'media_B', 'diferenca_A_menos_B', 'p_valor', 'significativo_0.05']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
