"""
Configuração, métricas, teste de Wilcoxon e gráficos.
Rótulos: 1 = maligno (classe positiva), 0 = benigno.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def load_config(config_path='config.yaml'):
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_grid(grid):
    """
    Converte o grid do YAML para o formato do scikit-learn:
      {logspace: [a, b, n]} -> np.logspace(a, b, n)
      hidden_layer_sizes: [5, 10] -> [(5,), (10,)]
    """
    out = {}
    for key, values in (grid or {}).items():
        if isinstance(values, dict) and 'logspace' in values:
            a, b, n = values['logspace']
            values = list(np.logspace(a, b, int(n)))
        if key == 'hidden_layer_sizes':
            values = [(v,) if isinstance(v, int) else tuple(v) for v in values]
        out[key] = values
    return out


def create_directories(results_dir='results'):
    for d in [results_dir, f'{results_dir}/images_with_pathology',
              f'{results_dir}/images_without_pathology']:
        Path(d).mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Métricas para a classe positiva = maligno (1).
    y_proba: probabilidade da classe 1 (necessária para o ROC-AUC).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }
    return metrics


def wilcoxon_test(scores_a, scores_b):
    """Wilcoxon signed-rank pareado (mesmos folds). Retorna p-valor."""
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    if np.allclose(a, b):
        return 1.0
    # sem arredondar, o ruído de ponto flutuante desfaz empates reais e muda os postos
    d = np.round(a - b, 10)
    return float(wilcoxon(d, zero_method='zsplit').pvalue)


def visualize_mol_images(images, labels, num_samples=10, save_dir='results', scale=40):
    """
    Salva imagens do MOL: maligno (1) em images_with_pathology/,
    benigno (0) em images_without_pathology/.

    A escala de cinza é a MESMA para todas as imagens (percentis 1-99 do conjunto),
    para que a intensidade de um pixel seja comparável entre pacientes.
    """
    from PIL import Image

    vmin, vmax = np.percentile(images, [1, 99])
    images = np.clip(images, vmin, vmax)
    for label, folder in [(1, 'images_with_pathology'), (0, 'images_without_pathology')]:
        out = Path(save_dir) / folder
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob('*.png'):
            old.unlink()
        for i, idx in enumerate(np.where(labels == label)[0][:num_samples]):
            arr = ((images[idx] - vmin) / (vmax - vmin + 1e-12) * 255).astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
            img = img.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
            img.save(out / f'sample_{i:03d}.png')
    print(f"Imagens do MOL salvas em {save_dir}/images_with_pathology (maligno) "
          f"e images_without_pathology (benigno)")


def plot_mol_grid(images, labels, save_path, n_per_class=8):
    """Grade lado a lado: benignos em cima, malignos embaixo (mesma escala)."""
    vmin, vmax = np.percentile(images, [1, 99])
    fig, axes = plt.subplots(2, n_per_class, figsize=(1.6 * n_per_class, 3.6))
    for row, (label, name) in enumerate([(0, 'Benigno'), (1, 'Maligno')]):
        idx = np.where(labels == label)[0][:n_per_class]
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([])
            if col < len(idx):
                ax.imshow(images[idx[col]], cmap='gray', vmin=vmin, vmax=vmax)
            if col == 0:
                ax.set_ylabel(name)
    fig.suptitle(f'Imagens MOL {images.shape[1]}x{images.shape[2]}')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_comparison(results_df, save_path):
    """Acurácia balanceada: média ± desvio na CV (treino) e no teste."""
    df = results_df.sort_values('cv_balanced_accuracy_mean', ascending=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(df) + 1.5))
    ax.barh(y - 0.2, df['cv_balanced_accuracy_mean'], 0.4,
            xerr=df['cv_balanced_accuracy_std'], label='CV (treino)', color='#4C78A8')
    ax.barh(y + 0.2, df['test_balanced_accuracy'], 0.4, label='Teste', color='#F58518')
    ax.set_yticks(y)
    ax.set_yticklabels(df['model'])
    lo = max(0.0, min(df['cv_balanced_accuracy_mean'].min(), df['test_balanced_accuracy'].min()) - 0.05)
    ax.set_xlim(lo, 1.0)
    ax.set_xlabel('Acurácia balanceada')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def print_summary(results_df, wilcoxon_df=None):
    cols = ['model', 'cv_balanced_accuracy_mean', 'cv_balanced_accuracy_std',
            'test_balanced_accuracy', 'test_sensitivity', 'test_specificity',
            'test_roc_auc', 'n_features']
    df = results_df.sort_values('cv_balanced_accuracy_mean', ascending=False)[cols]
    print("\nResumo, ordenado pela acurácia balanceada média na validação cruzada:")
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 200):
        print(df.to_string(index=False))
    if wilcoxon_df is not None and len(wilcoxon_df):
        print("\nWilcoxon (melhor MOL vs. demais, pareado nos folds):")
        with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 200):
            print(wilcoxon_df.to_string(index=False))

