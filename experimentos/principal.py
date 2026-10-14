"""
Experimento principal: MOL contra os baselines no Breast Cancer Wisconsin, com o
protocolo do artigo.

Grid search com 10 folds estratificados no treino (acurácia balanceada), em que cada
fold refaz escala, Map, Optimize e seleção de features. O melhor modelo de cada grid é
reajustado no treino inteiro e avaliado uma vez no teste. No fim, Wilcoxon pareado nos
folds entre o melhor MOL e os demais.

    python experimentos/principal.py
    python experimentos/principal.py --quick          # poucos folds e épocas, só para testar
    python experimentos/principal.py --skip-baselines
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')   # só grava figuras em arquivo
warnings.filterwarnings('ignore')

import keras
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from mol.baselines import build_baselines, n_selected_features
from mol.cnn import MOLCNNClassifier
from mol.data import prepare_data_pipeline
from mol.utils import (calculate_metrics, create_directories, load_config, parse_grid,
                       plot_comparison, plot_mol_grid, print_summary, visualize_mol_images,
                       wilcoxon_test)

RAIZ = Path(__file__).resolve().parents[1]


def apply_quick_mode(cfg):
    cfg['cross_validation']['n_splits'] = 3
    cfg['mol']['map_strategies'] = cfg['mol']['map_strategies'][:1]
    cfg['mol']['optimizers'] = cfg['mol']['optimizers'][:1]
    cfg['mol']['cnn']['epochs'] = 10
    cfg['mol']['cnn']['learning_rate_grid'] = [0.01]
    cfg['baselines']['grid_set'] = 'quick'
    return cfg


def run_grid_search(name, estimator, param_grid, data, cv, scoring, n_jobs):
    t0 = time.time()
    gs = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring,
                      n_jobs=n_jobs, refit=True, error_score='raise')
    gs.fit(data['X_train'], data['y_train'])

    best = gs.best_index_
    n_splits = cv.get_n_splits()
    fold_scores = np.array([gs.cv_results_[f'split{k}_test_score'][best] for k in range(n_splits)])

    y_pred = gs.predict(data['X_test'])
    y_proba = gs.predict_proba(data['X_test'])[:, 1]
    test = calculate_metrics(data['y_test'], y_pred, y_proba)

    row = {
        'model': name,
        'cv_balanced_accuracy_mean': fold_scores.mean(),
        'cv_balanced_accuracy_std': fold_scores.std(),
        **{f'test_{k}': v for k, v in test.items()},
        'best_params': json.dumps({k: (v if isinstance(v, (int, float, str)) else str(v))
                                   for k, v in gs.best_params_.items()}),
        'time_s': round(time.time() - t0, 1),
    }
    print(f"  {name:<28} CV {row['cv_balanced_accuracy_mean']:.4f} ± "
          f"{row['cv_balanced_accuracy_std']:.4f} | teste {test['balanced_accuracy']:.4f} "
          f"| {row['time_s']}s", flush=True)
    return gs, row, fold_scores


def main(args):
    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_mode(cfg)

    out_dir = RAIZ / cfg['output']['results_dir']
    if args.quick:
        out_dir = out_dir.with_name(out_dir.name + '_quick')   # não sobrescreve os resultados reais
    create_directories(str(out_dir))

    seed = cfg['dataset']['random_state']
    keras.utils.set_random_seed(seed)

    print("\n[1/4] Dados")
    data = prepare_data_pipeline(RAIZ / cfg['dataset']['path'], cfg['dataset']['test_size'], seed)
    n_features = data['X_train'].shape[1]

    cv_cfg = cfg['cross_validation']
    cv = StratifiedKFold(n_splits=cv_cfg['n_splits'], shuffle=cv_cfg['shuffle'],
                         random_state=cv_cfg['random_state'])
    scoring = cv_cfg['scoring']

    rows, folds = [], {}
    best_mol = None

    def save_partial():
        pd.DataFrame(rows).to_csv(out_dir / 'results.csv', index=False)
        pd.DataFrame(folds).to_csv(out_dir / 'fold_scores.csv', index_label='fold')

    if not args.skip_mol:
        mol_cfg = cfg['mol']
        cnn_cfg = mol_cfg['cnn']
        print(f"\n[2/4] MOL + CNN ({cv.get_n_splits()} folds, grid de learning rate "
              f"{cnn_cfg['learning_rate_grid']})")
        for map_strategy in mol_cfg['map_strategies']:
            for optimizer in mol_cfg['optimizers']:
                name = f"MOL-{map_strategy}-{optimizer}"
                est = MOLCNNClassifier(
                    map_strategy=map_strategy, optimizer=optimizer,
                    n_population=mol_cfg['n_population'],
                    max_evaluations=mol_cfg['max_evaluations'],
                    sample_order=mol_cfg['sample_order'], min_pixels=mol_cfg['min_pixels'],
                    scaler=mol_cfg['scaler'], epochs=cnn_cfg['epochs'],
                    batch_size=cnn_cfg['batch_size'], early_stopping=cnn_cfg['early_stopping'],
                    random_state=seed)
                gs, row, fs = run_grid_search(
                    name, est, {'learning_rate': cnn_cfg['learning_rate_grid']},
                    data, cv, scoring, n_jobs=1)

                mol = gs.best_estimator_.mol_
                row.update({'family': 'MOL', 'fs_strategy': 'MOL', 'map': map_strategy,
                            'optimizer': optimizer, 'n_features': int(mol.selected_mask_.sum()),
                            'image_shape': f"{mol.image_shape_[0]}x{mol.image_shape_[1]}",
                            'selected_features': json.dumps(
                                [data['feature_names'][i] for i in mol.selected_features_])})
                rows.append(row)
                folds[name] = fs
                if best_mol is None or row['cv_balanced_accuracy_mean'] > best_mol[1]['cv_balanced_accuracy_mean']:
                    best_mol = (gs.best_estimator_, row)
                save_partial()

    if not args.skip_baselines:
        b_cfg = cfg['baselines']
        grids = {k: parse_grid(v) if isinstance(v, dict) else v
                 for k, v in b_cfg['grids'][b_cfg['grid_set']].items()}
        print(f"\n[3/4] Baselines (grid '{b_cfg['grid_set']}', FS {b_cfg['fs_strategies']})")
        for model_name, fs, pipe, grid in build_baselines(
                grids, b_cfg['fs_strategies'], b_cfg['rfe_step'], seed, n_features):
            name = f"{model_name}-{fs}"
            gs, row, fs_scores = run_grid_search(name, pipe, grid, data, cv, scoring, n_jobs=-1)
            row.update({'family': model_name, 'fs_strategy': fs, 'map': '', 'optimizer': '',
                        'n_features': n_selected_features(gs.best_estimator_, n_features),
                        'image_shape': '', 'selected_features': ''})
            rows.append(row)
            folds[name] = fs_scores
            save_partial()

    print("\n[4/4] Relatório")
    results_df = pd.DataFrame(rows)
    save_partial()

    wilcoxon_df = pd.DataFrame()
    if best_mol is not None and len(rows) > 1:
        ref = best_mol[1]['model']
        wilcoxon_df = pd.DataFrame([{
            'reference': ref,
            'compared_to': r['model'],
            'diff_cv_mean': best_mol[1]['cv_balanced_accuracy_mean'] - r['cv_balanced_accuracy_mean'],
            'p_value': wilcoxon_test(folds[ref], folds[r['model']]),
        } for r in rows if r['model'] != ref])
        wilcoxon_df['significant_0.05'] = wilcoxon_df['p_value'] < 0.05
        wilcoxon_df.to_csv(out_dir / 'wilcoxon.csv', index=False)

    if len(results_df):
        plot_comparison(results_df, out_dir / 'comparison_plot.png')

    if best_mol is not None:
        clf = best_mol[0]
        images = clf.transform_to_images(data['X_train'])
        visualize_mol_images(images, data['y_train'], cfg['output']['n_images_per_class'], str(out_dir))
        plot_mol_grid(images, data['y_train'], out_dir / 'mol_images_grid.png')
        mol = clf.mol_
        params = {
            'model': best_mol[1]['model'],
            'image_shape': list(mol.image_shape_),
            'objective_value': float(mol.objective_value_),
            'pixel_order_features': [data['feature_names'][i] for i in mol.selected_features_],
            'map_ranking': [data['feature_names'][i] for i in mol.feature_order_],
            'learning_rate': clf.learning_rate,
        }
        (out_dir / 'best_mol_params.json').write_text(json.dumps(params, indent=2, ensure_ascii=False))

    print_summary(results_df, wilcoxon_df)
    print(f"\nResultados em {out_dir}")
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOL - Breast Cancer Wisconsin')
    parser.add_argument('--config', default=str(RAIZ / 'experimentos' / 'config.yaml'))
    parser.add_argument('--quick', action='store_true', help='teste rápido do pipeline')
    parser.add_argument('--skip-mol', action='store_true')
    parser.add_argument('--skip-baselines', action='store_true')
    main(parser.parse_args())
