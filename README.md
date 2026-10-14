# Map-Optimize-Learn em dados tabulares: implementação, avaliação e análise crítica

Implementação do método **Map-Optimize-Learn (MOL)**, de Serra Neto, Dutra e Mollinetti
(IJCNN 2022), que transforma cada linha de uma tabela numa pequena imagem para classificá-la
com uma rede convolucional. O método foi implementado a partir do texto do artigo e avaliado na base Breast Cancer Wisconsin (classificação benigno/maligno).

O repositório também traz os testes de controle que explicam os resultados, uma versão
corrigida do método (**MOL v2**) e uma apresentação animada de todo o trabalho.

## Resultados em resumo

Acurácia balanceada média em validação cruzada estratificada de 10 folds:

| Modelo | Acurácia balanceada | Features |
|---|---|---|
| Rede neural tabular (melhor baseline) | 0,974 | 30 |
| Regressão logística | 0,972 | 30 |
| MOL v2 (arranjo por correlação, imagem 5×6) | 0,970 | 30 |
| MOL original, Map euclidiano + ABC (melhor MOL) | 0,954 | 16 |
| MOL original, Map euclidiano + PSO | 0,931 | 2 |

- O MOL não supera os métodos tradicionais nesta base. A melhor versão empata
  estatisticamente com eles (Wilcoxon pareado, p > 0,05).
- A função objetivo do artigo (Eq. 1) tem três problemas de projeto: não depende do formato da
  imagem, não depende da posição dos pixels e seu mínimo é sempre uma imagem com 2 features.
  Por isso os otimizadores mais eficientes (PSO, EPSO) classificam pior.
- O MOL v2 corrige esses problemas e recupera o desempenho, mas o arranjo espacial não traz
  ganho sobre uma rede comum ou uma regressão logística nesta base.

Os detalhes estão em [results/controles/RELATORIO.md](results/controles/RELATORIO.md) e
[results/mol_v2/RELATORIO.md](results/mol_v2/RELATORIO.md).

Limite destes resultados: uma única base, pequena (569 pacientes, 30 features), em que quase
todo modelo passa de 0,95. Eles não dizem como o método se comporta com centenas ou milhares
de features correlacionadas.

## Estrutura

```
├── src/mol/              pacote com a implementação
│   ├── algorithm.py        Map e Optimize (transformer scikit-learn)
│   ├── optimizers.py       PSO, EPSO, ABC e busca aleatória
│   ├── cnn.py              CNN do artigo e MOLCNNClassifier
│   ├── v2.py               MOL v2 (MOLv2Classifier)
│   ├── baselines.py        ANN, LR, KNN, RF, XGBoost e LightGBM em S / E / W
│   ├── data.py             leitura do CSV e divisão treino/teste
│   └── utils.py            métricas, Wilcoxon e gráficos
├── src/eda_utils.py      funções de apoio do notebook de EDA
├── experimentos/         scripts que geram os resultados
├── results/              saídas de cada experimento (CSVs, figuras, relatórios)
├── models/               modelos finais treinados
├── data/                 breast-cancer.csv
├── notebooks/            exploração dos dados e aplicação do método
└── apresentacao/         cenas em Manim, dados das cenas e versões HTML
```

## Instalação

São dois ambientes separados, porque o Manim e o TensorFlow pedem versões diferentes do numpy.
Tudo foi testado com Python 3.13 no macOS (Apple Silicon).

### Experimentos e notebooks

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O `requirements.txt` instala as dependências com versões fixas e o pacote `mol` em modo editável.

### Apresentação

```bash
python3.13 -m venv apresentacao/.venv
apresentacao/.venv/bin/pip install -r apresentacao/requirements.txt
```

As fórmulas das cenas precisam de LaTeX e do `dvisvgm`. No macOS:

```bash
brew install texlive dvisvgm
```

## Como reproduzir

Com o ambiente dos experimentos ativo, a partir da raiz do repositório:

| Comando | O que faz |
|---|---|
| `python experimentos/principal.py` | MOL (5 critérios de Map × 3 otimizadores) e baselines, com o protocolo do artigo |
| `python experimentos/controles.py` | testes de controle |
| `python experimentos/analisar_controles.py` | tabelas, Wilcoxon e figura dos controles |
| `python experimentos/mol_v2.py` | MOL v2 e seus controles |
| `python experimentos/analisar_mol_v2.py` | tabelas, Wilcoxon e figuras do MOL v2 |
| `python experimentos/salvar_modelos.py` | treina e salva os modelos finais em `models/` 

Para testar se tudo está funcionando sem esperar, `principal.py` aceita `--quick` e os outros
dois experimentos aceitam `--rapido`. As configurações do experimento principal ficam em
[experimentos/config.yaml](experimentos/config.yaml); os grids completos da Tabela III do artigo
estão lá como `grid_set: paper`.

### Protocolo

- Divisão estratificada 80/20 em treino e teste.
- Grid search com 10 folds estratificados no treino; métrica: acurácia balanceada.
- Dentro de cada fold, tudo é refeito só com os dados de treino: escala, ranking do Map,
  otimização, seleção de features e modelo.
- O teste é usado uma única vez, com o modelo já escolhido.
- Comparações com o teste de Wilcoxon pareado nos 10 folds.

### Decisões de implementação

O artigo deixa três pontos sem especificar. As escolhas feitas aqui:

| Ponto em aberto | Escolha | Onde mudar |
|---|---|---|
| Ordem das amostras na Eq. 1 | ordenadas por classe | `mol.sample_order` no `config.yaml` |
| Tratamento das restrições da otimização | decodificação com reparo | `MapOptimizeLearn._decode` |
| "Distância euclidiana" no Map | distância entre os centróides das classes | `map_strategy` |

## Notebooks

| Notebook | Conteúdo |
|---|---|
| [01_exploracao.ipynb](notebooks/01_exploracao.ipynb) | escalas, distribuições por diagnóstico, correlações, valores extremos e PCA |
| [02_aplicacao_mol.ipynb](notebooks/02_aplicacao_mol.ipynb) | o MOL etapa por etapa, comparação com os baselines, testes de controle, MOL v2 e uso dos modelos salvos |

Os dois notebooks já vêm executados, com as saídas e as figuras. Para rodar de novo, use o ambiente dos experimentos:

```bash
jupyter notebook notebooks/
```

O `02_aplicacao_mol.ipynb` treina alguns modelos pequenos (menos de um minuto no total) e lê os
resultados dos experimentos completos em `results/`.

## Modelos salvos

`models/` guarda os modelos treinados no conjunto de treino, e `models/metricas_teste.json`
traz o desempenho de cada um no teste.

```python
import joblib
import keras
from mol.data import prepare_data_pipeline

dados = prepare_data_pipeline()

# MOL original
mol = joblib.load("models/mol_original/map_optimize.joblib")
cnn = keras.models.load_model("models/mol_original/cnn.keras")
prob = cnn.predict(mol.transform(dados["X_test"])[..., None])[:, 0]

# MOL v2
arranjo = joblib.load("models/mol_v2/arranjo.joblib")
cnn_v2 = keras.models.load_model("models/mol_v2/cnn.keras")
z = arranjo["scaler"].transform(dados["X_test"])[:, arranjo["colunas"]]
prob_v2 = cnn_v2.predict(z.reshape(-1, *arranjo["forma"], 1))[:, 0]

# Baselines
lr = joblib.load("models/lr.joblib")
prob_lr = lr.predict_proba(dados["X_test"])[:, 1]
```

## Apresentação

A apresentação tem seis cenas feitas com [Manim](https://www.manim.community/) e
[manim-slides](https://manim-slides.eertmans.be/): contexto, Map, Optimize, Learn, avaliação e
análise crítica.

Para assistir sem instalar nada, abra os arquivos `apresentacao/cena*.html` no navegador
(cada um precisa da pasta `_assets` correspondente ao lado).

Para renderizar e apresentar, de dentro de `apresentacao/`:

```bash
.venv/bin/manim-slides render cena1_contexto.py Cena1Contexto
.venv/bin/manim-slides present Cena1Contexto Cena2Map Cena3Optimize Cena4Learn Cena5Avaliacao Cena6Critica
```

Cada cena precisa ser renderizada antes de apresentar. Os nomes das classes são
`Cena1Contexto`, `Cena2Map`, `Cena3Optimize`, `Cena4Learn`, `Cena5Avaliacao` e `Cena6Critica`.

Os números mostrados nas cenas vêm de `apresentacao/dados/mol_dados.json`. Para regenerá-lo
depois de rodar os experimentos de novo, use o ambiente dos experimentos:

```bash
python apresentacao/gerar_dados.py
```

## Dados

`data/breast-cancer.csv` é a base Breast Cancer Wisconsin (Diagnostic), obtida no
[Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data): 569 pacientes,
30 features numéricas medidas no núcleo das células e o diagnóstico (M = maligno, B = benigno).

## Licença

O código está sob a licença MIT (veja [LICENSE](LICENSE)). A base de dados segue os termos
de uso da fonte original.

## Referências

- Serra Neto, M. T. R.; Dutra, I.; Mollinetti, M. A. F. Map-Optimize-Learn: Predicting Cardiac
  Pathology in Children and Teenagers with a Deep Learning Based Tabular Learning Method.
  IJCNN 2022. DOI: [10.1109/IJCNN55064.2022.9889788](https://doi.org/10.1109/IJCNN55064.2022.9889788)
- Zhu, Y. et al. Converting tabular data into images for deep learning with convolutional
  neural networks. Scientific Reports, 2021. (IGTD, origem do objetivo usado no MOL v2)
- Miranda, V.; Fonseca, N. EPSO: Evolutionary Particle Swarm Optimization. IEEE/PES T&D, 2002.
- Karaboga, D.; Basturk, B. On the performance of artificial bee colony (ABC) algorithm.
  Applied Soft Computing, 2008.
