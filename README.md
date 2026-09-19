# Pipeline de Segmentação de Talhões Agrícolas com YOLO

Pipeline de **Segmentação de Instâncias** para mapeamento automático dos limites de talhões agrícolas em imagens de drone e satélite, usando as arquiteturas YOLO (YOLOv8, YOLO11, YOLO26).

---

## 🗂 Estrutura do Projeto

```
talhao/
├── src/
│   └── talhao/               # Pacote Python principal
│       ├── __init__.py       # Versão e exports públicos
│       ├── cli.py            # CLI unificada (Typer)
│       ├── config.py         # Settings tipado (Pydantic-Settings)
│       ├── data/
│       │   ├── downloader.py # AgroSmart SDK download
│       │   ├── roboflow.py   # Roboflow download
│       │   └── splitter.py   # Split train/val/test
│       ├── training/
│       │   ├── trainer.py    # Treino YOLO (YOLOTrainer)
│       │   └── evaluator.py  # Métricas pós-treino (ModelEvaluator)
│       └── utils/
│           └── logging.py    # Logging centralizado (Rich)
├── configs/
│   └── train_params.yaml     # Hiperparâmetros YOLO
├── dataset/                  # Dados (não versionados)
│   ├──
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 🛠 Instalação

O projeto usa **`uv`** como gerenciador de pacotes.

```bash
# Instala todas as dependências (incluindo o AgroSmart SDK via GitHub)
uv sync
```

Copie o arquivo de variáveis de ambiente e preencha com suas credenciais:

```bash
cp .env.example .env
# Edite .env com seu editor preferido
```

---

## 💾 Obtenção dos Dados

Todos os comandos de dados são acessíveis via a CLI unificada `talhao`:

```bash
# Baixar o dataset interno (AgroSmart SDK)
uv run talhao download agrosmart

# Baixar o dataset validado (Roboflow v10)
uv run talhao download roboflow

# Dividir o dataset AgroSmart em train/val/test (70/15/15 por padrão)
uv run talhao split
```

> **Opções avançadas de split:**
> ```bash
> uv run talhao split --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 123
> ```

---

## 🚀 Treinamento

```bash
# Treinar no dataset AgroSmart
uv run talhao train \
  --dataset dataset/Talhao-GlobalDrones \
  --name treino_sdk \
  --model yolo11n-seg.pt \
  --eval \
  --test

# Treinar no dataset Roboflow
uv run talhao train \
  --dataset dataset/Talhao-Roboflow \
  --name treino_roboflow \
  --model yolo11n-seg.pt \
  --eval \
  --test
```

> **Dica:** Para ajustar épocas, batch size, learning rate e augmentations, edite `configs/train_params.yaml`.

---

## 📊 Métricas e Análise Comparativa

### Extrair métricas de um único run

```bash
uv run talhao evaluate runs/segment/treino_sdk \
  --model yolo11n-seg.pt \
  --dataset GlobalDrones
```

### Comparar dois ou mais runs lado a lado

```bash
uv run talhao compare \
  runs/segment/treino_sdk \
  runs/segment/treino_roboflow \
  --output-dir reports \
  --report-name comparacao_datasets
```

Isso gera `reports/comparacao_datasets.json` e `reports/comparacao_datasets.csv` com a tabela completa:

| Run | Dataset | mAP50 (M) | mAP50-95 (M) | Precision (M) | Recall (M) | Fitness |
|---|---|---|---|---|---|---|
| treino_sdk | GlobalDrones | — | — | — | — | — |
| treino_roboflow | Roboflow | — | — | — | — | — |

*(Os valores são preenchidos automaticamente após o treinamento.)*

---

## 👁 Inferência Visual

```bash
# Visualizar predições do modelo SDK
uv run yolo segment predict \
  model=runs/segment/treino_sdk/weights/best.pt \
  source=dataset/Talhao-GlobalDrones/test/images \
  save=True \
  name=resultado_sdk

# Visualizar predições do modelo Roboflow
uv run yolo segment predict \
  model=runs/segment/treino_roboflow/weights/best.pt \
  source=dataset/Talhao-GlobalDrones/test/images \
  save=True \
  name=resultado_roboflow
```

---

## ⚙️ Referência da CLI

```
talhao --help

Comandos disponíveis:
  download agrosmart   Baixa o dataset interno via AgroSmart SDK
  download roboflow    Baixa o dataset validado do Roboflow
  split                Divide o dataset AgroSmart em train/val/test
  train                Treina um modelo YOLO de segmentação
  evaluate             Extrai e exibe métricas de um run concluído
  compare              Compara métricas de dois ou mais runs
```

---

## 🔑 Variáveis de Ambiente

| Variável | Obrigatória | Descrição |
|---|---|---|
| `AGROSMART_SDK_TOKEN` | ✅ | Token de acesso à API AgroSmart |
| `AGROSMART_DATASET_ID` | ✅ | UUID do dataset no portal AgroSmart |
| `ROBOFLOW_API_KEY` | ✅ | Chave de API do Roboflow |
| `DEVICE` | — | Dispositivo de treino (`0`, `cpu`, `0,1`) |
| `ROBOFLOW_WORKSPACE` | — | Workspace Roboflow (padrão: `global-mapeamento`) |
| `ROBOFLOW_PROJECT` | — | Projeto Roboflow (padrão: `talhao-agrosmart-onarz`) |
| `ROBOFLOW_VERSION` | — | Versão do dataset Roboflow (padrão: `10`) |
| `SPLIT_RANDOM_SEED` | — | Seed para o split (padrão: `42`) |

Consulte `.env.example` para a lista completa.
