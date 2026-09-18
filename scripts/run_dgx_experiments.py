import subprocess
from pathlib import Path

from ultralytics import YOLO

WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)


def get_best_gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if line:
                idx, mem_free, util = map(int, [x.strip() for x in line.split(",")])
                gpus.append((idx, mem_free, util))

        gpus.sort(key=lambda x: (-x[1], x[2]))

        best_gpu_id = gpus[0][0]
        print(
            f"\n[Monitoramento] GPU {best_gpu_id} selecionada dinamicamente! (Memória Livre: {gpus[0][1]} MB | Uso: {gpus[0][2]}%)\n"
        )
        return str(best_gpu_id)

    except Exception as e:
        print(f"Erro ao tentar ler as GPUs: {e}. Usando GPU '0' como fallback padrão.")
        return "0"


datasets = ["dataset/Talhao-GlobalDrones/data.yaml", "dataset/Talhao-Roboflow/data.yaml"]

models = ["yolov8n-seg.pt", "yolo11n-seg.pt", "yolo26n-seg.pt"]

epochs = 500
patience = 50
img_size = 640

for data_yaml in datasets:
    dataset_name = data_yaml.split("/")[1]

    for model_name in models:
        best_gpu = get_best_gpu()

        print(f"\n{'=' * 65}")
        print("INICIANDO NOVO TREINAMENTO")
        print(f"Dataset: {dataset_name}")
        print(f"Modelo:  {model_name}")
        print(f"GPU ID:  {best_gpu}")
        print(f"{'=' * 65}\n")

        model = YOLO(WEIGHTS_DIR / model_name)

        project_name = "reports/experiments"
        run_name = f"{dataset_name}_{model_name.replace('.pt', '')}"

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            patience=patience,
            imgsz=img_size,
            device=best_gpu,
            project=project_name,
            name=run_name,
            task="segment",
        )
