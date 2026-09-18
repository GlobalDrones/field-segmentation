from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from talhao.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentationMetrics:
    run_name: str
    model: str
    dataset: str
    epochs_trained: int
    # Métricas de Bounding-box (Caixa limitadora)
    precision_box: float
    recall_box: float
    f1_box: float
    map50_box: float
    map50_95_box: float
    # Métricas de Máscara (Segmentação)
    precision_mask: float
    recall_mask: float
    f1_mask: float
    map50_mask: float
    map50_95_mask: float
    # Hardware e Performance
    model_size_mb: float = 0.0
    parameters_m: float = 0.0
    inference_ms: float = 0.0
    # Geral
    fitness: float = 0.0
    weights_path: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class ModelEvaluator:
    _COL_MAP = {
        "precision_box": "metrics/precision(B)",
        "recall_box": "metrics/recall(B)",
        "map50_box": "metrics/mAP50(B)",
        "map50_95_box": "metrics/mAP50-95(B)",
        "precision_mask": "metrics/precision(M)",
        "recall_mask": "metrics/recall(M)",
        "map50_mask": "metrics/mAP50(M)",
        "map50_95_mask": "metrics/mAP50-95(M)",
        "fitness": "fitness",
        "epochs_trained": "epoch",
    }

    def extract(
        self,
        run_dir: str | Path,
        model: str = "",
        dataset: str = "",
    ) -> SegmentationMetrics:
        run_dir = Path(run_dir)
        csv_path = run_dir / "results.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"results.csv não encontrado em: {run_dir}\n"
                "Certifique-se de que o treinamento foi concluído com 'plots=True'."
            )

        best_row = self._read_best_row(csv_path)
        weights_path = run_dir / "weights" / "best.pt"

        pb = float(self._get(best_row, "precision_box", "0"))
        rb = float(self._get(best_row, "recall_box", "0"))
        pm = float(self._get(best_row, "precision_mask", "0"))
        rm = float(self._get(best_row, "recall_mask", "0"))

        f1_box = (2 * pb * rb) / (pb + rb) if (pb + rb) > 0 else 0.0
        f1_mask = (2 * pm * rm) / (pm + rm) if (pm + rm) > 0 else 0.0

        metrics = SegmentationMetrics(
            run_name=run_dir.name,
            model=model,
            dataset=dataset,
            epochs_trained=int(float(self._get(best_row, "epochs_trained", "0"))),
            precision_box=pb,
            recall_box=rb,
            f1_box=f1_box,
            map50_box=float(self._get(best_row, "map50_box", "0")),
            map50_95_box=float(self._get(best_row, "map50_95_box", "0")),
            precision_mask=pm,
            recall_mask=rm,
            f1_mask=f1_mask,
            map50_mask=float(self._get(best_row, "map50_mask", "0")),
            map50_95_mask=float(self._get(best_row, "map50_95_mask", "0")),
            fitness=float(self._get(best_row, "fitness", "0")),
            weights_path=str(weights_path.resolve()) if weights_path.exists() else "",
        )

        if metrics.weights_path:
            self._extract_hardware_metrics(metrics)

        logger.info(
            f"[bold]{metrics.run_name}[/bold] — "
            f"F1(M)=[cyan]{metrics.f1_mask:.4f}[/cyan] | "
            f"mAP50(M)=[green]{metrics.map50_mask:.4f}[/green] | "
            f"Latência=[yellow]{metrics.inference_ms:.1f}ms[/yellow]"
        )
        return metrics

    def _extract_hardware_metrics(self, metrics: SegmentationMetrics) -> None:
        import os

        import numpy as np
        from ultralytics import YOLO

        try:
            metrics.model_size_mb = os.path.getsize(metrics.weights_path) / (1024 * 1024)

            model_obj = YOLO(metrics.weights_path)

            params = sum(x.numel() for x in model_obj.model.parameters()) / 1e6
            metrics.parameters_m = params

            dummy_img = np.zeros((1024, 1024, 3), dtype=np.uint8)

            model_obj.predict(dummy_img, imgsz=1024, verbose=False)

            res = model_obj.predict(dummy_img, imgsz=1024, verbose=False)[0]
            metrics.inference_ms = res.speed.get("inference", 0.0)

        except Exception as e:
            logger.warning(
                f"Aviso: Não foi possível calcular métricas de hardware para {metrics.run_name}. Erro: {e}"
            )

    def export_comparison(
        self,
        metrics_list: list[SegmentationMetrics],
        output_dir: str | Path = Path("reports"),
        report_name: str = "metrics_comparison",
    ) -> dict[str, Path]:
        if not metrics_list:
            raise ValueError("metrics_list não pode ser vazio.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"{report_name}.json"
        csv_path = output_dir / f"{report_name}.csv"

        rows = [m.as_dict() for m in metrics_list]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        logger.info(f"[green]Relatório JSON exportado:[/green] {json_path}")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"[green]Relatório CSV exportado:[/green] {csv_path}")

        self._print_summary_table(metrics_list)

        return {"json": json_path, "csv": csv_path}

    def print_metrics(self, metrics: SegmentationMetrics) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f"Métricas — {metrics.run_name}", show_header=True)
            table.add_column("Métrica", style="cyan", no_wrap=True)
            table.add_column("Valor", style="bold green", justify="right")

            rows = [
                ("Dataset", metrics.dataset),
                ("Modelo", metrics.model),
                ("Épocas Treinadas", str(metrics.epochs_trained)),
                ("──── Máscara (Segmentação) ────", ""),
                ("F1-Score (M)", f"{metrics.f1_mask:.4f}"),
                ("mAP50 (M)", f"{metrics.map50_mask:.4f}"),
                ("mAP50-95 (M)", f"{metrics.map50_95_mask:.4f}"),
                ("Precision (M)", f"{metrics.precision_mask:.4f}"),
                ("Recall (M)", f"{metrics.recall_mask:.4f}"),
                ("──── Performance (Hardware) ────", ""),
                ("Tamanho do Arquivo", f"{metrics.model_size_mb:.2f} MB"),
                ("Parâmetros", f"{metrics.parameters_m:.2f} M"),
                ("Latência de Inferência", f"{metrics.inference_ms:.2f} ms"),
                ("──── Bounding Box (Geral) ────", ""),
                ("F1-Score (B)", f"{metrics.f1_box:.4f}"),
                ("mAP50 (B)", f"{metrics.map50_box:.4f}"),
                ("Fitness Score Global", f"{metrics.fitness:.4f}"),
            ]
            for label, value in rows:
                table.add_row(label, value)

            console.print(table)
        except ImportError:
            logger.info(str(metrics))

    def _read_best_row(self, csv_path: Path) -> dict[str, str]:
        """Lê o results.csv e retorna a linha com a maior pontuação de fitness."""
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]

        if not rows:
            raise ValueError(f"O arquivo {csv_path} não contém dados.")

        fitness_col = self._COL_MAP["fitness"]

        try:
            best = max(rows, key=lambda r: float(r.get(fitness_col, 0) or 0))
        except (ValueError, TypeError):
            best = rows[-1]

        return best

    def _get(self, row: dict[str, str], field: str, default: str = "0") -> str:
        col_name = self._COL_MAP.get(field, field)
        return row.get(col_name, default) or default

    def _print_summary_table(self, metrics_list: list[SegmentationMetrics]) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Comparação de Modelos", show_header=True)
            table.add_column("Run", style="cyan")
            table.add_column("Dataset", style="white")
            table.add_column("F1 (M)", justify="right", style="bold green")
            table.add_column("mAP50 (M)", justify="right", style="green")
            table.add_column("mAP50-95", justify="right")
            table.add_column("Peso", justify="right", style="yellow")
            table.add_column("Params", justify="right", style="yellow")
            table.add_column("Latência", justify="right", style="bold yellow")

            for m in metrics_list:
                table.add_row(
                    m.run_name,
                    m.dataset,
                    f"{m.f1_mask:.4f}",
                    f"{m.map50_mask:.4f}",
                    f"{m.map50_95_mask:.4f}",
                    f"{m.model_size_mb:.1f} MB",
                    f"{m.parameters_m:.1f} M",
                    f"{m.inference_ms:.1f} ms",
                )

            console.print(table)
        except ImportError:
            for m in metrics_list:
                logger.info(str(m))
