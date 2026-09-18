from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from talhao.config import Settings
from talhao.config import settings as _default_settings
from talhao.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    dataset_dir: Path
    run_name: str
    model: str = "yolo11n-seg.pt"
    device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def data_yaml(self) -> Path:
        return self.dataset_dir / "data.yaml"


class YOLOTrainer:
    def __init__(self, settings: Settings = _default_settings) -> None:
        self._settings = settings

    def train(self, config: TrainConfig) -> Path:
        self._validate_paths(config)

        from ultralytics import YOLO

        device = config.device or self._settings.device
        cfg_path = self._settings.train_config_path

        logger.info(f"Carregando modelo: [bold]{config.model}[/bold]")
        model = YOLO(config.model)

        logger.info(
            f"Iniciando treinamento → "
            f"dataset=[cyan]{config.dataset_dir.name}[/cyan] | "
            f"device=[cyan]{device}[/cyan] | "
            f"run=[cyan]{config.run_name}[/cyan]"
        )

        model.train(
            model=config.model,
            data=str(config.data_yaml),
            cfg=str(cfg_path),
            name=config.run_name,
            exist_ok=True,
            device=device,
            **config.extra,
        )

        output_dir: Path = Path(model.trainer.save_dir)
        logger.info(f"[green]Treinamento concluído.[/green] Resultados em: {output_dir}")
        return output_dir

    def validate(
        self,
        output_dir: Path,
        split: str = "val",
        model_path: Path | None = None,
    ) -> Any:
        from ultralytics import YOLO

        weights = model_path or (output_dir / "weights" / "best.pt")
        if not weights.exists():
            raise FileNotFoundError(f"Pesos do modelo não encontrados: {weights}")

        val_run_name = f"val_{split}"
        logger.info(f"Iniciando validação no conjunto '[bold]{split}[/bold]'...")

        model = YOLO(str(weights))
        results = model.val(
            split=split,
            project=str(output_dir),
            name=val_run_name,
        )

        logger.info(
            f"[green]Validação concluída.[/green] Resultados em: {output_dir / val_run_name}"
        )
        return results

    def _validate_paths(self, config: TrainConfig) -> None:
        if not config.data_yaml.exists():
            raise FileNotFoundError(
                f"Arquivo data.yaml não encontrado: {config.data_yaml}\n"
                "Execute o download e o split do dataset antes de treinar."
            )
        cfg_path = self._settings.train_config_path
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração YOLO não encontrado: {cfg_path}\n"
                f"Verifique se '{cfg_path.name}' existe em '{cfg_path.parent}'."
            )
