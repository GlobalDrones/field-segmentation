from __future__ import annotations

import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from talhao.config import Settings
from talhao.config import settings as _default_settings
from talhao.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SplitResult:
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    total_captures: int = 0
    output_dir: Path = field(default_factory=Path)

    @property
    def total_images(self) -> int:
        return self.train_count + self.val_count + self.test_count

    def __str__(self) -> str:
        return (
            f"SplitResult("
            f"train={self.train_count}, val={self.val_count}, "
            f"test={self.test_count}, total={self.total_images}, "
            f"captures={self.total_captures})"
        )


class DatasetSplitter:
    def __init__(
        self,
        dataset_dir: Path | None = None,
        classes: list[str] | None = None,
        train_ratio: float | None = None,
        val_ratio: float | None = None,
        test_ratio: float | None = None,
        random_seed: int | None = None,
        settings: Settings = _default_settings,
    ) -> None:
        self._settings = settings
        self.dataset_dir = dataset_dir or settings.agrosmart_dataset_dir
        self.classes = classes or ["talhao-virtual"]
        self.train_ratio = train_ratio if train_ratio is not None else settings.split_train_ratio
        self.val_ratio = val_ratio if val_ratio is not None else settings.split_val_ratio
        self.test_ratio = test_ratio if test_ratio is not None else settings.split_test_ratio
        self.random_seed = random_seed if random_seed is not None else settings.split_random_seed

        self._validate_ratios()

    def run(self) -> SplitResult:
        self._guard_against_re_split()

        backup_dir = self.dataset_dir.with_name(f"{self.dataset_dir.name}_backup")
        result = SplitResult(output_dir=self.dataset_dir)

        try:
            if backup_dir.exists():
                logger.info(f"Limpando backup residual anterior: {backup_dir.name}")
                shutil.rmtree(backup_dir)

            logger.info(f"Movendo dados originais para backup temporário: {backup_dir.name}")
            self.dataset_dir.rename(backup_dir)

            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self._create_split_dirs(self.dataset_dir)

            captures = self._index_captures(backup_dir)
            result.total_captures = len(captures)

            splits = self._assign_captures(captures)
            result = self._copy_splits(splits, self.dataset_dir, result)

            self._write_yaml(self.dataset_dir)

            shutil.rmtree(backup_dir)
            logger.info(f"[green]Dataset organizado com sucesso em:[/green] {self.dataset_dir}")

        except Exception:
            logger.exception("Erro durante o split. Iniciando rollback...")
            if self.dataset_dir.exists():
                shutil.rmtree(self.dataset_dir)
            if backup_dir.exists():
                backup_dir.rename(self.dataset_dir)
                logger.info("[yellow]Rollback concluído. Diretório original restaurado.[/yellow]")
            raise

        return result

    def _validate_ratios(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio deve ser 1.0, mas é {total:.4f}"
            )

    def _guard_against_re_split(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Diretório do dataset não encontrado: {self.dataset_dir}")
        if (self.dataset_dir / "data.yaml").exists() or (self.dataset_dir / "train").exists():
            raise RuntimeError(
                f"O dataset em '{self.dataset_dir}' já parece ter sido dividido. "
                "Remova as pastas 'train', 'valid', 'test' e o arquivo 'data.yaml' "
                "para re-executar o split."
            )

    def _create_split_dirs(self, output_dir: Path) -> None:
        for split_name, ratio in [
            ("train", self.train_ratio),
            ("valid", self.val_ratio),
            ("test", self.test_ratio),
        ]:
            if ratio > 0:
                (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
                (output_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    def _index_captures(self, source_dir: Path) -> dict[str, list[Path]]:
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"Pastas 'images' e 'labels' não encontradas em: {source_dir}")

        captures: dict[str, list[Path]] = defaultdict(list)
        total = 0

        for img_path in sorted(images_dir.glob("*.png")):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                capture_id = img_path.stem.split("-")[0]
                captures[capture_id].append(img_path)
                total += 1

        logger.info(f"Imagens válidas (com label): [bold]{total}[/bold]")
        logger.info(f"Capturas únicas: [bold]{len(captures)}[/bold]")

        if not captures:
            logger.warning("Nenhuma imagem com anotação correspondente foi encontrada.")

        return dict(captures)

    def _assign_captures(self, captures: dict[str, list[Path]]) -> dict[str, list[Path]]:
        capture_ids = list(captures.keys())
        random.seed(self.random_seed)
        random.shuffle(capture_ids)

        n = len(capture_ids)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        splits: dict[str, list[Path]] = {"train": [], "valid": [], "test": []}
        for cid in capture_ids[:train_end]:
            splits["train"].extend(captures[cid])
        for cid in capture_ids[train_end:val_end]:
            splits["valid"].extend(captures[cid])
        for cid in capture_ids[val_end:]:
            splits["test"].extend(captures[cid])

        for name, imgs in splits.items():
            if not imgs:
                logger.warning(
                    f"O conjunto '{name}' ficou vazio — "
                    "considere usar mais capturas ou ajustar as proporções."
                )

        return splits

    def _copy_splits(
        self,
        splits: dict[str, list[Path]],
        output_dir: Path,
        result: SplitResult,
    ) -> SplitResult:
        source_labels_dir: Path | None = None

        for split_name, img_list in splits.items():
            if not img_list:
                continue

            if source_labels_dir is None:
                source_labels_dir = img_list[0].parent.parent / "labels"

            logger.info(f"Copiando [bold]{len(img_list)}[/bold] arquivos → '{split_name}/'...")

            for img_path in img_list:
                shutil.copy2(img_path, output_dir / split_name / "images" / img_path.name)
                label_name = f"{img_path.stem}.txt"
                shutil.copy2(
                    source_labels_dir / label_name,
                    output_dir / split_name / "labels" / label_name,
                )

            count = len(img_list)
            if split_name == "train":
                result.train_count = count
            elif split_name == "valid":
                result.val_count = count
            elif split_name == "test":
                result.test_count = count

        return result

    def _write_yaml(self, output_dir: Path) -> None:
        data = {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(self.classes),
            "names": self.classes,
        }
        yaml_path = output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        logger.info(f"[green]data.yaml gerado em:[/green] {yaml_path}")
