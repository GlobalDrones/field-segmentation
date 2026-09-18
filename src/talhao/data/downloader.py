from __future__ import annotations

import shutil
from pathlib import Path

from talhao.config import Settings
from talhao.config import settings as _default_settings
from talhao.utils.logging import get_logger

logger = get_logger(__name__)


class AgroSmartDownloader:
    def __init__(self, settings: Settings = _default_settings) -> None:
        self._settings = settings

    def run(
        self,
        tile_size_meters: int = 50,
        overlap: int = 10,
        min_annotation_coverage: int = 30,
        width: int = 1024,
        height: int = 1024,
        null_percent: int = 20,
        min_image_content_percent: int = 10,
        max_workers: int = 10,
    ) -> Path:

        self._validate_credentials()

        from agrosmartds import AgrosmartDatasetSDK

        dataset_id = self._settings.agrosmart_dataset_id
        result_dir = self._settings.dataset_dir
        final_dir = self._settings.agrosmart_dataset_dir

        logger.info("Iniciando conexão com a API da AgroSmart...")

        try:
            with AgrosmartDatasetSDK(
                access_token=self._settings.agrosmart_sdk_token,
                base_url=self._settings.agrosmart_api_url,
                result_dir=str(result_dir),
            ) as sdk:
                logger.info(f"Baixando metadados do dataset: [bold]{dataset_id}[/bold]")
                sdk.download_dataset(dataset_id)

                logger.info(
                    "Gerando grid de tiles "
                    f"(tile={tile_size_meters}m, overlap={overlap}m, "
                    f"min_coverage={min_annotation_coverage}%)..."
                )
                sdk.generate_tile_grids(
                    dataset_id,
                    tile_size_meters=tile_size_meters,
                    overlap=overlap,
                    min_annotation_coverage=min_annotation_coverage,
                )

                logger.info(f"Baixando tiles {width}×{height}px (formato segmentação)...")
                summary = sdk.download_tiles(
                    dataset_id,
                    width=width,
                    height=height,
                    format="segmentation",
                    null_percent=null_percent,
                    min_image_content_percent=min_image_content_percent,
                    max_workers=max_workers,
                )

                logger.info("=== Resumo do Download ===")
                logger.info(summary)
                logger.info("==========================")

            original_folder = result_dir / dataset_id
            if original_folder.exists():
                if final_dir.exists():
                    logger.info(f"Removendo diretório existente: {final_dir}")
                    shutil.rmtree(final_dir)
                original_folder.rename(final_dir)
                logger.info(f"[green]Dataset organizado em:[/green] {final_dir}")
            else:
                logger.warning(f"Pasta original não encontrada para renomear: {original_folder}")

        except Exception:
            logger.exception("Erro durante o download do AgroSmart")
            raise

        return final_dir

    def _validate_credentials(self) -> None:
        errors: list[str] = []
        if not self._settings.agrosmart_sdk_token:
            errors.append("AGROSMART_SDK_TOKEN não definido no .env")
        if not self._settings.agrosmart_dataset_id:
            errors.append("AGROSMART_DATASET_ID não definido no .env")
        if errors:
            for msg in errors:
                logger.error(msg)

            raise ValueError("; ".join(errors))
