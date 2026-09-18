from __future__ import annotations

from pathlib import Path

from talhao.config import Settings
from talhao.config import settings as _default_settings
from talhao.utils.logging import get_logger

logger = get_logger(__name__)


class RoboflowDownloader:
    _DOWNLOAD_FORMAT: str = "yolov8"

    def __init__(self, settings: Settings = _default_settings) -> None:
        self._settings = settings

    def run(self) -> Path:
        self._validate_credentials()

        from roboflow import Roboflow

        api_key = self._settings.roboflow_api_key
        workspace = self._settings.roboflow_workspace
        project_name = self._settings.roboflow_project
        version_number = self._settings.roboflow_version
        destination = str(self._settings.roboflow_dataset_dir)

        logger.info(
            f"Iniciando download: [bold]{workspace}/{project_name}[/bold] "
            f"v{version_number} → {destination}"
        )

        try:
            rf = Roboflow(api_key=api_key)
            version = rf.workspace(workspace).project(project_name).version(version_number)

            logger.info(f"Baixando no formato [cyan]{self._DOWNLOAD_FORMAT}[/cyan]...")
            dataset = version.download(self._DOWNLOAD_FORMAT, location=destination)

            result_path = Path(dataset.location)
            logger.info(f"[green]Dataset baixado com sucesso em:[/green] {result_path}")

        except Exception:
            logger.exception("Erro ao baixar dataset do Roboflow")
            raise

        return result_path

    def _validate_credentials(self) -> None:
        if not self._settings.roboflow_api_key:
            logger.error("ROBOFLOW_API_KEY não encontrada no .env")
            raise ValueError("ROBOFLOW_API_KEY não definida.")
