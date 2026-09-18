from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AgroSmart
    agrosmart_sdk_token: str = Field(
        default="",
        description="Token Bearer para a API do AgroSmart Dataset SDK.",
    )
    agrosmart_dataset_id: str = Field(
        default="",
        description="UUID do dataset para baixar do AgroSmart.",
    )
    agrosmart_api_url: str = Field(
        default="https://api.globalagrosmart.com.br",
        description="URL base para a API da AgroSmart.",
    )

    # Roboflow
    roboflow_api_key: str = Field(
        default="",
        description="Chave de API para a conta do Roboflow.",
    )
    roboflow_workspace: str = Field(
        default="global-mapeamento",
        description="Slug do workspace do Roboflow.",
    )
    roboflow_project: str = Field(
        default="talhao-agrosmart-onarz",
        description="Slug do projeto do Roboflow.",
    )
    roboflow_version: int = Field(
        default=10,
        description="Versão do dataset do Roboflow a ser baixada.",
    )

    # Caminhos
    dataset_dir: Path = Field(
        default=Path("dataset"),
        description="Diretório raiz para todos os datasets.",
    )
    agrosmart_dataset_name: str = Field(
        default="Talhao-GlobalDrones",
        description="Nome da pasta para o dataset AgroSmart dentro de dataset_dir.",
    )
    roboflow_dataset_name: str = Field(
        default="Talhao-Roboflow",
        description="Nome da pasta para o dataset Roboflow dentro de dataset_dir.",
    )
    runs_dir: Path = Field(
        default=Path("runs"),
        description="Diretório raiz onde o YOLO salva os resultados do treinamento.",
    )
    configs_dir: Path = Field(
        default=Path("configs"),
        description="Diretório contendo os arquivos de configuração de hiperparâmetros do YOLO.",
    )

    # Divisão do Dataset
    split_random_seed: int = Field(
        default=42,
        description="Semente aleatória para divisões de treino/val/teste reproduzíveis.",
    )
    split_train_ratio: float = Field(default=0.70, ge=0.0, le=1.0)
    split_val_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    split_test_ratio: float = Field(default=0.15, ge=0.0, le=1.0)

    # Treinamento
    device: str = Field(
        default="0",
        description="Dispositivo de treinamento: '0' para primeira GPU, 'cpu', ou ids de GPU separados por vírgula.",
    )
    train_config_file: str = Field(
        default="train_params.yaml",
        description="Nome do arquivo YAML de hiperparâmetros do YOLO dentro de configs_dir.",
    )

    # Propriedades computadas
    @property
    def agrosmart_dataset_dir(self) -> Path:
        return self.dataset_dir / self.agrosmart_dataset_name

    @property
    def roboflow_dataset_dir(self) -> Path:
        return self.dataset_dir / self.roboflow_dataset_name

    @property
    def train_config_path(self) -> Path:
        return self.configs_dir / self.train_config_file

    # Validadores
    @field_validator("split_train_ratio", "split_val_ratio", "split_test_ratio")
    @classmethod
    def _validate_ratio(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"As proporções de divisão devem estar entre 0 e 1, recebido {v}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings: Settings = get_settings()
