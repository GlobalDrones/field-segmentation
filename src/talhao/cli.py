from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from talhao import __version__
from talhao.config import settings
from talhao.utils.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="talhao",
    help="Pipeline de Segmentação de Talhões Agrícolas com YOLO.",
    add_completion=False,
    rich_markup_mode="rich",
)

download_app = typer.Typer(help="Comandos de download de datasets.", add_completion=False)
app.add_typer(download_app, name="download")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"talhao {__version__}")
        raise typer.Exit()


@app.callback()
def _global_options(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", "-V", callback=_version_callback, is_eager=True, help="Exibe a versão."
        ),
    ] = None,
) -> None:
    """Talhão — Pipeline profissional de segmentação de instâncias."""


# download agrosmart
@download_app.command("agrosmart")
def download_agrosmart(
    tile_size: Annotated[int, typer.Option(help="Tamanho do tile em metros.")] = 50,
    overlap: Annotated[int, typer.Option(help="Sobreposição entre tiles em metros.")] = 10,
    min_coverage: Annotated[int, typer.Option(help="Cobertura mínima de anotação (%).")] = 30,
    width: Annotated[int, typer.Option(help="Largura do tile em pixels.")] = 1024,
    height: Annotated[int, typer.Option(help="Altura do tile em pixels.")] = 1024,
    max_workers: Annotated[int, typer.Option(help="Número de workers paralelos.")] = 10,
) -> None:
    """Baixa o dataset interno via AgroSmart SDK."""
    from talhao.data import AgroSmartDownloader

    downloader = AgroSmartDownloader(settings)
    final_dir = downloader.run(
        tile_size_meters=tile_size,
        overlap=overlap,
        min_annotation_coverage=min_coverage,
        width=width,
        height=height,
        max_workers=max_workers,
    )
    typer.echo(f"✅  Dataset AgroSmart pronto em: {final_dir}")


# download roboflow
@download_app.command("roboflow")
def download_roboflow() -> None:
    """Baixa o dataset validado do Roboflow."""
    from talhao.data import RoboflowDownloader

    downloader = RoboflowDownloader(settings)
    result_path = downloader.run()
    typer.echo(f"✅  Dataset Roboflow pronto em: {result_path}")


# split
@app.command()
def split(
    dataset_dir: Annotated[
        Path | None,
        typer.Option(
            "--dataset",
            "-d",
            help="Diretório do dataset a dividir. Padrão: settings.agrosmart_dataset_dir.",
        ),
    ] = None,
    classes: Annotated[
        str | None,
        typer.Option(help="Classes separadas por vírgula. Ex: 'talhao-virtual,cultivo'."),
    ] = None,
    train_ratio: Annotated[float, typer.Option(help="Proporção para treino.")] = 0.7,
    val_ratio: Annotated[float, typer.Option(help="Proporção para validação.")] = 0.15,
    test_ratio: Annotated[float, typer.Option(help="Proporção para teste.")] = 0.15,
    seed: Annotated[int, typer.Option(help="Seed aleatória.")] = 42,
) -> None:
    """Divide o dataset AgroSmart em treino / validação / teste."""
    from talhao.data import DatasetSplitter

    classes_list = [c.strip() for c in classes.split(",")] if classes else None

    splitter = DatasetSplitter(
        dataset_dir=dataset_dir,
        classes=classes_list,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=seed,
        settings=settings,
    )
    result = splitter.run()
    typer.echo(
        f"✅  Split concluído → "
        f"train={result.train_count} | val={result.val_count} | test={result.test_count} "
        f"({result.total_images} imagens, {result.total_captures} capturas)"
    )


# train
@app.command()
def train(
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset", "-d", help="Caminho para o diretório do dataset (com data.yaml)."
        ),
    ],
    name: Annotated[
        str, typer.Option("--name", "-n", help="Nome do experimento / pasta de saída.")
    ],
    model: Annotated[
        str, typer.Option("--model", "-m", help="Modelo YOLO (ex: yolo11n-seg.pt).")
    ] = "yolo11n-seg.pt",
    device: Annotated[str | None, typer.Option(help="Dispositivo: '0', 'cpu', '0,1', etc.")] = None,
    run_val: Annotated[
        bool, typer.Option("--eval/--no-eval", help="Rodar validação no conjunto val ao final.")
    ] = False,
    run_test: Annotated[
        bool, typer.Option("--test/--no-test", help="Rodar validação no conjunto test ao final.")
    ] = False,
) -> None:
    """Treina um modelo YOLO de segmentação."""
    from talhao.training import YOLOTrainer
    from talhao.training.trainer import TrainConfig

    trainer = YOLOTrainer(settings)
    config = TrainConfig(
        dataset_dir=dataset,
        run_name=name,
        model=model,
        device=device,
    )

    output_dir = trainer.train(config)

    if run_val:
        trainer.validate(output_dir, split="val")

    if run_test:
        trainer.validate(output_dir, split="test")

    typer.echo(f"✅  Treinamento concluído. Resultados em: {output_dir}")


# evaluate


@app.command()
def evaluate(
    run_dir: Annotated[
        Path, typer.Argument(help="Diretório do run de treinamento (contém results.csv).")
    ],
    model: Annotated[str, typer.Option(help="Nome do modelo (para relatório).")] = "",
    dataset: Annotated[str, typer.Option(help="Nome do dataset (para relatório).")] = "",
    output_dir: Annotated[Path, typer.Option(help="Diretório para salvar o relatório.")] = Path(
        "reports"
    ),
) -> None:
    """Extrai e exibe as métricas de um run de treinamento."""
    from talhao.training.evaluator import ModelEvaluator

    evaluator = ModelEvaluator()
    metrics = evaluator.extract(run_dir, model=model, dataset=dataset)
    evaluator.print_metrics(metrics)
    evaluator.export_comparison([metrics], output_dir=output_dir, report_name=run_dir.name)


# compare
@app.command()
def compare(
    run_dirs: Annotated[
        list[Path], typer.Argument(help="Dois ou mais diretórios de runs para comparar.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Diretório para salvar o relatório comparativo.")
    ] = Path("reports"),
    report_name: Annotated[
        str, typer.Option(help="Nome base do arquivo de relatório.")
    ] = "metrics_comparison",
) -> None:
    """Compara as métricas de dois ou mais runs e exporta um relatório."""
    from talhao.training.evaluator import ModelEvaluator

    if len(run_dirs) < 2:
        typer.echo("❌  Forneça ao menos 2 diretórios de runs para comparar.", err=True)
        raise typer.Exit(code=1)

    evaluator = ModelEvaluator()
    all_metrics = [evaluator.extract(d) for d in run_dirs]
    paths = evaluator.export_comparison(all_metrics, output_dir=output_dir, report_name=report_name)

    typer.echo(f"✅  Relatório JSON: {paths['json']}")
    typer.echo(f"✅  Relatório CSV:  {paths['csv']}")
