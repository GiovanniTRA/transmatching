import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from evaluation.utils import PROJECT_ROOT

PERF_ROOT = Path(PROJECT_ROOT / "evaluation" / "performance")
console = Console()

datasets = sorted(set(x.name for x in PERF_ROOT.iterdir() if x.is_dir()))

methods = sorted(
    set(
        x.name
        for dataset in datasets
        for x in (PERF_ROOT / dataset).iterdir()
        if x.is_dir()
    )
)


def retrieve_performance(path: Path):
    with (path / "metrics.json").open() as f:
        obj = json.load(f)
        try:
            return obj["global_mean_geo_error"]
        except KeyError:
            return obj["metrics"]["global_mean_geo_error"]


performance = defaultdict(dict)

for dataset in datasets:
    for method in methods:
        try:
            performance[dataset][method] = retrieve_performance(
                PERF_ROOT / dataset / method
            )
        except FileNotFoundError:
            console.print(f"<{method}> not evaluated on <{dataset}>", style="bold red")
            performance[dataset][method] = None


best_method = {
    x: min(
        performance[x],
        key=lambda y: performance[x][y] if performance[x][y] is not None else np.inf,
    )
    for x in performance
}
worse_method = {
    x: max(
        performance[x],
        key=lambda y: performance[x][y] if performance[x][y] is not None else -np.inf,
    )
    for x in performance
}


table = Table(title="Evaluation performance: [cyan]mean geo error[/cyan]")

table.add_column("", justify="right", style="blue", no_wrap=True)
for dataset in datasets:
    table.add_column(dataset, style="cyan")

for method in methods:
    row = [method]

    for dataset in datasets:
        if performance[dataset][method] is None:
            row.append("")
        elif method in worse_method[dataset]:
            row.append(f"[bold red]{performance[dataset][method]:.05f}[/bold red]")
        elif method in best_method[dataset]:
            row.append(f"[bold green]{performance[dataset][method]:.05f}[/bold green]")
        else:
            row.append(f"{performance[dataset][method]:.05f}")
    table.add_row(*row)

console.print(table)
