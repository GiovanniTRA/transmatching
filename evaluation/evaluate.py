import itertools
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import hydra
import igl
import meshio
import numpy as np
import omegaconf
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from tqdm import tqdm

from evaluation.competitors.eval_dataset import EvalDataset
from evaluation.utils import (
    PROJECT_ROOT,
    Mesh,
    convert_colors,
    get_point_colors,
    plot_geo_errors,
    plot_mesh,
    plot_meshes,
)


def approximate_geodesic_distances(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute the geodesic distances approximated by the dijkstra method weighted by
    euclidean edge length

    Args:
        v: the mesh points
        f: the mesh faces

    Returns:
        an nxn matrix which contains the approximated distances
    """
    # todo vedi dentro igl se si possono fare esatte. "exact geodesics" method.
    a = igl.adjacency_matrix(f)
    dist = cdist(v, v)
    values = dist[np.nonzero(a)]
    matrix = sparse.coo_matrix((values, np.nonzero(a)), shape=(v.shape[0], v.shape[0]))
    d = dijkstra(matrix, directed=False)
    return d


def get_geodesic_errors(
    pred_matching_A_to_B: np.ndarray,
    gt_matching_A_to_B: np.ndarray,
    points_B: np.ndarray,
    faces_B: np.ndarray,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Compute the matching geodesic errors on the shape B, given the predicted and
    ground truth matching A -> B. The geodesics distances on the shape B are
    approximated using dijkstra considering the euclidean length of each edge.

    The two matching `x` are such that the index `i` refers to the shape A,
    the corresponding value `x[i]` refers to the shape B

    Args:
        pred_matching_A_to_B: the predicted matching
        gt_matching_A_to_B: the ground truth matching
        points_B: the points of the shape B
        faces_B: the faces of the shape B

    Returns:
        a array that indices the prediction geodesic error for each point of the shape A
    """
    cache_file = cache_path / "B_geo_dists.npy" if cache_path is not None else None

    geo_dists = None
    if cache_file is not None and cache_file.exists():
        geo_dists = np.load(str(cache_file))

    if geo_dists is None:
        geo_dists = approximate_geodesic_distances(points_B, faces_B)
        if cache_path is not None:
            cache_path.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_file), geo_dists)

    geo_dists /= np.max(geo_dists)
    geo_errors = geo_dists[pred_matching_A_to_B, gt_matching_A_to_B]
    return geo_errors


def get_registration_errors(
    registration_A_to_B: np.ndarray,
    points_B: np.ndarray,
    gt_matching_A_to_B: np.ndarray,
) -> Dict[str, float]:
    pointwise_distance = np.sqrt(
        ((registration_A_to_B - points_B[gt_matching_A_to_B]) ** 2).sum(-1)
    )

    max_eu = pointwise_distance.max()
    mean_eu = pointwise_distance.mean()

    dist = cdist(registration_A_to_B, points_B)
    chamfer_dist = (dist.min(-2).mean(-1) + dist.min(-1).mean(-1)) / 2

    return {
        "max_eu": max_eu,
        "mean_eu": mean_eu,
        "chamfer": chamfer_dist,
    }


def load_predictions(
    dataset_name: str, sample_idx: int, model_name: str
) -> Dict[str, np.ndarray]:
    """
    Load the pre-computed predictions

    Args:
        dataset_name: the name of the dataset to consider
        sample_idx:  the sample index inside the dataset to consider
        model_name: the name of the model to consider

    Returns:
        the computed predictions, i.e. the matching from A to B

    """
    prediction_folder = (
        PROJECT_ROOT
        / "evaluation"
        / "predictions"
        / dataset_name
        / model_name
        / f"{sample_idx:03}"
    )
    assert (
        prediction_folder.exists()
    ), f"Prediction folder does not exists: <{prediction_folder}>"

    pred_matching_A_to_B = np.loadtxt(
        str(prediction_folder / f"pred_matching_A_to_B.txt"), dtype=np.int32
    )

    out = {"pred_matching_A_to_B": pred_matching_A_to_B}

    registration_A_to_B_path = prediction_folder / "registration_A_to_B.off"
    if registration_A_to_B_path.exists():
        out["registration_A_to_B"] = meshio.read(registration_A_to_B_path)
    return out


def compute_and_store_pair_metrics(
    sample_path: Path,
    cache_path: Path,
    sample: Dict[str, np.ndarray],
    predictions: Dict[str, np.ndarray],
    model_name: str,
) -> Dict[str, np.ndarray]:
    """
    Given a pair sample and the corresponding prediction, compute the metrics to evaluate
    the performance.

    Args:
        sample: the sample being evaluated
        predictions: the predictions over that sample

    Returns:
        a dictionary containing the computed pair-metrics.
    """
    points_A = sample["points_A"]
    points_B = sample["points_B"]
    gt_matching_A_to_B = sample["gt_matching_A_to_B"]
    pred_matching_A_to_B = predictions["pred_matching_A_to_B"]

    if "registration_A_to_B" in predictions:
        meshio.write(
            str(sample_path / "registration_A_to_B.off"),
            predictions["registration_A_to_B"],
        )

        color = get_point_colors(
            points_B,
            frequency=np.pi,
        )
        shape_A_color_transfer = plot_meshes(
            meshes=[
                Mesh(
                    v=points_A,
                    f=None,
                    color=convert_colors(color[pred_matching_A_to_B]),
                ),
                Mesh(
                    v=predictions["registration_A_to_B"].points,
                    f=None,
                    color=convert_colors(color),
                ),
                Mesh(
                    v=points_B,
                    f=None,
                    color=convert_colors(color),
                ),
            ],
            titles=["Shape A", "A registration to B", "Shape B"],
            showtriangles=[False, False, False],
            showscales=None,
            autoshow=False,
        )
        shape_A_color_transfer.write_html(str(sample_path / "registration_A_to_B.html"))

        registration_metrics = get_registration_errors(
            registration_A_to_B=predictions["registration_A_to_B"].points,
            points_B=points_B,
            gt_matching_A_to_B=gt_matching_A_to_B,
        )
    # pulled-back errors! The errors are defined over the shape A
    geo_errors = get_geodesic_errors(
        pred_matching_A_to_B=pred_matching_A_to_B,
        gt_matching_A_to_B=gt_matching_A_to_B,
        points_B=points_B,
        faces_B=sample["faces_B"],
        cache_path=cache_path,
    )
    np.savetxt(
        fname=str(sample_path / f"geo_errors.txt"),
        X=geo_errors,
    )

    # errors on B pull-back visualization on shape A
    shape_A_errors = plot_mesh(
        Mesh(
            v=points_A,
            f=None,
            color=geo_errors,
        ),
        colorscale="hot",
        reversescale=True,
        cmin=0,
        cmax=0.25,
    )
    shape_A_errors.write_html(str(sample_path / "geo_errors_on_A.html"))

    # color pull-back transfer from B to A
    color = get_point_colors(
        points_B,
        frequency=np.pi,
    )

    shape_A_color_transfer = plot_meshes(
        meshes=[
            Mesh(
                v=points_A,
                f=None,
                color=convert_colors(color[pred_matching_A_to_B]),
            ),
            Mesh(
                v=points_B,
                f=None,
                color=convert_colors(color),
            ),
        ],
        titles=["Shape A", "Shape B"],
        showtriangles=[False, False],
        showscales=None,
        autoshow=False,
    )
    shape_A_color_transfer.write_html(str(sample_path / "transfer_on_A.html"))

    # cumulative error plot
    error_plot = plot_geo_errors(geo_errors)
    error_plot.savefig(sample_path / "cumulative_geo_errors.png")
    plt.close(error_plot)

    # save pair metrics
    metrics = {
        "item": sample["item"],
        "metrics": {
            "mean_geo_error": geo_errors.mean(),
        },
    }
    if "registration_A_to_B" in predictions:
        metrics["metrics"].update(registration_metrics)

    with (sample_path / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

    return {"model_name": model_name, "geo_errors": geo_errors, **metrics}


def aggregate_and_store_pair_metrics(
    global_metrics_path: Path, metrics: Sequence[Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, Union[int, np.ndarray]]]:
    """
    Given a list of pair-metrics, aggregate what's needed to compute global metrics.
    Pair-wise metrics that needs to be stored in their own folder are returned as-is

    Args:
        metrics: the list of computed pair-metrics

    Returns:
        the aggregated metrics and the pair-metrics that must be stored

    """
    errors_cat = np.concatenate([x["geo_errors"] for x in metrics])
    np.savetxt(
        fname=str(global_metrics_path / f"global_geo_errors.txt"),
        X=errors_cat,
    )

    # cumulative error plot
    error_plot = plot_geo_errors(errors_cat)
    error_plot.savefig(global_metrics_path / "mean_cumulative_geo_errors.png")
    plt.close(error_plot)

    metric_names = list(metrics[0]["metrics"].keys())
    aggregate_metrics = {
        f"global_{metric_name}": np.mean(
            [sample_metric["metrics"][metric_name] for sample_metric in metrics]
        )
        for metric_name in metric_names
    }

    global_metrics = {
        "model_name": metrics[0]["model_name"],
        "number_sample": len(metrics),
        "metrics": aggregate_metrics,
    }
    with (global_metrics_path / "metrics.json").open("w") as f:
        json.dump(global_metrics, f, indent=4, sort_keys=True)


@hydra.main(
    config_path=str(PROJECT_ROOT / "evaluation" / "conf"), config_name="default"
)
def run(cfg: omegaconf.DictConfig):
    seed_everything(0)

    dataset = EvalDataset(cfg["dataset"])
    model = hydra.utils.instantiate(cfg.model)

    assert model.name, "The model must have a name!"
    assert dataset.name, "The dataset must have a name!"

    global_metrics_path = (
        PROJECT_ROOT / "evaluation" / "performance" / dataset.name / model.name
    )
    global_metrics_path.mkdir(parents=True, exist_ok=cfg.overwrite_predictions)

    pair_metrics = []
    desc = f"Evaluating <{model.name}> on <{dataset.name}>"

    if "limit" in cfg and cfg["limit"]:
        iter_dataset = itertools.islice(dataset, cfg["limit"])
    else:
        iter_dataset = dataset

    for sample in tqdm(iter_dataset, desc=desc):
        sample_idx = sample["item"]

        sample_path = global_metrics_path / "data" / f"{sample_idx:03}"
        sample_path.mkdir(parents=True, exist_ok=cfg.overwrite_predictions)

        predictions = load_predictions(
            dataset_name=dataset.name,
            model_name=model.name,
            sample_idx=sample_idx,
        )

        pair_metrics.append(
            compute_and_store_pair_metrics(
                sample_path=sample_path,
                cache_path=PROJECT_ROOT
                / "evaluation"
                / ".cache"
                / dataset.name
                / "data"
                / f"{sample_idx:03}",
                sample=sample,
                predictions=predictions,
                model_name=model.name,
            )
        )

    aggregate_and_store_pair_metrics(
        global_metrics_path=global_metrics_path, metrics=pair_metrics
    )


if __name__ == "__main__":
    run()
