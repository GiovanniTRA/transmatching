import itertools
from typing import Dict, Union

import hydra
import meshio
import numpy as np
import omegaconf
from plotly import graph_objects as go
from pytorch_lightning import seed_everything
from tqdm import tqdm

from evaluation.competitors.eval_dataset import EvalDataset
from evaluation.utils import PROJECT_ROOT


def plot3dmesh(vertices: np.ndarray, faces: np.ndarray, color) -> None:
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=1,
                intensity=color,
                showscale=False,
            )
        ]
    )
    fig.show()


def store_prediction(
    cfg: omegaconf.DictConfig,
    model_name: str,
    dataset_name: str,
    sample_idx: int,
    sample: Dict[str, np.ndarray],
    prediction: Dict[str, Union[np.ndarray, meshio.Mesh]],
) -> None:
    if "s2t" in sample["name"] and "registration_A_to_B" not in prediction:
        raise ValueError(
            f"Shape2Template setting but {model_name} did not return a registration!"
        )

    if "export_shrec" in cfg and cfg["export_shrec"]:
        prediction_folder = (
            PROJECT_ROOT
            / "evaluation"
            / "predictions"
            / dataset_name
            / model_name
            / f"export"
        )
        prediction_folder.mkdir(parents=True, exist_ok=cfg.overwrite_predictions)
        np.savetxt(
            fname=str(prediction_folder / f"{sample['id_A']}_{sample['id_B']}.txt"),
            X=prediction["pred_matching_A_to_B"] + 1,  # matlab wants one!
            fmt="%d",
        )
    else:
        prediction_folder = (
            PROJECT_ROOT
            / "evaluation"
            / "predictions"
            / dataset_name
            / model_name
            / f"{sample_idx:03}"
        )
        prediction_folder.mkdir(parents=True, exist_ok=cfg.overwrite_predictions)
        np.savetxt(
            fname=str(prediction_folder / f"gt_matching_A_to_B.txt"),
            X=sample["gt_matching_A_to_B"],
            fmt="%d",
        )
        np.savetxt(
            fname=str(prediction_folder / f"pred_matching_A_to_B.txt"),
            X=prediction["pred_matching_A_to_B"],
            fmt="%d",
        )

        # if "s2t" in sample["name"]:
        meshio.write(
            str(prediction_folder / "registration_A_to_B.off"),
            prediction["registration_A_to_B"],
        )


@hydra.main(
    config_path=str(PROJECT_ROOT / "evaluation" / "conf"), config_name="default"
)
def run(cfg: omegaconf.DictConfig):
    seed_everything(0)

    dataset = EvalDataset(cfg["dataset"])
    model = hydra.utils.instantiate(cfg.model, device=cfg.device)

    if "limit" in cfg and cfg["limit"]:
        iter_dataset = itertools.islice(dataset, cfg["limit"])
    else:
        iter_dataset = dataset

    desc = f"Running <{model.name}> on <{dataset.name}>"
    for sample in tqdm(iter_dataset, desc=desc):
        sample_idx = sample["item"]
        prediction = model(sample)
        store_prediction(
            cfg=cfg,
            model_name=model.name,
            dataset_name=dataset.name,
            sample_idx=sample_idx,
            sample=sample,
            prediction=prediction,
        )

        # Test the pull-back color transfer, from B to A using a matching A -> B
        # plot3dmesh(
        #     sample["points_A"],
        #     sample["faces_A"],
        #     sample["points_B"][prediction["pred_matching_A_to_B"], 0],
        # )
        # plot3dmesh(
        #     sample["points_B"],
        #     sample["faces_B"],
        #     sample["points_B"][:, 0],
        # )
        # break


if __name__ == "__main__":
    run()
