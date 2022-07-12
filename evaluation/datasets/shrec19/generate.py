import json
from pathlib import Path

import meshio
import numpy as np
import scipy
from pytorch_lightning import seed_everything
from tqdm import tqdm

from plotly import graph_objects as go
from meshio import Mesh
from evaluation.utils import PROJECT_ROOT
from scipy import io

seed_everything(0)

SHREC_PATH = Path(PROJECT_ROOT / "evaluation/datasets/shrec19/origin_shrec/mat")
SHREC_PAIRS = (
    PROJECT_ROOT
    / "evaluation/datasets/shrec19/SHREC19_matching_humans/PAIRS_list_SHREC19_connectivity.txt"
)

with SHREC_PAIRS.open("r") as f:
    lines = f.read().splitlines()
    couples = [list(map(int, line.split(","))) for line in lines]


for i, (shape_idx1, shape_idx2) in tqdm(enumerate(couples)):
    sample_path = Path(__file__).parent / f"data/{i:03d}"
    assert (
        not sample_path.exists()
    ), "Folder already exists! Inconsistency risk, delete <data> and regenerate from scratch."
    sample_path.mkdir(parents=True, exist_ok=True)

    shape_1_mat = scipy.io.loadmat(str(SHREC_PATH / f"{shape_idx1}.mat"))
    shape_2_mat = scipy.io.loadmat(str(SHREC_PATH / f"{shape_idx2}.mat"))

    shape1 = shape_1_mat["M"][0][0][0]
    shape2 = shape_2_mat["M"][0][0][0]

    faces_1 = shape_1_mat["M"][0][0][1].astype("long") - 1
    faces_2 = shape_2_mat["M"][0][0][1].astype("long") - 1

    meshio.write(
        filename=str(sample_path / f"A.off"),
        mesh=Mesh(shape1, [("triangle", faces_1)]),
    )

    meshio.write(
        filename=str(sample_path / f"B.off"),
        mesh=Mesh(shape2, [("triangle", faces_2)]),
    )

    with (sample_path / "meta.json").open("w") as f:
        json.dump({"id": {"A": shape_idx1, "B": shape_idx2}}, f, indent=4)
