import json
from pathlib import Path

import meshio
import numpy as np
from meshio import Mesh
from pytorch_lightning import seed_everything
from tqdm import tqdm
from scipy import io

from evaluation.utils import PROJECT_ROOT


N_PAIRS = 100
FAUST_0NOISE = Path(
    PROJECT_ROOT / "evaluation/datasets/faust_1k_0noise/FAUST_noise_0.00.mat"
)

seed_everything(0)

shapes = io.loadmat(str(FAUST_0NOISE))

n_shapes = shapes["vertices"].shape[0]


for i in tqdm(range(N_PAIRS)):
    shape_A_idx = np.random.randint(n_shapes)

    shape_B_idx = np.random.randint(n_shapes)
    while shape_A_idx == shape_B_idx:  # avoid taking A exactly equal to B
        shape_B_idx = np.random.randint(n_shapes)

    sample_path = PROJECT_ROOT / "evaluation/datasets/faust_1k_0noise" / f"data/{i:03}"
    assert (
        not sample_path.exists()
    ), "Folder already exists! Inconsistency risk, delete <data> and regenerate from scratch."
    sample_path.mkdir(parents=True, exist_ok=True)

    shape_A = Mesh(
        points=shapes["vertices"][shape_A_idx],
        cells=[("triangle", shapes["faces"] - 1)],
    )
    shape_B = Mesh(
        points=shapes["vertices"][shape_B_idx],
        cells=[("triangle", shapes["faces"] - 1)],
    )

    gt_matching_A_to_B = np.arange(
        shapes["vertices"][shape_A_idx].shape[0], dtype=np.int64
    )

    meshio.write(filename=str(sample_path / f"A.off"), mesh=shape_A)
    meshio.write(filename=str(sample_path / f"B.off"), mesh=shape_B)
    np.savetxt(
        fname=str(sample_path / f"gt_matching_A_to_B.txt"),
        X=gt_matching_A_to_B,
        fmt="%d",
    )
    with (sample_path / "meta.json").open("w") as f:
        json.dump({"id": {"A": shape_A_idx, "B": shape_B_idx}}, f, indent=4)
