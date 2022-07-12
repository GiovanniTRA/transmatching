import json
from pathlib import Path

import meshio
import numpy as np
from pytorch_lightning import seed_everything
from tqdm import tqdm

FAUST_PATH = Path("/run/media/luca/LocalDisk/Datasets/MPI-FAUST/training/registrations")
assert FAUST_PATH.exists(), "Do not regenerate! Download from Drive or DVC."

N_PAIRS = 100

seed_everything(0)

shapes_paths = list(FAUST_PATH.glob("*.ply"))
n_shapes = len(shapes_paths)

for i in tqdm(range(N_PAIRS)):

    shape_A_idx = np.random.randint(n_shapes)

    shape_B_idx = np.random.randint(n_shapes)
    while shape_A_idx == shape_B_idx:  # avoid taking A exactly equal to B
        shape_B_idx = np.random.randint(n_shapes)

    shape_A_path = shapes_paths[shape_A_idx]
    shape_B_path = shapes_paths[shape_B_idx]

    sample_path = Path(f"data/{i:03}")
    assert (
        not sample_path.exists()
    ), "Folder already exists! Inconsistency risk, delete <data> and regenerate from scratch."
    sample_path.mkdir(parents=True, exist_ok=True)

    shape_A = meshio.read(shape_A_path)
    shape_B = meshio.read(shape_B_path)
    gt_matching_A_to_B = np.arange(shape_B.points.shape[0], dtype=np.int64)

    meshio.write(filename=str(sample_path / f"A.off"), mesh=shape_A)
    meshio.write(filename=str(sample_path / f"B.off"), mesh=shape_B)
    np.savetxt(
        fname=str(sample_path / f"gt_matching_A_to_B.txt"),
        X=gt_matching_A_to_B,
        fmt="%d",
    )
    with (sample_path / "meta.json").open("w") as f:
        json.dump({"id": {"A": shape_A_path.name, "B": shape_B_path.name}}, f, indent=4)
