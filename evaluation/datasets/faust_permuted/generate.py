import json
from pathlib import Path

import meshio
import numpy as np
from meshio import Mesh
from plotly import graph_objects as go
from pytorch_lightning import seed_everything
from tqdm import tqdm

from evaluation.utils import PROJECT_ROOT

FAUST_PATH = Path("/run/media/luca/LocalDisk/Datasets/MPI-FAUST/training/registrations")
assert FAUST_PATH.exists(), "Do not regenerate! Download from Drive or DVC."

N_PAIRS = 100

seed_everything(0)

shapes_paths = list(FAUST_PATH.glob("*.ply"))
n_shapes = len(shapes_paths)


def invert_permutation(p: np.ndarray) -> np.ndarray:
    """The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    """
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def plot3dmesh(vertices: np.ndarray, faces: np.ndarray) -> None:
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
                intensity=vertices[:, 0],
                showscale=False,
            )
        ]
    )
    fig.show()


for i in tqdm(range(N_PAIRS)):

    with (
        PROJECT_ROOT
        / "evaluation"
        / "datasets"
        / "faust"
        / "data"
        / f"{i:03d}"
        / "meta.json"
    ).open() as f:
        sample_meta = json.load(f)

    shape_A_path = FAUST_PATH / sample_meta["id"]["A"]
    shape_B_path = FAUST_PATH / sample_meta["id"]["B"]

    sample_path = Path(__file__).parent / f"data/{i:03d}"
    assert (
        not sample_path.exists()
    ), "Folder already exists! Inconsistency risk, delete <data> and regenerate from scratch."
    sample_path.mkdir(parents=True, exist_ok=True)

    shape_A = meshio.read(shape_A_path)
    shape_B = meshio.read(shape_B_path)

    one_to_n = np.arange(shape_A.points.shape[0], dtype=np.int32)

    permutation = np.random.permutation(shape_A.points.shape[0])
    gt_matching_A_to_B = invert_permutation(permutation)

    # apply permutation to mesh, points and faces
    shape_B_points = shape_B.points[permutation, :]
    shape_B_faces = invert_permutation(permutation)[shape_B.cells_dict["triangle"]]

    meshio.write(filename=str(sample_path / f"A.off"), mesh=shape_A)
    meshio.write(
        filename=str(sample_path / f"B.off"),
        mesh=Mesh(shape_B_points, [("triangle", shape_B_faces)]),
    )
    np.savetxt(
        fname=str(sample_path / f"gt_matching_A_to_B.txt"),
        X=gt_matching_A_to_B,
        fmt="%d",
    )
    with (sample_path / "meta.json").open("w") as f:
        json.dump({"id": {"A": shape_A_path.name, "B": shape_B_path.name}}, f, indent=4)
