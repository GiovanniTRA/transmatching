import json
from pathlib import Path

import meshio
import numpy as np
from pytorch_lightning import seed_everything
from tqdm import tqdm

from evaluation.utils import Mesh
from evaluation.utils import PROJECT_ROOT
from evaluation.utils import invert_permutation
from evaluation.utils import plot_meshes
from scipy.io import loadmat

FAUST_PATH = Path("/run/media/luca/LocalDisk/Datasets/MPI-FAUST/training/registrations")
assert FAUST_PATH.exists(), "Do not regenerate! Download from Drive or DVC."

TEMPLATE_PATH = Path(PROJECT_ROOT / "evaluation/datasets/faust_1k_s2t/12ktemplate.ply")
template = meshio.read(TEMPLATE_PATH)

template_to_faust_matchings = (
    PROJECT_ROOT / "evaluation/datasets/faust_s2t/1k_to_faust_matchings.mat"
)
template_idxs = loadmat(template_to_faust_matchings)["idx_rem"].squeeze() - 1
seed_everything(0)

shapes_paths = list(FAUST_PATH.glob("*.ply"))
n_shapes = len(shapes_paths)

for i in tqdm(range(n_shapes)):
    sample_path = PROJECT_ROOT / "evaluation/datasets/faust_s2t" / f"data/{i:03}"
    assert (
        not sample_path.exists()
    ), "Folder already exists! Inconsistency risk, delete <data> and regenerate from scratch."
    sample_path.mkdir(parents=True, exist_ok=True)

    shape_A = template
    shape_B_path = shapes_paths[i]
    shape_B = meshio.read(shape_B_path)
    gt_matching_A_to_B = template_idxs

    meshio.write(filename=str(sample_path / f"A.off"), mesh=shape_A)
    meshio.write(filename=str(sample_path / f"B.off"), mesh=shape_B)
    np.savetxt(
        fname=str(sample_path / f"gt_matching_A_to_B.txt"),
        X=gt_matching_A_to_B,
        fmt="%d",
    )
    with (sample_path / "meta.json").open("w") as f:
        json.dump({"id": {"A": "12ktemplate.ply", "B": shape_B_path.name}}, f, indent=4)


# plot_meshes(
#     meshes=[
#         Mesh(v=shape_A.points, f=None, color=shape_A.points[:, 0]),
#         Mesh(v=shape_B.points, f=None, color=shape_A.points[template_idxs, 0]),
#     ],
#     titles=["A", "B"],
#     showtriangles=[False, False],
#     showscales=None,
#     autoshow=False,
# ).show()
