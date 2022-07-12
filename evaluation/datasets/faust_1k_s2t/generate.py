import json
from pathlib import Path

import meshio
import numpy as np
from pytorch_lightning import seed_everything
from tqdm import tqdm
from scipy import io

from evaluation.utils import Mesh
from evaluation.utils import PROJECT_ROOT
from evaluation.utils import plot_meshes

N_PAIRS = 100
FAUST_REM = Path(PROJECT_ROOT / "evaluation/datasets/faust_1k/FAUSTS_rem.mat")
TEMPLATE_PATH = Path(PROJECT_ROOT / "evaluation/datasets/faust_1k_s2t/12ktemplate.ply")
template = meshio.read(TEMPLATE_PATH)

seed_everything(0)

shapes = io.loadmat(str(FAUST_REM))

n_shapes = shapes["vertices"].shape[0]

for i in tqdm(range(n_shapes)):

    sample_path = PROJECT_ROOT / "evaluation/datasets/faust_1k_s2t" / f"data/{i:03}"
    assert (
        not sample_path.exists()
    ), "Folder already exists! Inconsistency risk, delete <data> and regenerate from scratch."
    sample_path.mkdir(parents=True, exist_ok=True)

    shape_A = template
    shape_B = meshio.Mesh(
        points=shapes["vertices"][i], cells=[("triangle", shapes["f"] - 1)]
    )

    gt_matching_A_to_B = np.arange(shape_A.points.shape[0], dtype=np.int64)

    meshio.write(filename=str(sample_path / f"A.off"), mesh=shape_A)
    meshio.write(filename=str(sample_path / f"B.off"), mesh=shape_B)
    np.savetxt(
        fname=str(sample_path / f"gt_matching_A_to_B.txt"),
        X=gt_matching_A_to_B,
        fmt="%d",
    )
    with (sample_path / "meta.json").open("w") as f:
        json.dump({"id": {"A": "12ktemplate.ply", "B": i}}, f, indent=4)

# plot_meshes(
#     meshes=[
#         Mesh(v=template.points, f=None, color=shape_B.points[:, 0]),
#         Mesh(v=shape_B.points, f=None, color=shape_B.points[:, 0]),
#     ],
#     titles=["A", "B"],
#     showtriangles=[False, False],
#     showscales=None,
#     autoshow=False,
# ).show()
