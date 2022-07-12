from typing import Dict

import meshio
import numpy as np
import scipy.io
import torch
from transmatching.Model.model import Model
from transmatching.Utils.refine import refine, refine_hires

from evaluation.competitors.eval_dataset import EvalDataset
from evaluation.competitors.eval_model import ModelMatching
from evaluation.competitors.shape_normalization import (
    area_weighted_denormalization,
    area_weighted_normalization,
    naive_normalization,
    normalization_wrt_lowres_mesh,
)
from evaluation.utils import (
    PROJECT_ROOT,
    Mesh,
    chamfer_chunked,
    keops_cdist_argmin,
    plot_meshes,
)

# checkpoint = "data_aug_1_axis"
# checkpoint = "shape2template_area"
# checkpoint = "best_shape2template_area"
# checkpoint = "s2t_new"
# checkpoint = "s2t_new"

checkpoint_file = "s2s_weighted_bary"
# checkpoint_file = "s2s_noarea_2"
# checkpoint_file = "continue_best_s2s_bary"
# checkpoint_file = "best_fine_tune_best_s2s"

# checkpoint = "best_shape2shape_augRotation_Luca_weighted5"

CHECKPOINTS_ROOT = PROJECT_ROOT / "evaluation" / "competitors" / "our" / "checkpoints"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class OurMatching(ModelMatching):
    def __init__(
        self,
        checkpoint_name: str = checkpoint_file,
        refine: bool = False,
        area_normalization: bool = True,
        device=DEVICE,
        refine_steps: int = 100,
    ) -> None:
        self.refine_steps = refine_steps
        self.checkpoint_name = checkpoint_name
        # Loading Models
        self.refine = refine
        self.area_normalization = (
            checkpoint_name == "s2s_weighted_bary"
            if area_normalization is None
            else area_normalization
        )
        super(OurMatching, self).__init__()

        self.device = device
        self.model = Model(
            d_bottleneck=32,
            d_latent=64,
            d_channels=64,
            d_middle=512,
            N=8,
            heads=4,
            max_seq_len=100,
            d_origin=3,
            dropout=0,
        )
        self.model.load_state_dict(
            torch.load(CHECKPOINTS_ROOT / self.checkpoint_name, map_location=device)
        )
        self.model.eval()
        self.model = self.model.to(device)

        self.faust_1k = scipy.io.loadmat(
            PROJECT_ROOT / "evaluation/data_aux/FAUST_noise_0.00.mat"
        )["vertices"]

    def get_name(self) -> str:
        if self.refine:
            return f"our(ckp={self.checkpoint_name},refine={self.refine},area_norm={self.area_normalization},refine_steps={self.refine_steps})"
        else:
            return f"our(ckp={self.checkpoint_name},refine={self.refine},area_norm={self.area_normalization})"

    def get_simple_name(self) -> str:
        return "our"

    def shape2shape(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        assert "s2t" not in sample["name"]

        points_A = sample["points_A"]
        faces_A = sample["faces_A"]

        points_B = sample["points_B"]
        faces_B = sample["faces_B"]

        points_A = torch.from_numpy(points_A).float()
        points_B = torch.from_numpy(points_B).float()

        if self.area_normalization:
            if "noise" in sample["name"]:
                points_A = area_weighted_normalization(points_A, rescale=False)
                points_B = area_weighted_normalization(points_B, rescale=False)
            else:
                points_A = area_weighted_normalization(points_A)
                points_B = area_weighted_normalization(points_B)

        else:
            if sample["name"] in {"faust", "faust_permuted"}:
                points_A_1k = self.faust_1k[int(sample["id_A"][-7:-4])]
                points_B_1k = self.faust_1k[int(sample["id_B"][-7:-4])]

                points_A = normalization_wrt_lowres_mesh(points_A, points_A_1k)
                points_B = normalization_wrt_lowres_mesh(points_B, points_B_1k)

            else:
                points_A = naive_normalization(points_A)
                points_B = naive_normalization(points_B)

        points_A = points_A[None, ...].to(self.device)
        points_B = points_B[None, ...].to(self.device)

        with torch.no_grad():
            A_hat = self.model(points_A, points_B)
            B_hat = self.model(points_B, points_A)

            A_chamfer = chamfer_chunked(A_hat.squeeze(), points_A.squeeze())
            B_chamfer = chamfer_chunked(B_hat.squeeze(), points_B.squeeze())

        if A_chamfer < B_chamfer:
            if self.refine:
                A_hat, _ = refine_hires(
                    self.model, points_A, points_B, self.refine_steps
                )
                A_hat = A_hat[-1]
            A_hat = A_hat.to("cpu")
            points_A = points_A[None, ...].to("cpu")

            pred_matching_A_to_B = (
                keops_cdist_argmin(points_A.squeeze(), A_hat.squeeze(), argmin_axis=1)
                .cpu()
                .squeeze()
            )
            registration = {
                "registration_A_to_B": meshio.Mesh(
                    points=A_hat.cpu().detach().squeeze().numpy(),
                    cells=[("triangle", faces_B)],
                )
            }
        else:
            if self.refine:
                B_hat, _ = refine_hires(
                    self.model, points_B, points_A, self.refine_steps
                )
                B_hat = B_hat[-1]
            B_hat = B_hat.to("cpu")
            points_B = points_B[None, ...].to("cpu")

            pred_matching_A_to_B = (
                keops_cdist_argmin(points_B.squeeze(), B_hat.squeeze(), argmin_axis=0)
                .cpu()
                .squeeze()
            )
            B_hat = B_hat.detach().cpu().squeeze()
            B_hat = area_weighted_denormalization(
                B_hat, torch.from_numpy(sample["points_B"])
            )
            registration = {
                "registration_A_to_B": meshio.Mesh(
                    points=B_hat.cpu().detach().squeeze().numpy(),
                    cells=[("triangle", faces_A)],
                )
            }

        return {"pred_matching_A_to_B": pred_matching_A_to_B, **registration}

    def shape2template(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        assert "s2t" in sample["name"]

        points_A = sample["points_A"]
        faces_A = sample["faces_A"]

        points_B = sample["points_B"]
        faces_B = sample["faces_B"]

        points_A = torch.from_numpy(points_A).float()
        points_B = torch.from_numpy(points_B).float()

        if self.area_normalization:
            if "noise" in sample["name"]:
                points_A = area_weighted_normalization(points_A, rescale=False)
                points_B = area_weighted_normalization(points_B, rescale=False)
            else:
                points_A = area_weighted_normalization(points_A, rescale=False)
                points_B = area_weighted_normalization(points_B)
        else:
            points_A = naive_normalization(points_A, rescale=False)
            if sample["name"] in {"faust", "faust_permuted", "faust_s2t"}:
                points_B_1k = self.faust_1k[int(sample["id_B"][-7:-4])]

                points_B = points_B.numpy()
                points_B = normalization_wrt_lowres_mesh(points_B, points_B_1k)
                points_B = torch.from_numpy(points_B).float()
            else:
                points_B = naive_normalization(points_B)

        points_A = points_A[None, ...].to(self.device)
        points_B = points_B[None, ...].to(self.device)

        with torch.no_grad():
            B_hat = self.model(points_B, points_A)

        if self.refine:
            B_hat, _ = refine(self.model, points_B, points_A, self.refine_steps)
            B_hat = B_hat[-1]
        B_hat = B_hat.to("cpu")
        points_B = points_B.to("cpu")
        pred_matching_A_to_B = torch.cdist(points_B, B_hat).squeeze().argmin(0)

        B_hat = B_hat.detach().cpu().squeeze()
        B_hat = area_weighted_denormalization(
            B_hat, torch.from_numpy(sample["points_B"])
        )
        return {
            "pred_matching_A_to_B": pred_matching_A_to_B,
            "registration_A_to_B": meshio.Mesh(
                points=B_hat.cpu().detach().numpy(),
                cells=[("triangle", faces_A)],
            ),
        }

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "s2t" in sample["name"]:
            return self.shape2template(sample)
        else:
            return self.shape2shape(sample)


if __name__ == "__main__":
    dataset = EvalDataset("faust_1k_outliers")
    sample = dataset[0]
    print(sample["id_A"])
    print(sample["id_B"])
    print(sample["points_A"].shape[0] + sample["points_B"].shape[0])

    model = OurMatching()
    out = model(sample)

    plot_meshes(
        meshes=[
            Mesh(
                v=sample["points_A"],
                f=None,
            ),
            Mesh(
                v=sample["points_B"],
                f=None,
            ),
            Mesh(
                v=out["registration_A_to_B"].points,
                f=None,
            ),
        ],
        titles=["A", "B", "REG"],
        showtriangles=[False, False, False],
        showscales=None,
        autoshow=False,
    ).show()

    print("OK")

#     dataset = EvalDataset("faust_1k_noise")
#     model = OurMatching()
#     for i in dataset:
#         model(i)
#         break
#     print("OK")
# #
# plot_meshes(
#     meshes=[
#         Mesh(
#             v=points_A.detach().squeeze().numpy(),
#             f=None,
#             color=points_B[0, pred_matching_A_to_B, 0]
#         ),
#         Mesh(
#             v=points_B.detach().squeeze().numpy(),
#             f=None,
#             color=points_B[0, :, 0]
#         ),
#     ],
#     titles=["A", "B"],
#     showtriangles=[False, False],
#     showscales=None,
#     autoshow=False,
# ).show()
