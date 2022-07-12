import json
from typing import Dict

import meshio
import numpy as np
from torch.utils.data import Dataset

from evaluation.utils import PROJECT_ROOT


class EvalDataset(Dataset):
    def __init__(self, dataset_name: str):
        """
        A generic dataset that is able to read every dataset that follows the structure
        of an evaluation dataset
        Args:
            dataset_name: name of an evaluation dataset, a folder under
                          `evaluation/datasets` with this name must exist.
        """
        super().__init__()
        self.name = dataset_name
        self.dataset_path = (
            PROJECT_ROOT / "evaluation" / "datasets" / dataset_name
        ).absolute()

        if not self.dataset_path.exists():
            raise ValueError(f"The evaluation dataset <{dataset_name}> does not exist!")

        self.samples = sorted(
            x for x in (self.dataset_path / "data").iterdir() if x.is_dir()
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        item: int,
    ) -> Dict[str, np.ndarray]:
        """
        Return the i-th sample. It is composed of a pair of shapes

        Args:
            item: the index of the sample to retrieve

        Returns:
            A dictionary containing the shape A, shape B and the gt matching.
        """
        sample = self.samples[item]
        shape_A = meshio.read((sample / "A.off"))
        shape_B = meshio.read((sample / "B.off"))

        try:
            gt_matching_A_to_B = np.loadtxt(
                sample / "gt_matching_A_to_B.txt", dtype=np.int32
            )
        except (FileNotFoundError, OSError) as e:
            gt_matching_A_to_B = None

        with (self.samples[item] / "meta.json").open() as f:
            meta = json.load(f)

        return {
            "name": self.name,
            "id_A": meta["id"]["A"],
            "id_B": meta["id"]["B"],
            "item": item,
            "path": sample,
            "points_A": shape_A.points,
            "points_B": shape_B.points,
            "faces_A": shape_A.cells_dict["triangle"],
            "faces_B": shape_B.cells_dict["triangle"],
            "gt_matching_A_to_B": gt_matching_A_to_B,
        }

    def __repr__(self) -> str:
        return f"EvalDataset(dataset_name={self.name})"


if __name__ == "__main__":
    print(EvalDataset("faust")[0])
