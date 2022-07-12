from pytorch_lightning import seed_everything

from evaluation.competitors.eval_dataset import EvalDataset
from evaluation.competitors.our.our import OurMatching


class OurMatchingRefined(OurMatching):
    def __init__(self, **kwargs) -> None:
        super(OurMatchingRefined, self).__init__(refine=True, **kwargs)

    def get_simple_name(self) -> str:
        return "our_refined"


if __name__ == "__main__":
    seed_everything(0)
    dataset = EvalDataset("faust")
    sample = dataset[10]
    print(sample["id_A"])
    print(sample["id_B"])
    print(sample["points_A"].shape[0] + sample["points_B"].shape[0])

    model = OurMatchingRefined()
    model(sample)
    print("OK")

    # max_points = 0
    # max_sample_idx = None
    # for i, sample in tqdm(enumerate(dataset)):
    #     n_points = sample["points_A"].shape[0] + sample["points_B"].shape[0]
    #     if n_points > max_points:
    #         max_points = n_points
    #         max_sample_idx = i
    #
    # print(f"biggest sample: {max_sample_idx}, with {max_points} total points")

    # plot_meshes(
    #         meshes=[
    #             Mesh(
    #                 v=y_hat.detach().squeeze().numpy(),
    #                 f=None,
    #             ),
    #             Mesh(
    #                 v=points_A.detach().squeeze().numpy(),
    #                 f=None,
    #             ),
    #             Mesh(
    #                 v=points_B.detach().squeeze().numpy(),
    #                 f=None,
    #             ),
    #         ],
    #         titles=["y hat",'A', 'B'],
    #         showtriangles=[False, False, False],
    #         showscales=None,
    #         autoshow=False,
    #   ).show()
