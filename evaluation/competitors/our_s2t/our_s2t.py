from evaluation.competitors.eval_dataset import EvalDataset
from evaluation.competitors.our.our import OurMatching
from evaluation.utils import PROJECT_ROOT

checkpoint_file = "best_fine_tune_best_s2s"

CHECKPOINTS_ROOT = PROJECT_ROOT / "evaluation" / "competitors" / "our" / "checkpoints"


class OurMatchingS2T(OurMatching):
    def __init__(self, device="cpu", **kwargs) -> None:
        super(OurMatchingS2T, self).__init__(
            checkpoint_name=checkpoint_file, refine=False, device=device, **kwargs
        )

    def get_simple_name(self) -> str:
        return "our_s2t"


if __name__ == "__main__":

    dataset = EvalDataset("faust_1k_noise")
    model = OurMatchingS2T()
    for i in dataset:
        model(i)
        print(model.get_name())
        break
    print("OK")
