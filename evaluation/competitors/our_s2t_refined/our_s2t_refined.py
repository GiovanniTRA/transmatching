from evaluation.competitors.our.our import OurMatching

checkpoint_file = "best_fine_tune_best_s2s"


class OurMatchingRefinedS2T(OurMatching):
    def __init__(self, device="cpu", **kwargs) -> None:
        super(OurMatchingRefinedS2T, self).__init__(
            checkpoint_name=checkpoint_file, refine=True, device=device, **kwargs
        )

    def get_simple_name(self) -> str:
        return "our_s2t_refined"
