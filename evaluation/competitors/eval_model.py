import abc
from typing import Dict

import numpy as np


class ModelMatching:
    def __init__(self) -> None:
        """
        Abstract class that defines the generic (minimal) interface all approaches must
        expose:
            - All approaches must have a `name` attribute
            - All approaches must be callable, given a sample return the matching, this
              matching should be under the key 'pred_matching_A_to_B'
        """
        self.name = self.get_name()

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Define the name of this approach

        Returns:
            The name (str)
        """
        pass

    @abc.abstractmethod
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Computes the predicted matching on the given sample

        Args:
            sample: the current sample of shape pairs for which we want to compute the
                    matching

        Returns:
            a dictionary with at least an entry `pred_matching_A_to_B` that contains the
            predicted matching
        """
        pass

    def __repr__(self) -> str:
        return self.name
