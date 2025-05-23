from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


class Tuner(ABC):
    """
    Abstract base class for model parameter tuning.
    """

    def __init__(self, search_space: Dict[str, List[Any]]):
        self.search_space = search_space

    @abstractmethod
    def tune(
        self,
        config_path: Union[str, Path],
        metric: str,
        results_dir: Union[str, Path],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Run tuning over the given search space and return sorted results.

        Returns:
            List of (params, score), best first.
        """
        pass
