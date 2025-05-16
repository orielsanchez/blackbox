"""Base AlphaModel implementation that supports verbose mode."""

from typing import Dict, List, Optional, Union

import pandas as pd

from blackbox.models.interfaces import AlphaModel


class BaseAlphaModel(AlphaModel):
    """Base class for alpha models with verbose support."""

    def __init__(self, verbose: bool = False, **kwargs):
        """Initialize the alpha model with optional verbose flag.

        Args:
            verbose: Whether to output detailed logs
            **kwargs: Additional parameters for the model
        """
        self.verbose = verbose
        self.params = kwargs

    def set_verbose(self, verbose: bool) -> None:
        """Set verbose mode on or off.

        Args:
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose

    def generate(self, snapshot: Dict) -> pd.Series:
        """Generate alpha signals from snapshot data.

        This is a base implementation that should be overridden by child classes.

        Args:
            snapshot: Market data snapshot

        Returns:
            pd.Series: Alpha signals
        """
        raise NotImplementedError("Child classes must implement generate()")

    def predict(self, snapshot: Dict) -> pd.Series:
        """Generate signals and apply any post-processing.

        Args:
            snapshot: Market data snapshot

        Returns:
            pd.Series: Final alpha signals
        """
        # By default, just call generate
        return self.generate(snapshot)

    def log_debug(self, message: str) -> None:
        """Log debug message only if verbose is enabled."""
        if self.verbose:
            from blackbox.utils.context import get_logger

            logger = get_logger()
            logger.debug(message)

    def log_info(self, message: str, force: bool = False) -> None:
        """Log info message based on verbose setting.

        Args:
            message: Message to log
            force: Whether to log regardless of verbose setting
        """
        if self.verbose or force:
            from blackbox.utils.context import get_logger

            logger = get_logger()
            logger.info(message)
