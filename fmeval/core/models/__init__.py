"""Public surface of the models sub-package."""

from fmeval.core.models.base import ModelWrapper
from fmeval.core.models.mock_model import MockModel

__all__ = ["ModelWrapper", "MockModel"]
