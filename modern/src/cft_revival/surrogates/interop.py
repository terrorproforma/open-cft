"""Optional BoTorch hand-off contract without a torch-family dependency."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any

from .gp import ExactGP


class OptionalInteropDependencyError(ImportError):
    """The caller requested tensor conversion without installing torch."""


@dataclass(frozen=True, slots=True)
class BoTorchTrainingData:
    """Framework-neutral data matching BoTorch's ``SingleTaskGP`` shapes.

    ``train_x`` is ``n x d``; ``train_y`` and ``train_yvar`` are ``n x 1``.
    Observation variances are physical measurement-noise variances, not latent
    GP variance or cross-fidelity discrepancy.
    """

    train_x: tuple[tuple[float, ...], ...]
    train_y: tuple[tuple[float], ...]
    train_yvar: tuple[tuple[float], ...]
    input_names: tuple[str, ...]
    output_name: str
    schema_hash: str
    training_data_hash: str

    @classmethod
    def from_exact_gp(cls, model: ExactGP) -> BoTorchTrainingData:
        return cls(
            model.train_x,
            tuple((value,) for value in model.train_y),
            tuple((value,) for value in model.observation_variance),
            model.schema.input_names,
            model.schema.output_names[0],
            model.schema_hash,
            model.training_data_hash,
        )

    def to_torch(self, *, dtype: Any | None = None, device: Any | None = None) -> dict[str, Any]:
        if find_spec("torch") is None:
            raise OptionalInteropDependencyError(
                "BoTorch interoperability requires torch; no package was installed"
            )
        torch = import_module("torch")
        selected_dtype = torch.double if dtype is None else dtype
        return {
            "train_X": torch.tensor(self.train_x, dtype=selected_dtype, device=device),
            "train_Y": torch.tensor(self.train_y, dtype=selected_dtype, device=device),
            "train_Yvar": torch.tensor(
                self.train_yvar, dtype=selected_dtype, device=device
            ),
        }


def botorch_available() -> bool:
    return all(find_spec(name) is not None for name in ("torch", "botorch", "gpytorch"))
