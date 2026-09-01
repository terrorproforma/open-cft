"""Exact L0 leading mean and preregistered GP correction models."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

from cft_revival.physics import (
    ELEMENTARY_CHARGE_C,
    STANDARD_GRAVITY_M_PER_S2,
    XENON_ATOM_MASS_KG,
)
from cft_revival.surrogates import ExactGP, Prediction, SurrogateSchema
from cft_revival.surrogates.identity import canonical_hash

OUTPUT_NAMES = ("axial_thrust_n", "specific_impulse_s")
OUTPUT_UNITS = ("N", "s")
RAW_FAMILY = "raw-ard-matern52"
RESIDUAL_FAMILY = "exact-leading-residual-ard-matern52"
RATIO_FAMILY = "exact-leading-ratio-ard-matern52"


def analytic_outputs(row: Sequence[float]) -> tuple[float, float]:
    """Documented L0 momentum equations from five output-relevant coordinates."""
    voltage = 150.0 + 350.0 * row[0]
    mass_flow = 2.0e-7 + 1.8e-6 * row[1]
    ionized = 0.65 + 0.33 * row[2]
    double_share = 0.15 * row[3]
    axial = 0.75 + 0.23 * row[4]
    plus_speed = sqrt(
        2.0 * ELEMENTARY_CHARGE_C * voltage / XENON_ATOM_MASS_KG
    )
    charge_momentum = 1.0 + (sqrt(2.0) - 1.0) * double_share
    exhaust = ionized * charge_momentum * plus_speed * axial
    return mass_flow * exhaust, exhaust / STANDARD_GRAVITY_M_PER_S2


def analytic_quantities(normalized: Sequence[float]) -> dict[str, float]:
    """Explicit momentum and electrical-boundary identities for eight coordinates."""
    row = (
        normalized[0],
        normalized[1],
        normalized[2],
        normalized[3],
        normalized[5],
    )
    thrust, isp = analytic_outputs(row)
    voltage = 150.0 + 350.0 * normalized[0]
    mass_flow = 2.0e-7 + 1.8e-6 * normalized[1]
    ionized = 0.65 + 0.33 * normalized[2]
    double_share = 0.15 * normalized[3]
    beam_fraction = 0.75 + 0.23 * normalized[4]
    cathode_power = 5.0 + 20.0 * normalized[6]
    ppu_efficiency = 0.82 + 0.13 * normalized[7]
    total_rate = mass_flow / XENON_ATOM_MASS_KG
    beam_current = (
        total_rate
        * ELEMENTARY_CHARGE_C
        * ionized
        * (1.0 + double_share)
    )
    anode_current = beam_current / beam_fraction
    beam_power = beam_current * voltage
    anode_power = anode_current * voltage
    thruster_power = anode_power + cathode_power
    return {
        "axial_thrust_n": thrust,
        "specific_impulse_s": isp,
        "beam_current_a": beam_current,
        "anode_current_a": anode_current,
        "beam_kinetic_power_w": beam_power,
        "anode_input_power_w": anode_power,
        "cathode_input_power_w": cathode_power,
        "thruster_electrical_input_power_w": thruster_power,
        "ppu_input_power_w": thruster_power / ppu_efficiency,
    }


def _schema(output: int) -> SurrogateSchema:
    return SurrogateSchema(
        ("voltage", "flow", "ionized", "double_share", "axial"),
        (OUTPUT_NAMES[output],),
        ("1", "1", "1", "1", "1"),
        (OUTPUT_UNITS[output],),
    )


@dataclass(frozen=True)
class LeadingCorrectionGP:
    family: str
    output: int
    correction: ExactGP

    def predict(self, rows: Sequence[Sequence[float]]) -> tuple[Prediction, ...]:
        corrections = self.correction.predict(rows)
        values = []
        for row, correction in zip(rows, corrections, strict=True):
            leading = analytic_outputs(row)[self.output]
            if self.family == RESIDUAL_FAMILY:
                mean = leading + correction.mean
                variance = correction.variance
            elif self.family == RATIO_FAMILY:
                mean = leading * correction.mean
                variance = leading * leading * correction.variance
            else:
                raise ValueError(f"invalid correction family {self.family}")
            values.append(
                Prediction(
                    mean,
                    variance,
                    correction.nominal_probability,
                    f"{self.family}:{correction.uncertainty_semantics}",
                )
            )
        return tuple(values)

    @property
    def model_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": "l0-leading-correction-gp-v1",
            "family": self.family,
            "output": self.output,
            "correction": self.correction.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LeadingCorrectionGP":
        return cls(
            str(value["family"]),
            int(value["output"]),
            ExactGP.from_dict(value["correction"]),  # type: ignore[arg-type]
        )


def fit_models(
    family: str,
    selected: Sequence[int],
    inputs: Sequence[Sequence[float]],
    observed: Mapping[int, Sequence[float]],
) -> tuple[ExactGP | LeadingCorrectionGP, ExactGP | LeadingCorrectionGP]:
    rows = tuple(inputs[index] for index in selected)
    models = []
    for output in range(2):
        values = tuple(float(observed[index][output]) for index in selected)
        if family == RAW_FAMILY:
            targets = values
        elif family == RESIDUAL_FAMILY:
            targets = tuple(
                value - analytic_outputs(row)[output]
                for row, value in zip(rows, values, strict=True)
            )
        elif family == RATIO_FAMILY:
            targets = tuple(
                value / analytic_outputs(row)[output]
                for row, value in zip(rows, values, strict=True)
            )
        else:
            raise ValueError(f"unknown model family {family}")
        fitted = ExactGP.fit(
            rows,
            targets,
            schema=_schema(output),
            length_scale_mode="ard",
            nominal_probability=0.9,
        )
        models.append(
            fitted
            if family == RAW_FAMILY
            else LeadingCorrectionGP(family, output, fitted)
        )
    return models[0], models[1]
