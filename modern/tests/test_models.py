import pytest

from cft_revival.models import AppConfig, DesignPoint, ValidationError


VALID_DESIGN = (300.0, 1.0, 10.0, 3.0, 8.0, 12.0, 20.0, 30.0)


def test_design_point_preserves_legacy_order() -> None:
    assert DesignPoint.from_sequence(VALID_DESIGN).as_sequence() == VALID_DESIGN


@pytest.mark.parametrize(
    "values",
    [
        (*VALID_DESIGN[:4], 3.005, *VALID_DESIGN[5:]),
        (*VALID_DESIGN[:7], 50.0),
        (300.0, 1.0, 10.0, 2.5, 8.0, 12.0, 20.0, 30.0),
    ],
)
def test_design_point_rejects_legacy_geometry_violations(
    values: tuple[float, ...],
) -> None:
    with pytest.raises(ValidationError):
        DesignPoint.from_sequence(values)


def test_femm_configuration_cannot_enable_parallel_access(tmp_path) -> None:
    with pytest.raises(ValidationError, match="serialized"):
        AppConfig.from_mapping(
            {
                "femm_export_directory": "data",
                "magnetic_backend": "femm-export",
                "serialize_femm": False,
            },
            tmp_path,
        )
