import pytest

from cft_revival.plasma import PlasmaState, XenonGlobalInputs


@pytest.fixture
def dm92_inputs() -> XenonGlobalInputs:
    return XenonGlobalInputs(
        anode_voltage_v=1000.0,
        anode_current_a=1.0,
        cusp_arrival_probabilities=(0.060, 0.119, 0.160, 0.254),
    )


@pytest.fixture
def dm92_published_state() -> PlasmaState:
    return PlasmaState(
        plasma_potential_v=(14.1, 1000.0, 1000.0, 1000.0),
        electron_temperature_ev=(8.9, 100.1, 43.1, 23.5),
        ionization_source_current_a=(0.008, 0.543, 0.310, 0.157),
        electron_current_a=(0.106, 0.107, 0.637, 0.845, 1.002),
        ion_current_a=(0.894, 0.893, 0.363, 0.155, -0.002),
        cusp_ion_current_a=(0.007, 0.013, 0.102),
    )


@pytest.fixture
def self_consistent_case() -> tuple[XenonGlobalInputs, PlasmaState]:
    probability = (0.0, 0.0, 0.0, 0.0)
    voltage = 1000.0
    phi = (10.0, 500.0, 800.0, 1000.0)
    electron = [0.002 * phi[0] ** 1.5]
    source: list[float] = []
    temperature: list[float] = []
    for cell in range(4):
        gain = (
            phi[cell]
            - (0.0 if cell == 0 else phi[cell - 1])
            + (0.0 if cell == 0 else temperature[cell - 1])
        )
        ionization = electron[cell] * 0.07 * gain / 12.1
        source.append(ionization)
        transported = electron[cell] + ionization
        temperature.append(0.68 * electron[cell] * gain / transported)
        if cell < 3:
            electron.append(transported)
    electron.append(electron[3] + source[3])
    current = electron[4]
    inputs = XenonGlobalInputs(voltage, current, probability)
    ion = tuple(current - value for value in electron)
    state = PlasmaState(
        plasma_potential_v=phi,
        electron_temperature_ev=tuple(temperature),  # type: ignore[arg-type]
        ionization_source_current_a=tuple(source),  # type: ignore[arg-type]
        electron_current_a=tuple(electron),  # type: ignore[arg-type]
        ion_current_a=ion,  # type: ignore[arg-type]
        cusp_ion_current_a=(0.0, 0.0, 0.0),
    )
    return inputs, state
