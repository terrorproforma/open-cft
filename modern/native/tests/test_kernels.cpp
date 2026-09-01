#include "cft_revival/kernels.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

int main() {
    using cft_revival::cusp_arrival_probability;

    assert(cusp_arrival_probability(0.0, 1.0) == 0.0);
    const double canonical_zero = cusp_arrival_probability(-0.0, 1.0);
    assert(canonical_zero == 0.0);
    assert(!std::signbit(canonical_zero));
    assert(std::abs(cusp_arrival_probability(1.0, 1.0) - 0.5) < 1e-15);
    assert(std::abs(cusp_arrival_probability(0.2, 1.0) - 0.05278640450004207) < 1e-15);
    const double tiny_expected = 2.5e-19;
    const double tiny_actual = cusp_arrival_probability(1.0e-18, 1.0);
    assert(std::abs(tiny_actual - tiny_expected) / tiny_expected < 1e-15);
    const double denormal = std::numeric_limits<double>::denorm_min();
    assert(cusp_arrival_probability(4.0 * denormal, 1.0) == denormal);

    bool rejected = false;
    try {
        static_cast<void>(cusp_arrival_probability(2.0, 1.0));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        static_cast<void>(cusp_arrival_probability(
            std::numeric_limits<double>::infinity(), 1.0));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
