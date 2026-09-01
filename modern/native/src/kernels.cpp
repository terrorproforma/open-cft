#include "cft_revival/kernels.hpp"

#include <cmath>
#include <stdexcept>

namespace cft_revival {

double cusp_arrival_probability(const double low_field_t, const double high_field_t) {
    if (!std::isfinite(low_field_t) || !std::isfinite(high_field_t)) {
        throw std::invalid_argument("magnetic fields must be finite");
    }
    if (low_field_t < 0.0 || high_field_t <= 0.0) {
        throw std::invalid_argument("fields require low >= 0 and high > 0");
    }
    if (low_field_t > high_field_t) {
        throw std::invalid_argument("low field cannot exceed high field");
    }
    if (low_field_t == 0.0) {
        return 0.0;
    }

    // Rationalizing 1 - sqrt(1-r) avoids catastrophic cancellation as r -> 0.
    const double ratio = low_field_t / high_field_t;
    return 0.5 * ratio / (1.0 + std::sqrt(1.0 - ratio));
}

std::array<double, 4> cusp_arrival_probabilities(
    const std::array<double, 4>& low_field_t,
    const std::array<double, 4>& high_field_t) {
    std::array<double, 4> probabilities{};
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
        probabilities[index] =
            cusp_arrival_probability(low_field_t[index], high_field_t[index]);
    }
    return probabilities;
}

}  // namespace cft_revival
