#pragma once

#include <array>

namespace cft_revival {

// Legacy isotropic loss-cone model:
// theta_m = asin(sqrt(B_low / B_high))
// p = integral_0^theta_m sin(theta) dtheta * (2*pi)/(4*pi)
double cusp_arrival_probability(double low_field_t, double high_field_t);

std::array<double, 4> cusp_arrival_probabilities(
    const std::array<double, 4>& low_field_t,
    const std::array<double, 4>& high_field_t);

}  // namespace cft_revival
