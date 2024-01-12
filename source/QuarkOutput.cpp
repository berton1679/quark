#include "quark/QuarkOutput.h"

double quark::get_srt(const Eigen::VectorXd &pnls) {
  double mean = pnls.mean();
  double sum2 = (pnls.array() - mean).cwiseAbs2().sum();
  double std_v = std::sqrt(sum2 / (pnls.size() - 1));
  return mean / std_v * std::sqrt(365.);
};