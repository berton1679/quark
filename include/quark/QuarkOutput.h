#pragma once
#include "quark/AlphaReader.h"

namespace quark {
  struct LearningFoldReport {
    double ntrx_{0};
    Eigen::VectorXd pnls_;
  };
  struct LearningReport {
    double ntrx_{0};
    double yreturn_{0};
    double srt_{0};
  };
  struct LearningModel {
    std::vector<double> coefs_;
    double holding_sec_{0};
    double thf1_{0.};
    double thf2_{0.};
    LearningReport final_report_;
    double best_beta2_{0.};
  };

  double get_srt(const Eigen::VectorXd &pnls);
}  // namespace quark