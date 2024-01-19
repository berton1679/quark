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
    Eigen::VectorXd pnls_;
  };
  struct LearningModel {
    std::vector<double> coefs_;
    double holding_sec_{0};
    double thf1_{0.};
    double thf2_{0.};
    LearningReport final_report_;
    double best_beta2_{0.};
  };

  struct ExplorerModel {
    // dim = nfeatures
    std::vector<LearningReport> oos_reports_;
    std::vector<double> thfs_;
    std::vector<double> force_thfs_;
    std::vector<int> signs_;
    double ntrx_;
  };

  struct ExplorerReport {
    std::vector<ExplorerModel> models_;
  };

  double get_srt(const Eigen::VectorXd &pnls);
}  // namespace quark