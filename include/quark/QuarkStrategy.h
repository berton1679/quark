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
  class QuarkStrategy {
  public:
    QuarkStrategy(const AlphaReader &reader, const Config &cfg);
    void optimize_all();
    LearningModel optimize();
    LearningFoldReport get_test_report(size_t ifold, const std::vector<double> &coefs);
    LearningFoldReport get_train_report(size_t ifold, const std::vector<double> &coefs);
    void write_model(const LearningModel &report);

  private:
    std::vector<double> optimize_fold(size_t ifold, double ntrx, double beta2sum);
    double get_srt(const Eigen::VectorXd &pnls);

  private:
    const AlphaReader &alpha_reader_;
    const Config &cfg_;
  };
}  // namespace quark