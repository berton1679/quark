#pragma once
#include "quark/AlphaReader.h"
#include "quark/QuarkOutput.h"

namespace quark {
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

  private:
    const AlphaReader &alpha_reader_;
    const Config &cfg_;
  };
}  // namespace quark