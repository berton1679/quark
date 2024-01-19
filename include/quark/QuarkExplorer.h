#pragma once
#include "quark/AlphaReader.h"
#include "quark/QuarkOutput.h"

namespace quark {
  class QuarkExplorer {
  public:
    QuarkExplorer(const AlphaReader &reader, const Config &cfg);
    ExplorerReport optimize();
    std::vector<double> optimize_fold(size_t ifold, const double ntrx,
                                      const std::ptrdiff_t ifeature, const bool flip_sign);
    LearningFoldReport get_test_report(size_t ifold, const double ntrx,
                                       const std::ptrdiff_t ifeature,
                                       const std::vector<double> &thfs, const bool flip);

  private:
    const AlphaReader &reader_;
    const Config &cfg_;
  };
}  // namespace quark