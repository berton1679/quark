#pragma once
#include "quark/AlphaReader.h"
#include "quark/QuarkOutput.h"

namespace quark {
  struct ExplorerModel {
    // dim = nfeatures
    std::vector<LearningReport> oos_reports_;
    std::vector<double> thfs_;
    std::vector<int> signs_;
    double ntrx_;
  };

  struct ExplorerReport {
    std::vector<ExplorerModel> models_;
  };

  class QuarkExplorer {
  public:
    QuarkExplorer(const AlphaReader &reader, const Config &cfg);
    ExplorerReport optimize();
    double optimize_fold(size_t ifold, const double ntrx, const std::ptrdiff_t ifeature,
                         const bool flip_sign);
    LearningFoldReport get_test_report(size_t ifold, const double ntrx,
                                       const std::ptrdiff_t ifeature, const double thf,
                                       const bool flip);

  private:
    const AlphaReader &reader_;
    const Config &cfg_;
  };
}  // namespace quark