#pragma once
#include <string>
#include <vector>

namespace quark {
  struct DeCfg {
    size_t pop_size_{15};
    size_t max_iters_{100};
  };

  struct QuarkSCfg {
    double ntrx_{0};
    double thf_{0};
    std::vector<double> beta2_;
    bool l1_penalty_{false};
    size_t delay_test_index_{0};
  };

  struct ExplorerCfg {
    std::vector<double> ntrxs_;
    bool flip_sign_{true};
  };

  class Config {
  public:
    Config(const std::string &cfg_file, const std::string &symbol, const std::string &mode_date,
           const std::string &output_folder);
    void parse();

  private:
    const std::string cfg_file_;

  public:
    const std::string symbol_;
    const std::string model_date_str_;
    const std::string output_folder_;
    const int64_t nfold_{4};
    const size_t train_size_{0};
    const size_t test_size_{0};
    const DeCfg de_cfg_;
    const QuarkSCfg quark_s_cfg_;
    const ExplorerCfg explorer_cfg_;
  };
}  // namespace quark