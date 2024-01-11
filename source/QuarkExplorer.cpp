#include "quark/QuarkExplorer.h"

#include "quark/QuarkLearning.h"
#include "spdlog/spdlog.h"

using namespace quark;

QuarkExplorer::QuarkExplorer(const AlphaReader &reader, const Config &cfg)
    : reader_{reader}, cfg_{cfg} {}

ExplorerReport QuarkExplorer::optimize() {
  const auto &explr_cfg = cfg_.explorer_cfg_;
  const auto &meta = reader_.get_meta();

  ExplorerReport exp_report;

  for (size_t intrx = 0; intrx < explr_cfg.ntrxs_.size(); ++intrx) {
    exp_report.models_.emplace_back();
    exp_report.models_.back().ntrx_ = explr_cfg.ntrxs_[intrx];
  }

  for (std::ptrdiff_t ifeature = 0; ifeature < meta.features_.size(); ++ifeature) {
    spdlog::info("====================ifeature: {},{}====================", ifeature, meta.features_[ifeature]);
    for (size_t intrx = 0; intrx < explr_cfg.ntrxs_.size(); ++intrx) {
      const double ntrx = explr_cfg.ntrxs_[intrx];
      double final_thf = 0.;
      LearningReport ntrx_oos_report;
      ntrx_oos_report.yreturn_ = -10000000000000;
      bool ntrx_flip = false;
      for (size_t iflip = 0; iflip < 2; ++iflip) {
        bool flip = iflip == 1;
        std::vector<LearningFoldReport> reports;
        for (size_t ifold = 0; ifold < cfg_.nfold_; ++ifold) {
          double thf = optimize_fold(ifold, ntrx, ifeature, flip);
          const auto test_report = get_test_report(ifold, ntrx, ifeature, thf, flip);
          reports.push_back(test_report);
        }

        // transform pnl to Eigen::VectorXd
        Eigen::VectorXd all_pnls;
        size_t total_days = 0;
        for (size_t ifold = 0; ifold < reports.size(); ++ifold) {
          total_days += reports[ifold].pnls_.size();
        }
        all_pnls.resize(total_days);
        int64_t idays = 0;
        double sum_ntrx = 0;
        for (size_t ifold = 0; ifold < reports.size(); ++ifold) {
          for (int64_t ii = 0; ii < reports[ifold].pnls_.size(); ++ii) {
            all_pnls(idays) = reports[ifold].pnls_(ii);
            sum_ntrx += reports[ifold].ntrx_;
            idays += 1;
          }
        }
        LearningReport oos_report;
        oos_report.ntrx_ = sum_ntrx / total_days;
        oos_report.yreturn_ = all_pnls.mean() * 365;
        oos_report.srt_ = get_srt(all_pnls);

        if (oos_report.yreturn_ > ntrx_oos_report.yreturn_) {
          ntrx_oos_report = oos_report;
          ntrx_flip = flip;
        }
      }

      auto &cur_model = exp_report.models_[intrx];
      double ntrx_thf = optimize_fold(cfg_.nfold_, ntrx, ifeature, ntrx_flip);
      cur_model.oos_reports_.push_back(ntrx_oos_report);
      if (ntrx_flip)
        cur_model.signs_.push_back(-1);
      else
        cur_model.signs_.push_back(1);
      cur_model.thfs_.push_back(ntrx_thf);

      spdlog::info("ntrx:{}, srT:{}, yreturn:{}, test_ntrx:{}, flip:{}, thf:{}",
                   explr_cfg.ntrxs_[intrx], ntrx_oos_report.srt_, ntrx_oos_report.yreturn_,
                   ntrx_oos_report.ntrx_, ntrx_flip, ntrx_thf);
    }
  }

  return exp_report;
}

double QuarkExplorer::optimize_fold(size_t ifold, const double ntrx, const std::ptrdiff_t ifeature,
                                    const bool flip_sign) {
  const auto &x_matrix = reader_.get_x_matrix_train(ifold);
  const auto &price_vec = reader_.get_price_vec_train(ifold);
  const auto &ts_vec = reader_.get_ts_vec_train(ifold);

  QuarkExplorerLearning ql{ifeature, x_matrix, price_vec, ts_vec, ntrx, flip_sign};
  const auto &de_cfg = cfg_.de_cfg_;
  DifferentialEvo<QuarkExplorerLearning> de{&ql, de_cfg.pop_size_};
  de.optimize(de_cfg.max_iters_, false);
  const auto &coefs = de.get_best_agent();
  if (false) {
    const auto &p_trx = ql.get_position(coefs);
    double cur_ntrx = ql.get_num_trx(p_trx.second);
    spdlog::info("ifold:{},cur_ntrx:{},ntrx:{}", ifold, cur_ntrx, ntrx);
  }
  return coefs.front();
}

LearningFoldReport QuarkExplorer::get_test_report(size_t ifold, const double ntrx,
                                                  const std::ptrdiff_t ifeature, const double thf,
                                                  const bool flip) {
  const auto &x_matrix = reader_.get_x_matrix_test(ifold);
  const auto &price_vec = reader_.get_price_vec_test(ifold);
  const auto &ts_vec = reader_.get_ts_vec_test(ifold);

  QuarkExplorerLearning ql{ifeature, x_matrix, price_vec, ts_vec, ntrx, flip};
  std::vector<double> coefs;
  coefs.push_back(thf);
  const auto &p_trx = ql.get_position(coefs);
  const auto pnls = ql.get_pnl_day(p_trx.first);

  LearningFoldReport ret;
  ret.ntrx_ = ql.get_num_trx(p_trx.second);
  ret.pnls_ = pnls;
  return ret;
}
