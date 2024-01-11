#include "quark/QuarkStrategy.h"

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

#include <filesystem>
#include <fstream>

#include "quark/DifferentialEvo.h"
#include "quark/QuarkLearning.h"
#include "quark/QuarkOutput.h"
#include "spdlog/spdlog.h"

using namespace quark;

QuarkStrategy::QuarkStrategy(const AlphaReader &reader, const Config &cfg)
    : alpha_reader_{reader}, cfg_{cfg} {}

void QuarkStrategy::optimize_all() {
  const Meta &meta = alpha_reader_.get_meta();
  const QuarkSCfg &qks_cfg = cfg_.quark_s_cfg_;
  std::shared_ptr<QuarkBaseLearning> ql;
  if (qks_cfg.thf_ == 0) {
    ql = std::make_shared<QuarkLearning>(meta.features_.size(), alpha_reader_.get_x_matrix(),
                                         alpha_reader_.get_price_vec(), alpha_reader_.get_ts_vec(),
                                         0, qks_cfg.ntrx_, qks_cfg.l1_penalty_);
  } else {
    ql = std::make_shared<QuarkSLearning>(meta.features_.size(), alpha_reader_.get_x_matrix(),
                                          alpha_reader_.get_price_vec(), alpha_reader_.get_ts_vec(),
                                          0, qks_cfg.thf_, qks_cfg.l1_penalty_);
  }

  // QuarkLearning ql{meta.features_.size(),
  //                  alpha_reader_.get_x_matrix(),
  //                  alpha_reader_.get_price_vec(),
  //                  alpha_reader_.get_ts_vec(),
  //                  0.,
  //                  qks_cfg.ntrx_,
  //                  qks_cfg.l1_penalty_};
  const auto &de_cfg = cfg_.de_cfg_;
  DifferentialEvo<QuarkBaseLearning> de{ql.get(), de_cfg.pop_size_};
  de.optimize(de_cfg.max_iters_, true);
  const auto &p_trx = ql->get_position(de.get_best_agent());
  const auto pnls = ql->get_pnl_day(p_trx.first);
  double srt = get_srt(pnls);
  double yreturn = pnls.mean() * 365;
  spdlog::info("yreturn:{} srT:{}", yreturn, srt);
}

LearningModel QuarkStrategy::optimize() {
  const auto &qks_cfg = cfg_.quark_s_cfg_;
  const double arg = qks_cfg.thf_ == 0 ? qks_cfg.ntrx_ : qks_cfg.thf_;
  const auto &meta = alpha_reader_.get_meta();
  std::vector<LearningReport> final_reports;
  for (size_t ibeta = 0; ibeta < qks_cfg.beta2_.size(); ++ibeta) {
    double beta2_sum = qks_cfg.beta2_[ibeta];
    std::vector<LearningFoldReport> reports;
    for (size_t ifold = 0; ifold < cfg_.nfold_; ++ifold) {
      const auto coefs = optimize_fold(ifold, arg, beta2_sum);
      // get test_pnl
      const auto test_report = get_test_report(ifold, coefs);
      reports.push_back(test_report);
    }
    // get all pnls, calculate metric

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

    LearningReport final_test_report;
    final_test_report.ntrx_ = sum_ntrx / total_days;
    final_test_report.yreturn_ = all_pnls.mean() * 365;
    final_test_report.srt_ = get_srt(all_pnls);

    spdlog::info("beta2: {}, ntrx:{}, yreturn:{}, srT:{}", beta2_sum, final_test_report.ntrx_,
                 final_test_report.yreturn_, final_test_report.srt_);

    final_reports.push_back(final_test_report);
  }

  // choose from srT
  size_t best_model_index = 0;
  double best_v = -1e6;
  for (size_t ibeta = 0; ibeta < qks_cfg.beta2_.size(); ++ibeta) {
    if (final_reports[ibeta].srt_ > best_v) {
      best_v = final_reports[ibeta].srt_;
      best_model_index = ibeta;
    }
  }

  const auto best_ceofs = optimize_fold(cfg_.nfold_, arg, qks_cfg.beta2_[best_model_index]);
  spdlog::info("best beta2: {}", qks_cfg.beta2_[best_model_index]);
  const auto final_train_report = get_train_report(cfg_.nfold_, best_ceofs);
  spdlog::info("ntrx: {}, yreturn:{}", final_train_report.ntrx_,
               final_train_report.pnls_.mean() * 365);
  // return output model
  LearningModel learning_model;
  learning_model.final_report_ = final_reports[best_model_index];
  for (size_t ii = 0; ii < meta.features_.size(); ++ii) {
    learning_model.coefs_.push_back(best_ceofs[ii]);
  }
  if (best_ceofs.size() > meta.features_.size()) {
    learning_model.holding_sec_ = best_ceofs[meta.features_.size()];
    learning_model.thf1_ = best_ceofs[meta.features_.size() + 1];
    learning_model.thf2_ = best_ceofs[meta.features_.size() + 2];
    learning_model.best_beta2_ = qks_cfg.beta2_[best_model_index];
  } else {
    learning_model.thf1_ = qks_cfg.thf_;
  }
  return learning_model;
}

LearningFoldReport QuarkStrategy::get_test_report(size_t ifold, const std::vector<double> &coefs) {
  const Meta &meta = alpha_reader_.get_meta();
  const QuarkSCfg &qks_cfg = cfg_.quark_s_cfg_;
  const auto &x_matrix = alpha_reader_.get_x_matrix_test(ifold);
  const auto &price_vec = alpha_reader_.get_price_vec_test(ifold);
  const auto &ts_vec = alpha_reader_.get_ts_vec_test(ifold);

  std::shared_ptr<QuarkBaseLearning> ql;
  if (qks_cfg.thf_ == 0) {
    ql = std::make_shared<QuarkLearning>(meta.features_.size(), x_matrix, price_vec, ts_vec, 0.,
                                         qks_cfg.ntrx_, qks_cfg.l1_penalty_);
  } else {
    ql = std::make_shared<QuarkSLearning>(meta.features_.size(), x_matrix, price_vec, ts_vec, 0.,
                                          qks_cfg.thf_, qks_cfg.l1_penalty_);
  }
  const auto &p_trx = ql->get_position(coefs);
  const auto pnls = ql->get_pnl_day(p_trx.first);

  // for (size_t ii = 0; ii < price_vec.size(); ++ii)
  // {
  //   std::cout << price_vec(ii) << " " << ts_vec(ii) << " " << p_trx.second(ii) << " " <<
  //   p_trx.first(ii) << std::endl;
  // }

  LearningFoldReport ret;
  ret.ntrx_ = ql->get_num_trx(p_trx.second);
  ret.pnls_ = pnls;
  return ret;
}

LearningFoldReport QuarkStrategy::get_train_report(size_t ifold, const std::vector<double> &coefs) {
  const Meta &meta = alpha_reader_.get_meta();
  const QuarkSCfg &qks_cfg = cfg_.quark_s_cfg_;
  const auto &x_matrix = alpha_reader_.get_x_matrix_train(ifold);
  const auto &price_vec = alpha_reader_.get_price_vec_train(ifold);
  const auto &ts_vec = alpha_reader_.get_ts_vec_train(ifold);
  std::shared_ptr<QuarkBaseLearning> ql;
  if (qks_cfg.thf_ == 0) {
    ql = std::make_shared<QuarkLearning>(meta.features_.size(), x_matrix, price_vec, ts_vec, 0.,
                                         qks_cfg.ntrx_, qks_cfg.l1_penalty_);
  } else {
    ql = std::make_shared<QuarkSLearning>(meta.features_.size(), x_matrix, price_vec, ts_vec, 0.,
                                          qks_cfg.thf_, qks_cfg.l1_penalty_);
  }
  const auto &p_trx = ql->get_position(coefs);
  const auto pnls = ql->get_pnl_day(p_trx.first);
  // for (size_t ii = 0; ii < price_vec.size(); ++ii)
  // {
  //   std::cout << price_vec(ii) << " " << ts_vec(ii) << " " << p_trx.second(ii) << " " <<
  //   p_trx.first(ii) << std::endl;
  // }

  LearningFoldReport ret;
  ret.ntrx_ = ql->get_num_trx(p_trx.second);
  ret.pnls_ = pnls;
  return ret;
}

std::vector<double> QuarkStrategy::optimize_fold(size_t ifold, double ntrx, double beta2sum) {
  const Meta &meta = alpha_reader_.get_meta();
  const auto &x_matrix = alpha_reader_.get_x_matrix_train(ifold);
  const auto &price_vec = alpha_reader_.get_price_vec_train(ifold);
  const auto &ts_vec = alpha_reader_.get_ts_vec_train(ifold);
  const QuarkSCfg &qks_cfg = cfg_.quark_s_cfg_;
  std::shared_ptr<QuarkBaseLearning> ql;
  if (qks_cfg.thf_ == 0) {
    ql = std::make_shared<QuarkLearning>(meta.features_.size(), x_matrix, price_vec, ts_vec,
                                         beta2sum, ntrx, qks_cfg.l1_penalty_);
  } else {
    ql = std::make_shared<QuarkSLearning>(meta.features_.size(), x_matrix, price_vec, ts_vec,
                                          beta2sum, ntrx, qks_cfg.l1_penalty_);
  }
  // QuarkLearning ql{meta.features_.size(), x_matrix, price_vec, ts_vec, beta2sum, ntrx,
  //                  qks_cfg.l1_penalty_};
  const auto &de_cfg = cfg_.de_cfg_;
  DifferentialEvo<QuarkBaseLearning> de{ql.get(), de_cfg.pop_size_};
  de.optimize(de_cfg.max_iters_, false);

  const auto &coefs = de.get_best_agent();
  const auto p_trx = ql->get_position(coefs);
  const auto ntrx_fold = ql->get_num_trx(p_trx.second);
  const double beta2_sum = ql->get_beta2(coefs);
  if (false) {
    spdlog::info("ifold: {}, ntrx:{}, beta2:{}", ifold, ntrx_fold, beta2_sum);
    std::stringstream ss;
    ss << "[";
    for (size_t ii = 0; ii < coefs.size(); ++ii) {
      ss << coefs[ii] << " ";
    }
    ss << "]";
    spdlog::info("coefs: {}", ss.str());
    spdlog::info("-----------------------");
  }

  return coefs;
}

void QuarkStrategy::write_model(const LearningModel &report) {
  using path = std::filesystem::path;
  const std::string output_folder
      = path(cfg_.output_folder_) / path(cfg_.symbol_) / path(cfg_.model_date_str_);
  std::filesystem::create_directories(output_folder);
  const Meta &meta = alpha_reader_.get_meta();
  const std::string model_file_name = "model_" + cfg_.model_date_str_ + ".json";
  const std::string model_json_path = path(output_folder) / path(model_file_name);
  spdlog::info("Write model file, {}", model_json_path);
  rapidjson::Document doc;
  doc.SetObject();
  std::ofstream ofs{model_json_path};
  rapidjson::OStreamWrapper osw{ofs};

  auto &allocator = doc.GetAllocator();

  // write features
  {
    rapidjson::Value features{rapidjson::kArrayType};
    for (size_t ii = 0; ii < meta.features_.size(); ++ii) {
      rapidjson::Value feature;
      feature.SetString(meta.features_[ii].c_str(), allocator);
      features.PushBack(feature, allocator);
    }

    doc.AddMember("features", features, allocator);
  }

  // write scale
  {
    rapidjson::Value scalers{rapidjson::kArrayType};
    for (size_t ii = 0; ii < meta.features_.size(); ++ii) {
      rapidjson::Value scaler;
      scaler.SetObject();
      const auto &cur_scaler = alpha_reader_.get_scale(ii);
      // const auto &cur_scaler = alpha_reader_.get_fold_scale(cfg_.nfold_, ii);
      scaler.AddMember("mean", cur_scaler.mean_, allocator);
      scaler.AddMember("val", cur_scaler.val_, allocator);
      scalers.PushBack(scaler, allocator);
    }
    doc.AddMember("scalers", scalers, allocator);
  }
  // write model
  {
    rapidjson::Value model_details{rapidjson::kObjectType};
    {
      rapidjson::Value coefs{rapidjson::kArrayType};
      for (size_t ii = 0; ii < meta.features_.size(); ++ii) {
        rapidjson::Value coef;
        coef.SetDouble(report.coefs_[ii]);
        coefs.PushBack(coef, allocator);
      }
      model_details.AddMember("coefs", coefs, allocator);

      rapidjson::Value holding_sec;
      holding_sec.SetDouble(report.holding_sec_);
      model_details.AddMember("holding_sec", holding_sec, allocator);

      rapidjson::Value thf1;
      thf1.SetDouble(report.thf1_);
      model_details.AddMember("thf1", thf1, allocator);

      rapidjson::Value thf2;
      thf2.SetDouble(report.thf2_);
      model_details.AddMember("thf2", thf2, allocator);

      rapidjson::Value best_beta2_v;
      best_beta2_v.SetDouble(report.best_beta2_);
      model_details.AddMember("beta2", best_beta2_v, allocator);
    }
    doc.AddMember("model", model_details, allocator);
  }
  // write report
  {
    rapidjson::Value report_v{rapidjson::kObjectType};
    {
      rapidjson::Value ntrx_v;
      ntrx_v.SetDouble(cfg_.quark_s_cfg_.ntrx_);
      report_v.AddMember("ntrx_model", ntrx_v, allocator);

      rapidjson::Value thf_v;
      thf_v.SetDouble(cfg_.quark_s_cfg_.thf_);
      report_v.AddMember("thf", thf_v, allocator);

      rapidjson::Value ntrx_test;
      ntrx_test.SetDouble(report.final_report_.ntrx_);
      report_v.AddMember("ntrx", ntrx_test, allocator);

      rapidjson::Value yreturn_v;
      yreturn_v.SetDouble(report.final_report_.yreturn_);
      report_v.AddMember("yreturn", yreturn_v, allocator);

      rapidjson::Value srt_v;
      if (!std::isnan(report.final_report_.srt_)) {
        srt_v.SetDouble(report.final_report_.srt_);
        report_v.AddMember("srT", srt_v, allocator);
      }

      rapidjson::Value l1_v;
      l1_v.SetInt64(cfg_.quark_s_cfg_.l1_penalty_);
      report_v.AddMember("L1Penalty", l1_v, allocator);
    }

    doc.AddMember("report", report_v, allocator);
  }

  // write ndays
  {
    rapidjson::Value ndays_v;
    ndays_v.SetInt64(meta.ndays_);
    doc.AddMember("ndays", ndays_v, allocator);
  }

  // write nrows
  {
    rapidjson::Value nrows_v;
    nrows_v.SetInt64(meta.nrows_);
    doc.AddMember("nrows", nrows_v, allocator);
  }

  // write start_ts
  {
    rapidjson::Value start_ts_v;
    start_ts_v.SetInt64(meta.start_ts_);
    doc.AddMember("start_ts", start_ts_v, allocator);
  }

  // write model_date
  {
    rapidjson::Value model_date_str_v;
    model_date_str_v.SetString(cfg_.model_date_str_.c_str(), allocator);
    doc.AddMember("model_date", model_date_str_v, allocator);
  }

  // write symbol
  {
    rapidjson::Value symbol_v;
    symbol_v.SetString(cfg_.symbol_.c_str(), allocator);
    doc.AddMember("symbol", symbol_v, allocator);
  }

  // write nfold
  {
    rapidjson::Value nfold_v;
    nfold_v.SetInt64(cfg_.nfold_);
    doc.AddMember("nfold", nfold_v, allocator);
  }

  // write train_size
  {
    rapidjson::Value train_size_v;
    train_size_v.SetInt64(cfg_.train_size_);
    doc.AddMember("train_size", train_size_v, allocator);
  }

  // write test_size
  {
    rapidjson::Value test_size_v;
    test_size_v.SetInt64(cfg_.test_size_);
    doc.AddMember("test_size", test_size_v, allocator);
  }

  // write deconfig
  {
    rapidjson::Value decfg_v{rapidjson::kObjectType};

    rapidjson::Value pop_size_v;
    pop_size_v.SetInt64(cfg_.de_cfg_.pop_size_);
    decfg_v.AddMember("pop_size", pop_size_v, allocator);

    rapidjson::Value max_iters_v;
    max_iters_v.SetInt64(cfg_.de_cfg_.max_iters_);
    decfg_v.AddMember("max_iters", max_iters_v, allocator);

    doc.AddMember("DeCfg", decfg_v, allocator);
  }

  rapidjson::Writer<rapidjson::OStreamWrapper> writer{osw};
  doc.Accept(writer);

  // copy alpha_cfg_{date}.yaml file
  const std::string alpha_cfg_name = "alpha_cfg_" + cfg_.model_date_str_ + ".yaml";
  const std::string alpha_cfg_yaml = alpha_reader_.alpha_folder_ / path(alpha_cfg_name);
  const std::string alpha_cfg_output = path(output_folder) / path(alpha_cfg_name);
  spdlog::info("copy {} to {}", alpha_cfg_yaml, alpha_cfg_output);
  std::filesystem::copy(alpha_cfg_yaml, alpha_cfg_output,
                        std::filesystem::copy_options::overwrite_existing);
}

void QuarkStrategy::run_mpt(const ExplorerReport &report) {
  const auto &explr_cfg = cfg_.explorer_cfg_;
  const auto &ntrxs = explr_cfg.ntrxs_;
  for (size_t intrx = 0; intrx < ntrxs.size(); ++intrx) {
    run_mpt_ntrx(report, intrx);
  }
}

void QuarkStrategy::run_mpt_ntrx(const ExplorerReport &report, size_t intrx) {
  const auto &mpt_cfg = cfg_.mpt_cfg_;
  const auto &explr_cfg = cfg_.explorer_cfg_;
  const double ntrx = explr_cfg.ntrxs_[intrx];
  const auto &explr_model = report.models_[intrx];
  const size_t nfeature = explr_model.oos_reports_.size();
  const auto &oos_report = explr_model.oos_reports_;
  const auto &meta = alpha_reader_.get_meta();
  std::vector<Eigen::VectorXd> pnls;
  std::vector<size_t> index_mapping;
  const double diff_ntrx_thf = 3;
  spdlog::info("=================NTRX={}=================", ntrx);
  for (size_t ifeature = 0; ifeature < nfeature; ++ifeature) {
    const double cur_ntrx = oos_report[ifeature].ntrx_;
    if (std::abs(cur_ntrx - ntrx) > diff_ntrx_thf) {
      spdlog::warn("skip {}, {} != {}", meta.features_[ifeature], cur_ntrx, ntrx);
      continue;
    }
    const double cur_srt = oos_report[ifeature].srt_;
    const double cur_yreturn = oos_report[ifeature].yreturn_;
    if (cur_srt < mpt_cfg.srt_thf_) {
      spdlog::warn("skip {} srT, {} < {}", meta.features_[ifeature], cur_srt, mpt_cfg.srt_thf_);
      continue;
    }
    if (cur_yreturn < mpt_cfg.yreturn_thf_) {
      spdlog::warn("skip {} yreturn, {} < {}", meta.features_[ifeature], cur_yreturn,
                   mpt_cfg.yreturn_thf_);
      continue;
    }

    pnls.push_back(oos_report[ifeature].pnls_);
    index_mapping.push_back(ifeature);
  }

  std::vector<double> rescale_coef;
  rescale_coef.resize(nfeature);
  if (!pnls.empty()) {
    QuarkPortfolio qp{pnls, mpt_cfg.risk_a_};
    const auto &de_cfg = cfg_.de_cfg_;
    DifferentialEvo<QuarkPortfolio> de{&qp, de_cfg.pop_size_};
    de.optimize(de_cfg.max_iters_, false);
    const auto &coefs = de.get_best_agent();
    double mean = qp.get_mean(coefs);
    double var = qp.get_variance(coefs);
    double srT = mean / std::sqrt(var) * std::sqrt(365);
    double yreturn = mean * 365;
    double coef_sum = std::accumulate(coefs.begin(), coefs.end(), 0.);
    if (coef_sum > 0.) {
      for (size_t icoef = 0; icoef < coefs.size(); ++icoef) {
        size_t index = index_mapping[icoef];
        rescale_coef[index] = coefs[icoef] / coef_sum;
      }
    }
    spdlog::info("MPT done, srT={}, yreturn={}", srT, yreturn);
  }
  std::stringstream ss;
  ss << "(";
  for (size_t icoef = 0; icoef < rescale_coef.size(); ++icoef) {
    ss << rescale_coef[icoef];
    if (icoef != rescale_coef.size() - 1) {
      ss << ",";
    }
  }
  ss << ")";
  spdlog::info("coefs={}", ss.str());
}