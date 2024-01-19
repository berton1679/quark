#include "quark/AlphaReader.h"

#include <rapidjson/istreamwrapper.h>

#include <fstream>
#include <iostream>

#include "rapidjson/document.h"
#include "spdlog/spdlog.h"

using namespace quark;
using path = std::filesystem::path;

AlphaReader::AlphaReader(const std::string &alpha_folder, const Config &cfg)
    : alpha_folder_{path(alpha_folder) / path(cfg.symbol_)}, cfg_{cfg} {}

void AlphaReader::read() {
  read_meta();
  read_alphas();
  // each fold owns different scales
  if (!cfg_.apply_fold_scale_) standardization();
  split_folds();

  if (cfg_.apply_fold_scale_) {
    fold_scales_.resize(cfg_.nfold_ + 1);
    for (size_t ifold = 0; ifold < fold_scales_.size(); ++ifold) {
      standardize_by_fold(ifold);
    }
  }
}

void AlphaReader::read_meta() {
  rapidjson::Document doc;
  const std::string meta_file = alpha_folder_ / path("meta_" + cfg_.model_date_str_ + ".json");

  std::ifstream ifs{meta_file};
  rapidjson::IStreamWrapper iswr{ifs};

  doc.ParseStream(iswr);
  meta_.features_.clear();
  for (const auto &name : doc["features"].GetArray()) {
    meta_.features_.push_back(name.GetString());
  }
  meta_.ndays_ = doc["model_cfg"]["Days"].GetInt64();
  meta_.start_ts_ = doc["start_ts"].GetInt64();
  meta_.nrows_ = doc["nrows"].GetUint64();

  size_t required_nrows = cfg_.train_size_ + cfg_.nfold_ * cfg_.test_size_;
  if (meta_.nrows_ < required_nrows) {
    spdlog::critical("required_nrows < meta_.nrows, ({}, {})", required_nrows, meta_.nrows_);
    std::abort();
  }
}

void AlphaReader::read_alphas() {
  const std::string alpha_file = alpha_folder_ / path("alphas_" + cfg_.model_date_str_ + ".csv");
  CsvReader reader{alpha_file};

  x_matrix_.resize(meta_.nrows_, meta_.features_.size());
  price_vec_.resize(meta_.nrows_);
  ts_vec_.resize(meta_.nrows_);

  size_t irow = 0;
  while (reader.parse()) {
    for (size_t ii = 0; ii < meta_.features_.size(); ++ii) {
      x_matrix_(irow, ii) = reader.get_double(meta_.features_[ii]);
    }
    price_vec_(irow) = reader.get_double("open");
    ts_vec_(irow) = reader.get_int64("open_ts");
    irow += 1;
  }
}

void AlphaReader::standardization() {
  spdlog::info("xdim=({},{})", x_matrix_.rows(), x_matrix_.cols());
  scales_.resize(meta_.features_.size());
  spdlog::info("============features stats============");
  for (size_t ii = 0; ii < meta_.features_.size(); ++ii) {
    double mean = 0;
    for (std::ptrdiff_t irow = 0; irow < x_matrix_.rows(); ++irow) {
      mean += x_matrix_(irow, ii);
    }
    mean /= x_matrix_.rows();

    double sum2 = 0;
    for (std::ptrdiff_t irow = 0; irow < x_matrix_.rows(); ++irow) {
      sum2 += (x_matrix_(irow, ii) - mean) * (x_matrix_(irow, ii) - mean);
    }

    double var = sum2 / (x_matrix_.rows() - 1);

    scales_[ii].mean_ = mean;
    scales_[ii].val_ = std::sqrt(var);

    for (std::ptrdiff_t irow = 0; irow < x_matrix_.rows(); ++irow) {
      x_matrix_(irow, ii) = (x_matrix_(irow, ii) - mean) / scales_[ii].val_;
    }

    spdlog::info("{}, (mean,val)=({},{})", meta_.features_[ii], scales_[ii].mean_,
                 scales_[ii].val_);
  }

  spdlog::info("============features stats end============");
}

void AlphaReader::standardize_by_fold(size_t ifold) {
  const Eigen::MatrixXd cur_x_matrix = get_raw_x_matrix_train(ifold);
  fold_scales_[ifold].resize(meta_.features_.size());
  spdlog::info("===============ifold:{}===============", ifold);
  spdlog::info("xdim=({},{})", cur_x_matrix.rows(), cur_x_matrix.cols());
  for (size_t ii = 0; ii < meta_.features_.size(); ++ii) {
    double mean = 0;
    for (std::ptrdiff_t irow = 0; irow < cur_x_matrix.rows(); ++irow) {
      mean += cur_x_matrix(irow, ii);
    }
    mean /= cur_x_matrix.rows();

    double sum2 = 0;
    for (std::ptrdiff_t irow = 0; irow < cur_x_matrix.rows(); ++irow) {
      sum2 += (cur_x_matrix(irow, ii) - mean) * (cur_x_matrix(irow, ii) - mean);
    }
    double var = sum2 / (cur_x_matrix.rows() - 1);

    fold_scales_[ifold][ii].mean_ = mean;
    fold_scales_[ifold][ii].val_ = std::sqrt(var);

    spdlog::info("{}, (mean,val)=({}, {})", meta_.features_[ii], fold_scales_[ifold][ii].mean_,
                 fold_scales_[ifold][ii].val_);
  }
}

void AlphaReader::split_folds() {
  const auto nfolds = cfg_.nfold_;
  const auto train_size = cfg_.train_size_;
  const auto test_size = cfg_.test_size_;
  const auto delay_index = cfg_.delay_test_index_;

  size_t test_start_index = train_size;
  // for (size_t ii = 0; ii < nfolds + 1; ++ii) {
  //   test_indices_.push_back({test_start_index, test_start_index + test_size + 1});
  //   train_indices_.push_back({test_start_index - train_size, test_start_index + 1});
  //   test_start_index += test_size;
  // }
  for (size_t ii = 0; ii < nfolds + 1; ++ii) {
    std::ptrdiff_t test_start = test_start_index + delay_index;
    std::ptrdiff_t test_end = test_start_index + test_size + 1 + delay_index;
    test_end = std::min(x_matrix_.rows(), test_end);
    test_indices_.push_back({test_start, test_end});
    train_indices_.push_back({test_start_index - train_size, test_start_index + 1});
    test_start_index += test_size;
  }
}

const Eigen::MatrixXd AlphaReader::get_x_matrix_index(int64_t st, int64_t ed) const {
  const Eigen::MatrixXd m = x_matrix_.block(st, 0, ed - st, x_matrix_.cols());
  return m;
}

const Eigen::VectorXd AlphaReader::get_price_vec_index(int64_t st, int64_t ed) const {
  return price_vec_.block(st, 0, ed - st, 1);
}

const TsVecType AlphaReader::get_ts_vec_index(int64_t st, int64_t ed) const {
  return ts_vec_.block(st, 0, ed - st, 1);
}

Eigen::MatrixXd AlphaReader::standardization(const Eigen::MatrixXd &matrix, size_t ifold) const {
  Eigen::MatrixXd ret = matrix;
  for (std::ptrdiff_t icol = 0; icol < matrix.cols(); ++icol) {
    for (std::ptrdiff_t irow = 0; irow < matrix.rows(); ++irow) {
      ret(irow, icol)
          = (matrix(irow, icol) - fold_scales_[ifold][icol].mean_) / fold_scales_[ifold][icol].val_;
    }
  }
  return ret;
}
