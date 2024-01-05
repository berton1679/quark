#pragma once
#include <filesystem>

#include "Eigen/Dense"
#include "quark/Config.h"
#include "quark/CsvReader.h"

namespace quark {
  // path structure
  // {folder}/{symbol}/alpha_cfg_{date}.yaml(only for debug)
  // {folder}/{symbol}/alphas_{date}.csv(o)
  // {folder}/{symbol}/meta_{date}.json(o)

  struct Meta {
    std::vector<std::string> features_;
    int64_t ndays_{0};
    size_t nrows_{0};
    int64_t start_ts_{0};
  };

  struct Scale {
    double mean_;
    double val_;
  };

  using TsVecType = Eigen::Matrix<int64_t, Eigen::Dynamic, 1>;
  class AlphaReader {
  public:
    AlphaReader(const std::string& alpha_folder, const Config& cfg);
    void read();
    const Meta& get_meta() const { return meta_; }
    const Eigen::MatrixXd& get_x_matrix() const { return x_matrix_; }

    const Eigen::VectorXd& get_price_vec() const { return price_vec_; }

    const TsVecType& get_ts_vec() const { return ts_vec_; }

    const Eigen::MatrixXd get_raw_x_matrix_train(size_t index) const {
      const int64_t st = train_indices_[index].first;
      const int64_t ed = train_indices_[index].second;
      return get_x_matrix_index(st, ed);
    }

    const Eigen::MatrixXd get_x_matrix_train(size_t index) const {
      return get_raw_x_matrix_train(index);
      // const Eigen::MatrixXd raw_x_m = get_raw_x_matrix_train(index);
      // return standardization(raw_x_m, index);
    }
    const Eigen::MatrixXd get_x_matrix_test(size_t index) const {
      const int64_t st = test_indices_[index].first;
      const int64_t ed = test_indices_[index].second;
      return get_x_matrix_index(st, ed);
      // const Eigen::MatrixXd raw_x_m = get_x_matrix_index(st, ed);
      // return standardization(raw_x_m, index);
    }
    const Eigen::MatrixXd get_x_matrix_index(int64_t st, int64_t ed) const;

    const Eigen::VectorXd get_price_vec_train(size_t index) const {
      const int64_t st = train_indices_[index].first;
      const int64_t ed = train_indices_[index].second;
      return get_price_vec_index(st, ed);
    }
    const Eigen::VectorXd get_price_vec_test(size_t index) const {
      const int64_t st = test_indices_[index].first;
      const int64_t ed = test_indices_[index].second;
      return get_price_vec_index(st, ed);
    }
    const Eigen::VectorXd get_price_vec_index(int64_t st, int64_t ed) const;

    const TsVecType get_ts_vec_train(size_t index) const {
      const int64_t st = train_indices_[index].first;
      const int64_t ed = train_indices_[index].second;
      return get_ts_vec_index(st, ed);
    }
    const TsVecType get_ts_vec_test(size_t index) const {
      const int64_t st = test_indices_[index].first;
      const int64_t ed = test_indices_[index].second;
      return get_ts_vec_index(st, ed);
    }
    const TsVecType get_ts_vec_index(int64_t st, int64_t ed) const;
    const Scale& get_scale(size_t ii) const { return scales_[ii]; }
    const Scale& get_fold_scale(size_t ifold, size_t ii) const { return fold_scales_[ifold][ii]; }

    Eigen::MatrixXd standardization(const Eigen::MatrixXd &matrix, size_t ifold) const;

  public:
    const std::filesystem::path alpha_folder_;

  private:
    void read_meta();
    void read_alphas();
    void standardization();
    void split_folds();

    void standardize_by_fold(size_t ifold);

  private:
    const Config& cfg_;
    Eigen::MatrixXd x_matrix_;
    Eigen::VectorXd price_vec_;
    TsVecType ts_vec_;
    Meta meta_;
    std::vector<Scale> scales_;
    std::vector<std::vector<Scale>> fold_scales_;

    std::vector<std::pair<int64_t, int64_t>> test_indices_;
    std::vector<std::pair<int64_t, int64_t>> train_indices_;
  };
}  // namespace quark