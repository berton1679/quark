#pragma once

#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

namespace quark {
  struct Constraints {
    constexpr static double lower_bound_ = -std::numeric_limits<double>::infinity();
    constexpr static double upper_bound_ = std::numeric_limits<double>::infinity();
    Constraints(double lower, double upper) : lower_{lower}, upper_{upper} {
      if (lower == upper) {
        const_cast<double &>(lower_) = lower_bound_;
        const_cast<double &>(upper_) = upper_bound_;
      }
    }

    bool check(double val) const { return val >= lower_ && val <= upper_; }

    const double lower_{0};
    const double upper_{0};
  };
  // Differential Evolution
  template <class Cost_t> class DifferentialEvo {
  public:
    DifferentialEvo(const Cost_t *cost_t, size_t pop_size, int random_seed = 87)
        : cost_t_{cost_t}, pop_size_{pop_size}, num_paras_{cost_t_->get_num_paras()} {
      generator_.seed(random_seed);

      populations_.resize(pop_size_);

      for (auto &pop : populations_) {
        pop.resize(num_paras_);
      }

      min_cost_agents_.resize(pop_size_);
    }
    void init() {
      std::shared_ptr<std::uniform_real_distribution<double>> distribution;

      const auto &cts = cost_t_->get_constraints();
      for (auto &val : populations_) {
        for (size_t ii = 0; ii < num_paras_; ++ii) {
          distribution = std::make_shared<std::uniform_real_distribution<double>>(cts[ii].lower_,
                                                                                  cts[ii].upper_);
          val[ii] = (*distribution)(generator_);
        }
      }

      for (size_t ii = 0; ii < pop_size_; ++ii) {
        min_cost_agents_[ii] = cost_t_->evaluate_cost(populations_[ii]);

        if (min_cost_agents_[ii] < min_cost_) {
          min_cost_ = min_cost_agents_[ii];
          best_agent_ii_ = ii;
        }
      }
    }

    void selection_crossing() {
      std::uniform_real_distribution<double> distribution(0, pop_size_);
      std::uniform_real_distribution<double> f_distribution(0.5, 1.);

      double min_cost = min_cost_agents_[0];
      size_t best_agent_ii = 0;

      for (size_t x = 0; x < pop_size_; ++x) {
        // a, b, c should be different from x
        size_t a = x;
        size_t b = x;
        size_t c = x;

        while (a == x || b == x || c == x || a == b || a == c || b == c) {
          a = distribution(generator_);
          b = distribution(generator_);
          c = distribution(generator_);
        }

        F_ = f_distribution(generator_);
        // F_ = 0.8;
        std::vector<double> z(num_paras_);
        for (size_t ii = 0; ii < num_paras_; ++ii) {
          z[ii] = populations_[a][ii] + F_ * (populations_[b][ii] - populations_[c][ii]);
        }

        std::uniform_real_distribution<double> distribution_para(0, num_paras_);
        int R = distribution_para(generator_);

        std::vector<double> r(num_paras_);
        std::uniform_real_distribution<double> distribution_per_x(0, 1);
        for (auto &val : r) {
          val = distribution_per_x(generator_);
        }

        std::vector<double> new_x(num_paras_);

        // crossing
        for (size_t ii = 0; ii < num_paras_; ++ii) {
          if (r[ii] < CR_ || ii == R) {
            new_x[ii] = z[ii];
          } else {
            new_x[ii] = populations_[x][ii];
          }
        }

        if (!check_cts(new_x)) {
          x--;
          continue;
        }

        // update cost per agent
        double new_cost = cost_t_->evaluate_cost(new_x);
        if (new_cost < min_cost_agents_[x]) {
          populations_[x] = new_x;
          min_cost_agents_[x] = new_cost;
        }

        // update global cost
        if (min_cost_agents_[x] < min_cost) {
          min_cost = min_cost_agents_[x];
          best_agent_ii = x;
        }
      }
      min_cost_ = min_cost;
      best_agent_ii_ = best_agent_ii;
    }

    const std::vector<double> &get_best_agent() const { return populations_[best_agent_ii_]; }

    double get_best_cost() const { return min_cost_agents_[best_agent_ii_]; }

    void optimize(size_t niters, bool verbose = false) {
      init();

      for (size_t ii = 0; ii < niters; ++ii) {
        selection_crossing();

        if (verbose) {
          std::cout << std::fixed << std::setprecision(5);
          std::cout << "current min cost=" << min_cost_ << "\t\t";
          std::cout << "best agent:";
          for (size_t jj = 0; jj < num_paras_; ++jj) {
            std::cout << populations_[best_agent_ii_][jj] << " ";
          }
          std::cout << std::endl;
        }
      }

      if (verbose) {
        std::cout << "done" << std::endl;
      }
    }

  private:
    bool check_cts(const std::vector<double> &agent) {
      const auto &cts = cost_t_->get_constraints();
      for (size_t ii = 0; ii < agent.size(); ++ii) {
        if (!cts[ii].check(agent[ii])) {
          return false;
        }
      }
      return true;
    }

  private:
    const Cost_t *cost_t_;
    const size_t pop_size_{0};
    double F_;
    const double CR_{0.9};
    size_t num_paras_{0};
    std::default_random_engine generator_;
    std::vector<std::vector<double>> populations_;
    std::vector<double> min_cost_agents_;

    size_t best_agent_ii_{0};
    double min_cost_{1e6};
  };
}  // namespace quark