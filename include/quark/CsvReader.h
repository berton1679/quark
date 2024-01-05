#pragma once
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace quark {
  class CsvReader {
  public:
    CsvReader(const std::string &filename);
    ~CsvReader();

    const std::vector<std::string> get_headers() { return headers_; }

    bool parse();

    inline std::string get_string(const std::string &name) {
      auto it = header_index_map_.find(name);
      if (it == header_index_map_.end()) {
        return "";
      }
      return values_[it->second];
    }
    inline double get_double(const std::string &name) { return std::stod(get_string(name)); }
    inline int64_t get_int64(const std::string &name) { return std::stoll(get_string(name)); }

  private:
    const std::string filename_;
    std::ifstream ifs_;
    std::string cur_line_;
    std::vector<std::string> headers_;
    std::unordered_map<std::string, size_t> header_index_map_;
    std::vector<std::string> values_;
  };
}  // namespace quark