#include "quark/CsvReader.h"

#include <iostream>

#include "quark/StringUtils.h"

using namespace quark;

CsvReader::CsvReader(const std::string &filename) : filename_{filename}, ifs_{filename} {
  std::getline(ifs_, cur_line_);

  // get header
  headers_ = split(cur_line_, ',');
  for (size_t ii = 0; ii < headers_.size(); ++ii) {
    header_index_map_.insert({headers_[ii], ii});
  }
}

CsvReader::~CsvReader() { ifs_.close(); }

bool CsvReader::parse() {
  bool ret = false;
  if (!std::getline(ifs_, cur_line_)) {
    return ret;
  }
  values_ = split(cur_line_, ',');
  ret = true;
  return ret;
}
