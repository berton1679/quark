#include "quark/Config.h"

#include <rapidjson/istreamwrapper.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "quark/StringUtils.h"
#include "rapidjson/document.h"
#include "spdlog/spdlog.h"

using namespace quark;

Config::Config(const std::string &cfg_file, const std::string &symbol, const std::string &mode_date,
               const std::string &output_folder)
    : cfg_file_{cfg_file},
      symbol_{symbol},
      model_date_str_{mode_date},
      output_folder_{output_folder} {}

void Config::parse() {
  if (!std::filesystem::exists(cfg_file_)) {
    spdlog::critical("{} does not exists!!", cfg_file_);
    std::abort();
  }

  std::ifstream ifs{cfg_file_};

  std::string json_data;
  std::string line;
  while (std::getline(ifs, line)) {
    line = trim(line);
    if (line.size() >= 2 && line[0] == '/' && line[1] == '/') {
      continue;
    }
    json_data += line;
  }
  rapidjson::Document doc;
  doc.Parse(json_data.c_str());
  if (doc.HasMember("nfold")) const_cast<int64_t &>(nfold_) = doc["nfold"].GetInt64();

  const_cast<size_t &>(train_size_) = doc["train_size"].GetUint64();
  const_cast<size_t &>(test_size_) = doc["test_size"].GetUint64();

  if (doc.HasMember("de")) {
    DeCfg &de_cfg = const_cast<DeCfg &>(de_cfg_);
    const auto de_json = doc["de"].GetObject();
    if (de_json.HasMember("pop_size")) de_cfg.pop_size_ = de_json["pop_size"].GetUint64();
    if (de_json.HasMember("max_iters")) de_cfg.max_iters_ = de_json["max_iters"].GetUint64();
  }

  if (doc.HasMember("quark_strategy")) {
    QuarkSCfg &quark_s_cfg = const_cast<QuarkSCfg &>(quark_s_cfg_);
    const rapidjson::Value &qks_json = doc["quark_strategy"];
    if (qks_json.HasMember("ntrx"))
      quark_s_cfg.ntrx_ = qks_json["ntrx"].GetDouble();
    else if (qks_json.HasMember("thf"))
      quark_s_cfg.thf_ = qks_json["thf"].GetDouble();
    for (const rapidjson::Value &beta2 : qks_json["beta2"].GetArray()) {
      quark_s_cfg.beta2_.push_back(beta2.GetDouble());
    }
    if (qks_json.HasMember("L1Penalty") && qks_json["L1Penalty"].GetBool()) {
      quark_s_cfg.l1_penalty_ = true;
    }
    if (qks_json.HasMember("DelayTest")) {
      quark_s_cfg.delay_test_index_ = qks_json["DelayTest"].GetUint64();
    }
  }

  if (doc.HasMember("explorer")) {
    ExplorerCfg &explr_cfg = const_cast<ExplorerCfg &>(explorer_cfg_);
    const rapidjson::Value &explr_json = doc["explorer"];
    for (const rapidjson::Value &ntrx : explr_json["ntrxs"].GetArray()) {
      explr_cfg.ntrxs_.push_back(ntrx.GetDouble());
    }
    if (explr_json.HasMember("flip_sign")) {
      explr_cfg.flip_sign_ = explr_json["flip_sign"].GetBool();
    }
    const_cast<bool &>(run_explr_) = true;
  }

  if (doc.HasMember("mpt")) {
    MPTCfg &mpt_cfg = const_cast<MPTCfg &>(mpt_cfg_);
    const rapidjson::Value &mpt_json = doc["mpt"];
    if (mpt_json.HasMember("risk_a")) {
      mpt_cfg.risk_a_ = mpt_json["risk_a"].GetDouble();
    }
    if (mpt_json.HasMember("srT_thf")) {
      mpt_cfg.srt_thf_ = mpt_json["srT_thf"].GetDouble();
    }
    if (mpt_json.HasMember("yreturn_thf")) {
      mpt_cfg.yreturn_thf_ = mpt_json["yreturn_thf"].GetDouble();
    }
  }

  const_cast<bool &>(run_mpt_) = doc.HasMember("explorer") && doc.HasMember("mpt");
}