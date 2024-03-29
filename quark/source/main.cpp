#include <quark/AlphaReader.h>
#include <quark/Config.h>
#include <quark/QuarkExplorer.h>
#include <quark/QuarkStrategy.h>

#include <cxxopts.hpp>
#include <iostream>
#include <string>
#include <unordered_map>

auto main(int argc, char** argv) -> int {
  cxxopts::Options options(*argv, "A program to welcome the world!");

  std::string language;
  std::string name;

  // clang-format off
  options.add_options()
    ("h,help", "Show help")
    ("v,version", "Print the current version number")
    ("output_folder", "Output folder", cxxopts::value<std::string>())
    ("alpha_folder", "Output folder", cxxopts::value<std::string>())
    ("date", "Model date", cxxopts::value<std::string>())
    ("json", "json file", cxxopts::value<std::string>())
    ("symbol", "symbol", cxxopts::value<std::string>())
    ("l,lang", "Language code to use", cxxopts::value(language)->default_value("en"))
  ;
  // clang-format on

  auto result = options.parse(argc, argv);

  if (result["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const std::string symbol_name = result["symbol"].as<std::string>();
  const std::string model_date = result["date"].as<std::string>();
  const std::string json_file = result["json"].as<std::string>();
  const std::string alpha_folder = result["alpha_folder"].as<std::string>();
  const std::string output_folder = result["output_folder"].as<std::string>();

  quark::Config cfg{json_file, symbol_name, model_date, output_folder};
  cfg.parse();

  quark::AlphaReader alpha_reader{alpha_folder, cfg};
  alpha_reader.read();

  if (cfg.run_explr_ && !cfg.run_mpt_) {
    quark::QuarkExplorer explorer{alpha_reader, cfg};
    const auto explr_report = explorer.optimize();
  } else if (cfg.run_mpt_) {
    // feature filter/sign, based on yreturn
    quark::QuarkExplorer explorer{alpha_reader, cfg};
    const auto explr_report = explorer.optimize();
    quark::QuarkStrategy quark_strategy{alpha_reader, cfg};
    quark_strategy.run_mpt(explr_report);
  } else {
    quark::QuarkStrategy quark_strategy{alpha_reader, cfg};
    // quark_strategy.optimize_all();
    const auto lmr = quark_strategy.optimize();
    quark_strategy.write_model(lmr);
  }

  return 0;
}
