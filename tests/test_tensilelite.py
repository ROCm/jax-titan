"""TensileLite generation: log selection for the config generator, problem ids of Tensile sizes,
library logic reading and the config type fix."""

import pytest
import yaml

from jax_titan.tuning import tensilelite

# A Llama-like gate/up forward GEMM as hipBLASLt 1.4 logs it.
LINE = ("hipblaslt-bench --api_method c -m 14336 -n 8192 -k 4096 --lda 14336 --ldb 4096 --ldc 14336 "
        "--ldd 14336  --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0  --alpha 1.000000 "
        "--alphai 0.000000 --beta 0.000000 --betai 0.000000 --transA N --transB N --batch_count 1 "
        "--batch_mode 0 --scaleA 0 --scaleB 0   --a_type bf16_r --b_type bf16_r --c_type bf16_r "
        "--d_type bf16_r --scale_type f32_r --bias_type f32_r   --compute_type f32_r --algo_method index "
        "--solution_index 75696 --activation_type none --any_stride --rotating 0 --cold_iters 0 --iters 0")
PID = "N,N,1,14336,8192,4096,bf16_r,bf16_r,bf16_r,f32_r"


def test_select_keeps_one_line_per_problem_without_the_newer_fields(tmp_path):
  log = tmp_path / "hbl_1.log"
  log.write_text("\n".join([LINE, LINE, LINE.replace("-k 4096", "-k 1024").replace("--ldb 4096", "--ldb 1024")]))
  lines = tensilelite.select_lines([str(log)], only="^N,N,1,14336,8192,4096,")
  assert len(lines) == 1
  assert "--alphai" not in lines[0] and "--betai" not in lines[0] and "--batch_mode" not in lines[0]
  assert "--alpha 1.000000 --beta 0.000000 --transA N" in lines[0]
  assert "--batch_count 1 --scaleA 0" in lines[0]


def test_tensile_size_maps_to_the_hipblaslt_problem_id():
  problem_type = {"DataType": "B", "DestDataType": "B", "ComputeDataType": "S", "TransposeA": False,
                  "TransposeB": True, "OperationType": "GEMM"}
  assert tensilelite._key(problem_type, [14336, 4096, 1, 8192]) == \
      "N,T,1,14336,4096,8192,bf16_r,bf16_r,bf16_r,f32_r"
  # Library logic files spell data types as ids.
  assert tensilelite._key(dict(problem_type, DataType=7, DestDataType=7, ComputeDataType=0),
                          [14336, 4096, 1, 8192]).endswith("bf16_r,bf16_r,bf16_r,f32_r")


def test_read_logic_gives_each_exact_size_its_solution(tmp_path):
  doc = [{"MinimumRequiredVersion": "5.0.0"}, "aquavanjaram", "gfx942", ["Device 0049"],
         {"OperationType": "GEMM", "DataType": 7, "DestDataType": 7, "ComputeDataType": 0,
          "TransposeA": False, "TransposeB": False},
         [{"SolutionIndex": 0, "SolutionNameMin": "Cijk_Ailk_Bljk_BBS_BH_MT256x256x64", "MacroTile0": 256,
           "MacroTile1": 256, "DepthU": 64}],
         [2, 3, 0, 1], [[[14336, 8192, 1, 4096, 14336, 14336, 14336, 4096], [0, 650000.0]]],
         None, None, "DeviceEfficiency", "Equality"]
  path = tmp_path / "logic.yaml"
  path.write_text(yaml.dump(doc))
  (rec,) = tensilelite.read_logic(path)
  assert rec["id"] == PID and rec["perf"] == 650000.0
  assert rec["solution"].endswith("MT256x256x64") and rec["macro_tile"] == [256, 256, 64]
  # Tensile itself writes the same parts as a mapping.
  mapping = {"ProblemType": doc[4], "Solutions": doc[5], "ExactLogic": doc[7], "PerfMetric": "DeviceEfficiency"}
  path.write_text(yaml.dump(mapping))
  assert tensilelite.read_logic(path)[0]["id"] == PID


def test_benchmark_csv_gives_the_fastest_kernel_per_size(tmp_path):
  path = tmp_path / "Cijk_Ailk_Bljk_BBS_BH_UserArgs_00.csv"
  path.write_text("GFlops, SizeI, SizeJ, SizeK, SizeL, LDD, LDC, LDA, LDB, TotalFlops, GbpsBW, MT_a, MT_b\n"
                  "0, 14336, 8192, 1, 4096, 14336, 14336, 14336, 4096, 962072674304, 0, 600000.0, 648000.5\n")
  assert tensilelite.read_benchmark(path) == {(14336, 8192, 1, 4096): (648000.5, "MT_b")}


def test_generated_configs_get_boolean_transposes(tmp_path):
  path = tmp_path / "tuning_template.0.yaml"
  path.write_text(yaml.dump({"BenchmarkProblems": [[{"TransposeA": 0, "TransposeB": 1, "DataType": "B"},
                                                    {"ForkParameters": []}]],
                             "LibraryLogic": {"ArchitectureName": {"Architecture": "gfx942", "CUCount": 304}}}))
  tensilelite._fix_types(path)
  doc = yaml.safe_load(path.read_text())
  problem_type = doc["BenchmarkProblems"][0][0]
  assert problem_type["TransposeA"] is False and problem_type["TransposeB"] is True
  assert doc["LibraryLogic"]["ArchitectureName"] == "gfx942"


def test_a_problem_called_with_beta_1_is_timed_with_beta_1(tmp_path):
  path = tmp_path / "tuning_template.1.yaml"
  problem = [{"TransposeA": 0, "TransposeB": 0},
             {"BenchmarkFinalParameters": [{"ProblemSizes": [{"Exact": [4096, 8192, 1, 14336]}]}]}]
  path.write_text(yaml.dump({"BenchmarkProblems": [problem], "GlobalParameters": {"DataInitTypeBeta": 0}}))
  tensilelite._fix_types(path, {(4096, 8192, 1, 14336): 1.0})
  assert yaml.safe_load(path.read_text())["GlobalParameters"]["DataInitTypeBeta"] == 1
  path.write_text(yaml.dump({"BenchmarkProblems": [problem], "GlobalParameters": {"DataInitTypeBeta": 0}}))
  tensilelite._fix_types(path, {(4096, 8192, 1, 14336): 0.0})
  assert yaml.safe_load(path.read_text())["GlobalParameters"]["DataInitTypeBeta"] == 0


def test_the_stock_arm_never_loads_the_built_library(monkeypatch):
  monkeypatch.setenv("HIPBLASLT_TENSILE_LIBPATH", "/somewhere/else")
  assert "HIPBLASLT_TENSILE_LIBPATH" not in tensilelite._library_env(3)
  env = tensilelite._library_env(3, "/built/library")
  assert env["HIPBLASLT_TENSILE_LIBPATH"] == "/built/library" and env["HIP_VISIBLE_DEVICES"] == "3"


def test_compare_table_reports_the_change_against_stock():
  row = {"id": PID, "default_ms": 1.8617, "candidate_ms": 1.7428, "separated": True,
         "library_kernel": "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT256x192x64_MI16x16x1_SN"}
  assert "| -6.4% | yes | MT256x192x64_MI16x16x1 |" in tensilelite.compare_table([row])


def test_config_needs_a_matching_problem(tmp_path):
  log = tmp_path / "hbl_1.log"
  log.write_text(LINE + "\n")
  with pytest.raises(ValueError):
    tensilelite.config([str(log)], tmp_path / "src", tmp_path / "out", only="^T,T,")
