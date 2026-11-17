"""Per-launch counter quantities from rocprofv3's CSV files."""

import csv

import pytest

from jax_titan.observability import counters

AGENT_FIELDS = ["Node_Id", "Logical_Node_Id", "Agent_Type", "Num_Xcc", "Cu_Count", "Simd_Per_Cu",
                "Max_Engine_Clk_Fcompute", "Product_Name"]
COUNTER_FIELDS = ["Correlation_Id", "Dispatch_Id", "Agent_Id", "Queue_Id", "Process_Id", "Thread_Id", "Grid_Size",
                  "Kernel_Id", "Kernel_Name", "Workgroup_Size", "LDS_Block_Size", "Scratch_Size", "VGPR_Count",
                  "Accum_VGPR_Count", "SGPR_Count", "Counter_Name", "Counter_Value", "Start_Timestamp",
                  "End_Timestamp"]


def write_run(tmp_path, launches):
  with open(tmp_path / "counters_agent_info.csv", "w", newline="") as f:
    w = csv.DictWriter(f, AGENT_FIELDS)
    w.writeheader()
    w.writerow(dict(Node_Id=0, Logical_Node_Id=0, Agent_Type="CPU", Num_Xcc=1, Cu_Count=96, Simd_Per_Cu=0,
                    Max_Engine_Clk_Fcompute=0, Product_Name="cpu"))
    # 8 chiplets, 304 compute units of 4 SIMDs: 152 SIMDs per chiplet.
    w.writerow(dict(Node_Id=2, Logical_Node_Id=2, Agent_Type="GPU", Num_Xcc=8, Cu_Count=304, Simd_Per_Cu=4,
                    Max_Engine_Clk_Fcompute=2100, Product_Name="gpu"))
  with open(tmp_path / "counters_counter_collection.csv", "w", newline="") as f:
    w = csv.DictWriter(f, COUNTER_FIELDS, restval=0)
    w.writeheader()
    for i, (name, grid, start, end, values) in enumerate(launches, 1):
      for counter, value in values.items():
        w.writerow(dict(Dispatch_Id=i, Agent_Id="Agent 2", Process_Id=7, Grid_Size=grid, Kernel_Name=name,
                        Counter_Name=counter, Counter_Value=value, Start_Timestamp=start, End_Timestamp=end))


# A 4 ms launch on 8 chiplets at 1,400 MHz with 100 of 152 matrix units busy per cycle.
GEMM = {"GRBM_GUI_ACTIVE": 8 * 1400 * 4000, "SQ_VALU_MFMA_BUSY_CYCLES": 100 * 8 * 1400 * 4000,
        "TCC_HIT_sum": 750, "TCC_MISS_sum": 250, "TCC_EA0_RDREQ_sum": 1_000_000, "TCC_EA0_WRREQ_sum": 500_000}


def test_metrics_from_counters():
  agent = {"chiplets": 8, "simds_per_chiplet": 152}
  m = counters.metrics({"counters": GEMM, "start_ns": 0, "end_ns": 4_000_000}, agent)
  assert m["duration_us"] == 4000
  assert m["clock_mhz"] == pytest.approx(1400)
  assert m["matrix_busy"] == pytest.approx(100 / 152)
  assert m["l2_hit"] == pytest.approx(0.75)
  assert m["read_mb"] == pytest.approx(64.0)
  assert m["write_mb"] == pytest.approx(32.0)
  # 32-byte read requests count half.
  m = counters.metrics({"counters": dict(GEMM, TCC_EA0_RDREQ_32B_sum=1_000_000), "start_ns": 0,
                        "end_ns": 4_000_000}, agent)
  assert m["read_mb"] == pytest.approx(32.0)


def test_missing_counters_give_none():
  m = counters.metrics({"counters": {"GRBM_GUI_ACTIVE": 1000}, "start_ns": 0, "end_ns": 1000},
                       {"chiplets": 1, "simds_per_chiplet": 4})
  assert m["clock_mhz"] == pytest.approx(1000)
  assert m["matrix_busy"] is None and m["l2_hit"] is None and m["read_mb"] is None


def test_report_groups_launches_and_reads_the_gpu_agent(tmp_path):
  slow = dict(GEMM, GRBM_GUI_ACTIVE=8 * 1100 * 4000)
  write_run(tmp_path, [("gemm_a", 1024, 0, 4_000_000, GEMM), ("gemm_a", 1024, 5_000_000, 9_000_000, slow),
                       ("gemm_a", 2048, 10_000_000, 14_000_000, GEMM), ("copy", 64, 15_000_000, 15_010_000,
                                                                       {"GRBM_GUI_ACTIVE": 8 * 10})])
  result = counters.report(tmp_path)
  assert result["agents"]["Agent 2"] == {"chiplets": 8, "simds_per_chiplet": 152, "max_clock_mhz": 2100,
                                         "name": "gpu"}
  gemm = result["kernels"][0]
  assert gemm["kernel"] == "gemm_a" and gemm["launches"] == 3 and gemm["total_ms"] == pytest.approx(12)
  assert gemm["clock_mhz"] == pytest.approx(1400) and gemm["clock_mhz_range"] == pytest.approx([1100, 1400])
  by_grid = counters.report(tmp_path, by="grid")["kernels"]
  assert {r["kernel"] for r in by_grid} == {"gemm_a grid=1024", "gemm_a grid=2048", "copy grid=64"}
  # Filters: a kernel regex and a minimum duration.
  assert [r["kernel"] for r in counters.report(tmp_path, kernels="^copy")["kernels"]] == ["copy"]
  assert [r["kernel"] for r in counters.report(tmp_path, min_us=100)["kernels"]] == ["gemm_a"]
  assert "| gemm_a | 3 | 12.0 |" in counters.table(result)


def test_options_before_the_separator_stay_with_the_tool(tmp_path):
  argv = ["run", "OUT", "--kernels", "^Cijk_", "--", "bench", "--kernels", "x"]
  assert counters.split_command(argv) == (["run", "OUT", "--kernels", "^Cijk_"], ["bench", "--kernels", "x"])
  assert counters.split_command(["report", "DIR"]) == (["report", "DIR"], [])
  # `true` accepts any arguments, so this runs the whole command line without rocprofv3.
  assert counters.main(["run", str(tmp_path / "out"), "--kernels", "^k", "--rocprof", "true", "--", "app"]) == 0


def test_command_counts_only_the_matching_kernels():
  cmd = counters.command("OUT", ["python", "train.py"], kernels="Cijk_.*")
  assert cmd[:2] == ["rocprofv3", "--pmc"]
  assert cmd[2:2 + len(counters.DEFAULT_COUNTERS)] == list(counters.DEFAULT_COUNTERS)
  assert cmd[cmd.index("--kernel-include-regex") + 1] == "Cijk_.*"
  assert cmd[cmd.index("-d") + 1] == "OUT"
  assert cmd[-3:] == ["--", "python", "train.py"]
  assert "--kernel-include-regex" not in counters.command("OUT", ["x"])
