"""Scaling laws: the presets against their papers, the closed forms against brute force,
fitting, and the budget planner's loss, compute-optimal size and target-loss mode."""

import json
import math

import pytest

from jax_titan.planning import budget
from jax_titan.planning import models
from jax_titan.planning import scaling

GOPHER_FLOPS = 5.76e23  # Hoffmann et al. 2022, Table 3: the compute Gopher and Chinchilla used
REFIT, ORIGINAL = scaling.LAWS["chinchilla_refit"], scaling.LAWS["chinchilla_2022"]
MIXTRAL_8X7B = {"vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32,
                "num_attention_heads": 32, "num_key_value_heads": 8,
                "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2}


def test_presets_match_their_papers():
  # Hoffmann et al.: approach 3 puts the optimum for Gopher's compute at 40B parameters.
  n, d = ORIGINAL.optimal(GOPHER_FLOPS)
  assert n == pytest.approx(40e9, rel=0.05)
  # The refit gives about Chinchilla itself, 70B on 1.4T tokens: about 20 per parameter.
  n, d = REFIT.optimal(GOPHER_FLOPS)
  assert n == pytest.approx(70e9, rel=0.1) and 15 < d / n < 25
  for compute in (1e21, 1e23, 1e25, 1e26):
    n, d = REFIT.optimal(compute)
    assert 15 < d / n < 30
  # Besiroglu et al.: Hoffmann et al.'s estimates imply about 70 tokens per parameter.
  n, d = ORIGINAL.optimal(1e25)
  assert 50 < d / n < 90
  # The rule of thumb against Table 3: 10B on 205.1B tokens, and 1T on 21.2T.
  assert scaling.rule_of_thumb(1.23e22)[0] == pytest.approx(10e9, rel=0.02)
  assert scaling.rule_of_thumb(1.27e26)[1] == pytest.approx(21.2e12, rel=0.05)


@pytest.mark.parametrize("law", [REFIT, ORIGINAL])
def test_closed_forms_against_brute_force(law):
  compute = 1e23
  grid = [10 ** (8 + 5 * i / 20000) for i in range(20001)]
  brute = min(grid, key=lambda n: law.loss(n, compute / (6 * n)))
  n, d = law.optimal(compute)
  assert n == pytest.approx(brute, rel=2e-3) and 6 * n * d == pytest.approx(compute)

  assert law.tokens_for(8e9, law.loss(8e9, 2e12)) == pytest.approx(2e12, rel=1e-6)
  assert law.tokens_for(8e9, law.floor(8e9)) is None

  target = law.loss(n, d)
  cn, cd = law.cheapest(target)
  assert law.loss(cn, cd) == pytest.approx(target, rel=1e-9)
  # The cheapest way to a loss lies on the compute-optimal frontier.
  assert cn == pytest.approx(n, rel=1e-6) and cd == pytest.approx(d, rel=1e-6)
  for other in (cn / 1.5, cn * 1.5):
    assert other * law.tokens_for(other, target) > cn * cd
  assert law.cheapest(law.E) is None


def _points(law, sizes, ratios):
  out = []
  for i, n in enumerate(sizes):
    for j, ratio in enumerate(ratios):
      noise = 1 + 1e-3 * math.sin(7 * i + 3 * j)
      out.append((n, ratio * n, law.loss(n, ratio * n) * noise))
  return out


def test_fit_recovers_a_law():
  points = _points(REFIT, (1e8, 3e8, 1e9, 3e9, 1e10), (5, 10, 20, 40, 80))
  fitted = scaling.fit(points, base=ORIGINAL)
  assert fitted.alpha == pytest.approx(REFIT.alpha, abs=0.03)
  assert fitted.beta == pytest.approx(REFIT.beta, abs=0.03)
  assert fitted.optimal(1e23)[0] == pytest.approx(REFIT.optimal(1e23)[0], rel=0.15)

  # With its exponents fixed, a few runs still give E, A and B.
  few = _points(REFIT, (1e8, 1e9), (10, 40))
  fixed = scaling.fit(few, base=REFIT, fixed=("alpha", "beta"))
  assert (fixed.alpha, fixed.beta) == (REFIT.alpha, REFIT.beta)
  assert fixed.E == pytest.approx(REFIT.E, rel=0.01)
  with pytest.raises(ValueError, match="cannot fit"):
    scaling.fit(few[:2], base=REFIT)


def test_points_from_runs_and_files(tmp_path):
  run = tmp_path / "run"
  run.mkdir()
  records = [{"event": "start", "time": 0, "args": {"model": "llama3-test", "model_overrides": ""}},
             {"event": "layout", "time": 1, "tokens_per_step": 65536},
             {"step": 0, "loss": 12.0, "time": 2},
             {"event": "eval", "time": 3, "step": 8, "loss": 7.5},
             {"event": "eval", "time": 4, "step": 16, "loss": 6.25}]
  (run / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
  (tmp_path / "points.csv").write_text("params,tokens,loss\n1e9,2e10,2.5\n")
  points = scaling.points_from([run, tmp_path / "points.csv"])
  assert points == [(models.from_name("llama3-test").params(), 16 * 65536, 6.25), (1e9, 2e10, 2.5)]

  records[0]["args"]["model_overrides"] = '{"base_num_decoder_layers": 4}'
  (run / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
  with pytest.raises(ValueError, match="model_overrides"):
    scaling.points_from([run])


def test_budget_reports_the_law(tmp_path, capsys):
  out = tmp_path / "plan.json"
  assert budget.main(["--model=llama3.1-70b", "--tokens=20e12", "--gpu=MI350X", "--gpus=1024",
                      "--fsdp", f"--json={out}"]) == 0
  s = json.loads(out.read_text())["scaling"]
  n = models.from_name("llama3.1-70b").params()
  assert s["compute_6nd"] == pytest.approx(6 * n * 20e12)
  assert s["rule_of_thumb"]["params"] == pytest.approx(math.sqrt(s["compute_6nd"] / 120))
  assert s["optimal"]["params"] > 200e9 and s["optimal"]["loss"] < s["loss"]
  assert "Scaling Law Report" in capsys.readouterr().out


def test_budget_target_loss(tmp_path, capsys):
  out = tmp_path / "plan.json"
  assert budget.main(["--model=llama3.1-8b", "--target_loss=2.05", "--gpu=MI355X", "--gpus=256",
                      "--critical_batch_tokens=2e6", f"--json={out}"]) == 0
  r = json.loads(out.read_text())
  n = models.from_name("llama3.1-8b").params()
  assert r["tokens"] == pytest.approx(REFIT.tokens_for(n, 2.05))
  s = r["scaling"]
  assert s["loss"] == pytest.approx(2.05)
  assert s["target"]["cheapest"]["compute_6nd"] < s["compute_6nd"]
  assert s["batch"]["token_factor"] == pytest.approx(1 + 4 * 2**20 / 2e6)
  with pytest.raises(SystemExit):
    budget.main(["--model=llama3.1-8b", "--target_loss=1.9", "--gpu=MI355X", "--gpus=8"])
  assert "out of reach" in capsys.readouterr().err


def test_moe_needs_a_law_fit_to_moe_runs(tmp_path, capsys):
  path = tmp_path / "mixtral" / "config.json"
  path.parent.mkdir()
  path.write_text(json.dumps(MIXTRAL_8X7B))
  out = tmp_path / "plan.json"
  assert budget.main([f"--model=hf:{path}", "--tokens=2e12", "--gpu=MI355X", "--gpus=64",
                      f"--json={out}"]) == 0
  s = json.loads(out.read_text())["scaling"]
  assert "loss" not in s and not s["dense"] and s["rule_of_thumb"]["params"] > 0
  assert "fit to dense models" in capsys.readouterr().out
  with pytest.raises(SystemExit):
    budget.main([f"--model=hf:{path}", "--target_loss=2.0", "--gpu=MI355X", "--gpus=64"])


def test_cli_fits_and_reuses_a_law(tmp_path, capsys):
  points = tmp_path / "points.json"
  points.write_text(json.dumps(_points(REFIT, (1e8, 1e9), (10, 40))))
  law_file = tmp_path / "law.json"
  assert scaling.main(["--fit", str(points), "--fix=alpha,beta", f"--out={law_file}"]) == 0
  name, law = scaling.load(str(law_file))
  assert law.E == pytest.approx(REFIT.E, rel=0.01) and "fixed" in law.source
  assert scaling.main(["--compute=8.4e24"]) == 0
  assert "compute-optimal" in capsys.readouterr().out
  out = tmp_path / "plan.json"
  assert budget.main(["--model=llama3.1-8b", "--tokens=1e12", f"--law={law_file}",
                      f"--json={out}"]) == 0
  s = json.loads(out.read_text())["scaling"]
  assert s["law"] == name and s["loss"] == pytest.approx(law.loss(s["params"], 1e12))
