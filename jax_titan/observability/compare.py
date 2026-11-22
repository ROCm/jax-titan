"""Compares training runs of one model in two implementations, such as JAX-Titan's
native model and an adapter's: their loss curves and their speed.

  python -m jax_titan.observability.compare REFERENCE RUN [RUN ...] [--window 50]
      [--skip 5] [--names A,B] [--figure loss.svg] [--json OUT]

Each RUN is a trainer `--run_dir` or log file, the first the reference. Runs
whose trainer arguments are the same but for where they write are repeats of
one configuration and are reported together. It prints, as Markdown tables:

- the trainer arguments in which the configurations differ: two models
  compared on the same training differ in `--model` alone;
- the mean training loss over each window of `--window` steps and the
  validation losses, each with its difference to the reference; for scale,
  how much the reference's loss varies from one step to the next, and how
  much the repeats of a configuration differ;
- tokens/s per GPU, the median over steady steps (as `summarize` takes it)
  and over the repeats, with the repeats' spread and the ratio to the
  reference, and peak HBM per GPU, as a range where the GPUs differ.

With `--figure`, it also draws each configuration's loss against the step,
its mean over the window and the difference of that mean to the
reference's (needs matplotlib, the `plots` extra). It judges nothing: models
that start from other random weights part at first, and the tables give the
numbers to judge their curves by.
"""

import argparse
import json
import math
import pathlib
import statistics

from jax_titan.observability import records as rec
from jax_titan.observability import summarize

# Trainer arguments that say where a run writes, which repeats differ in.
_OUTPUTS = ("run_dir", "log_file", "checkpoint_dir", "profile_dir", "save_params")


def load(path, skip: int = 5) -> dict:
  """One run: its trainer arguments, its loss by step and validation loss by
  step, and `summarize`'s metrics."""
  path = pathlib.Path(path)
  records = rec.read(summarize._logs(path)[0])  # pylint: disable=protected-access
  start = next(iter(rec.events(records, "start")), None)
  if start is None:
    raise ValueError(f"{path} has no start record, so its trainer arguments are unknown")
  args = {k: v for k, v in start["args"].items() if k not in _OUTPUTS}
  return {
      "run": str(path),
      "args": args,
      "loss": {r["step"]: r["loss"] for r in rec.steps(records)},
      "validation": {e["step"]: e["loss"] for e in rec.events(records, "eval")},
      "metrics": summarize.run_metrics(path, skip=skip),
  }


def configurations(runs: list[dict]) -> list[dict]:
  """Runs grouped by their trainer arguments, the reference's configuration first,
  each named by its `--model` or, where models repeat, by the arguments that
  set it apart."""
  groups = {}
  for r in runs:
    groups.setdefault(json.dumps(r["args"], sort_keys=True), []).append(r)
  configs = [{"args": g[0]["args"], "runs": g} for g in groups.values()]
  names = sorted({k for c in configs for k in c["args"]})
  differ = [k for k in names if len({json.dumps(c["args"].get(k)) for c in configs}) > 1]
  models = [c["args"].get("model") for c in configs]
  for c in configs:
    if len(set(models)) == len(models):
      c["name"] = c["args"].get("model")
    else:
      c["name"] = " ".join(f"{k}={c['args'].get(k)}" for k in differ)
    c["differs"] = {k: c["args"].get(k) for k in differ}
  return configs


def _windows(loss: dict, window: int) -> list[tuple[int, int]]:
  last = max(loss) + 1 if loss else 0
  return [(s, min(s + window, last)) for s in range(0, last, window)]


def _mean(loss: dict, lo: int, hi: int) -> float | None:
  values = [loss[s] for s in range(lo, hi) if s in loss]
  return statistics.fmean(values) if values else None


def _step_to_step(loss: dict, lo: int, hi: int) -> float | None:
  """How much the loss of a step varies around the curve: the deviation of the
  differences of neighbouring steps over sqrt(2), which the curve's own slope
  hardly reaches."""
  diffs = [loss[s + 1] - loss[s] for s in range(lo, hi - 1) if s in loss and s + 1 in loss]
  return statistics.pstdev(diffs) / math.sqrt(2) if len(diffs) > 1 else None


def compare(paths: list, window: int = 50, skip: int = 5, names: list | None = None) -> dict:
  """The numbers of the tables `markdown` prints, for runs at `paths`, the first
  the reference; `names`, when given, names the configurations in the order of
  their first run."""
  configs = configurations([load(p, skip) for p in paths])
  if names:
    if len(names) != len(configs):
      raise ValueError(f"{len(names)} names for {len(configs)} configurations")
    for c, name in zip(configs, names):
      c["name"] = name
  reference = configs[0]
  windows = _windows(reference["runs"][0]["loss"], window)
  eval_steps = sorted(reference["runs"][0]["validation"])
  for c in configs:
    runs = c["runs"]
    means = [[_mean(r["loss"], lo, hi) for r in runs] for lo, hi in windows]
    c["window_loss"] = [statistics.fmean(m) if None not in m else None for m in means]
    c["repeats_differ"] = [max(m) - min(m) if len(m) > 1 and None not in m else None
                           for m in means]
    vals = [[r["validation"].get(s) for r in runs] for s in eval_steps]
    c["validation"] = [statistics.fmean(v) if None not in v else None for v in vals]
    speeds = [r["metrics"].get("tokens_per_s_per_gpu") for r in runs]
    speeds = [s for s in speeds if s]
    c["tokens_per_s_per_gpu"] = statistics.median(speeds) if speeds else None
    c["speed_spread"] = ((max(speeds) - min(speeds)) / statistics.median(speeds)
                         if len(speeds) > 1 else None)
    hbm = [r["metrics"]["peak_hbm_gib"] for r in runs if r["metrics"].get("peak_hbm_gib")]
    c["peak_hbm_gib"] = {"min": min(h["min"] for h in hbm),
                         "max": max(h["max"] for h in hbm)} if hbm else None
    c["devices"] = runs[0]["metrics"].get("devices")
  first = reference["runs"][0]["loss"]
  return {
      "window": window,
      "windows": windows,
      "eval_steps": eval_steps,
      "step_to_step": [_step_to_step(first, lo, hi) for lo, hi in windows],
      "configurations": [
          {k: v for k, v in c.items() if k != "runs"} | {"runs": [r["run"] for r in c["runs"]]}
          for c in configs],
      "_loss": [[r["loss"] for r in c["runs"]] for c in configs],
  }


def _fmt(x, spec: str) -> str:
  return "-" if x is None else format(x, spec)


def _hbm(peak: dict | None) -> str:
  """The peak HBM of the GPUs, or its range where the GPUs differ by 1 GiB or more."""
  if peak is None:
    return "-"
  if peak["max"] - peak["min"] < 1:
    return f"{peak['max']:.1f}"
  return f"{peak['min']:.1f} to {peak['max']:.1f}"


def markdown(report: dict) -> str:
  """The report as Markdown tables."""
  configs = report["configurations"]
  ref = configs[0]
  differs = sorted({k for c in configs for k in c["differs"] if k != "model"})
  lines = [f"Reference: `{ref['name']}`." + ("" if len(configs) == 1 else (
      " The configurations differ in `--model` alone." if not differs else
      " The configurations also differ in " + ", ".join(f"`--{k}`" for k in differs) + ":"))]

  def table(title, columns, rows):
    lines.extend(["", f"| {title} | " + " | ".join(columns) + " |",
                  "|---" * (len(columns) + 1) + "|"])
    lines.extend(f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows)

  def named(c):
    return f"`{c['name']}`" + (" (reference)" if c is ref else "")

  def against_reference(c, key):
    return [_fmt(v, ".4f") + ("" if c is ref or v is None or b is None else f" ({v - b:+.4f})")
            for v, b in zip(c[key], ref[key])]

  if differs:
    table("Configuration", [f"`--{k}`" for k in differs],
          [(named(c), [f"`{json.dumps(c['differs'].get(k))}`" for k in differs])
           for c in configs])
  columns = [f"{lo}-{hi - 1}" for lo, hi in report["windows"]]
  rows = [(named(c), against_reference(c, "window_loss")) for c in configs]
  rows.append(("Step to step, reference", [_fmt(v, ".4f") for v in report["step_to_step"]]))
  repeats = [max((c["repeats_differ"][i] for c in configs if c["repeats_differ"][i] is not None),
                 default=None) for i in range(len(columns))]
  if any(r is not None for r in repeats):
    rows.append(("Repeats differ by up to", [_fmt(v, ".4f") for v in repeats]))
  table("Training loss, mean over steps", columns, rows)
  if report["eval_steps"]:
    table("Validation loss, after step", [str(s) for s in report["eval_steps"]],
          [(named(c), against_reference(c, "validation")) for c in configs])
  rows = []
  for c in configs:
    speed, base, spread = c["tokens_per_s_per_gpu"], ref["tokens_per_s_per_gpu"], c["speed_spread"]
    rows.append((named(c), [
        str(c["devices"]), str(len(c["runs"])), _fmt(speed, ",.0f"),
        "-" if spread is None else f"{100 * spread:.1f}%",
        _fmt(speed / base if speed and base else None, ".3f"), _hbm(c["peak_hbm_gib"])]))
  table("Speed", ["GPUs", "Runs", "Tokens/s per GPU", "Spread over runs", "Ratio to reference",
                  "Peak HBM per GPU (GiB)"], rows)
  return "\n".join(lines) + "\n"


def figure(report: dict, path) -> None:
  """Each configuration's loss against the step with its mean over the window
  around each step, and the difference of that mean to the reference's."""
  from jax_titan.observability import plot  # pylint: disable=g-import-not-at-top

  plt = plot._pyplot()  # pylint: disable=protected-access
  window, configs = report["window"], report["configurations"]
  if len(configs) > 1:
    fig, (loss_ax, diff_ax) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]})
  else:
    fig, loss_ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

  def centred(loss: dict) -> dict:
    steps = sorted(loss)
    half = window // 2
    return {s: statistics.fmean(loss[t] for t in steps[max(0, i - half):i + half + 1])
            for i, s in enumerate(steps)}

  means, colors = [], []
  for c, losses in zip(configs, report["_loss"]):
    first = losses[0]
    line, = loss_ax.plot(sorted(first), [first[s] for s in sorted(first)], linewidth=0.5,
                         alpha=0.35)
    colors.append(line.get_color())
    means.append(centred(first))
    loss_ax.plot(list(means[-1]), list(means[-1].values()), color=colors[-1], linewidth=1.5,
                 label=c["name"])
  loss_ax.set(ylabel="training loss",
              title=f"Training loss of each step, and its mean over the {window} steps around it")
  loss_ax.legend(fontsize=8)
  if len(configs) > 1:
    for c, mean, color in list(zip(configs, means, colors))[1:]:
      common = sorted(set(mean) & set(means[0]))
      diff_ax.plot(common, [mean[s] - means[0][s] for s in common], color=color, linewidth=1.2,
                   label=c["name"])
    diff_ax.axhline(0, color="gray", linewidth=0.8)
    diff_ax.set(xlabel="step", ylabel="difference",
                title=f"Mean over {window} steps minus the reference's ({configs[0]['name']})")
    diff_ax.legend(fontsize=8)
  else:
    loss_ax.set_xlabel("step")
  fig.savefig(path, dpi=120)
  plt.close(fig)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("runs", nargs="+", help="Run directories or logs, the reference first.")
  parser.add_argument("--window", type=int, default=50, help="Steps per mean of the loss.")
  parser.add_argument("--skip", type=int, default=5,
                      help="Warmup steps of each run left out of its speed.")
  parser.add_argument("--names", default="",
                      help="Comma-separated names of the configurations, in the order of their "
                      "first run, instead of their --model.")
  parser.add_argument("--figure", default="", help="Draw the loss curves here (svg, png, pdf).")
  parser.add_argument("--json", default="", help="Write every number here.")
  args = parser.parse_args(argv)
  report = compare(args.runs, window=args.window, skip=args.skip,
                   names=args.names.split(",") if args.names else None)
  print(markdown(report), end="")
  if args.figure:
    figure(report, args.figure)
    print(f"\nwrote {args.figure}")
  if args.json:
    out = {k: v for k, v in report.items() if not k.startswith("_")}
    pathlib.Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
