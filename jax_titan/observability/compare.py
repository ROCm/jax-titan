"""Compares training runs of one model structure in two or more implementations, such as
JAX-Titan's native model and an adapter's: their loss curves and their speed.

  python -m jax_titan.observability.compare REFERENCE RUN [RUN ...] [--window 50]
      [--skip 5] [--names A,B] [--figure loss.svg] [--json OUT]

Each RUN is a trainer `--run_dir` or log file, the first the reference. Runs
whose trainer arguments and GPU count are the same, but for where they write
and how they are watched, are repeats of one configuration and are reported
together. It prints, as Markdown tables:

- the trainer arguments, and the GPU counts, in which the configurations
  differ: two models compared on the same training differ in `--model` alone;
- the mean training loss over each window of `--window` steps and the
  validation losses, each with its difference to the reference and, for
  scale, how much the repeats of a configuration differ;
- tokens/s per GPU, the median over steady steps (as `summarize` takes it)
  and over the repeats, with the repeats' spread and the ratio to the
  reference, and the highest peak HBM per GPU of a run, as a range where its
  GPUs differ.

A run counts in a window only with every step of it, a configuration's value
is the mean of the repeats that have it, and the output names the runs that
end before the reference. With `--figure`, it also draws each configuration's
loss, the mean of its repeats, against the step, with its mean over the
window around each step (`centred_means`) and that mean's difference to the
reference's (needs matplotlib, the `plots` extra). It judges nothing: models
that start from other random weights part at first, and the tables give the
numbers to judge their curves by.
"""

import argparse
import json
import pathlib
import statistics

from jax_titan.observability import records as rec
from jax_titan.observability import summarize

# Trainer arguments that change neither a training step nor its speed: where a run
# writes, how it is watched, a rehearsed failure, and the peak its MFU is taken against.
_ASIDE = ("run_dir", "log_file", "checkpoint_dir", "profile_dir", "save_params", "step_timeout",
          "power_sample_s", "fail_at_step", "peak_tflops_per_device")


def _training(start: dict) -> dict:
  """What sets a run's training apart: its arguments but `_ASIDE`, and its GPU count."""
  args = {k: v for k, v in start["args"].items() if k not in _ASIDE}
  return args | {"num_devices": start.get("num_devices")}


def load(path, skip: int = 5) -> dict:
  """One run: its training (`_training`), its loss by step and validation loss by
  step, and `summarize`'s metrics.

  Raises ValueError for a log whose attempts trained otherwise, since their
  steps would mix two trainings under one name.
  """
  path = pathlib.Path(path)
  records = rec.read(summarize._logs(path)[0])  # pylint: disable=protected-access
  starts = rec.events(records, "start")
  if not starts:
    raise ValueError(f"{path} has no start record, so its trainer arguments are unknown")
  training = [_training(s) for s in starts]
  changed = sorted({k for t in training[1:] for k in t.keys() | training[0].keys()
                    if t.get(k) != training[0].get(k)})
  if changed:
    raise ValueError(f"{path} holds attempts that differ in {', '.join(changed)}")
  return {
      "run": str(path),
      "args": _training(starts[0]),
      "loss": {r["step"]: r["loss"] for r in rec.steps(records)},
      "validation": {e["step"]: e["loss"] for e in rec.events(records, "eval")},
      "metrics": summarize.run_metrics(path, skip=skip),
  }


def configurations(runs: list[dict]) -> list[dict]:
  """Runs grouped by their training, the reference's configuration first, each
  named by its `--model` or, where models repeat, by the arguments that set it
  apart."""
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
  """The mean loss of steps lo to hi - 1, or None for a run that lacks one of them."""
  if any(s not in loss for s in range(lo, hi)):
    return None
  return statistics.fmean(loss[s] for s in range(lo, hi))


def _spread(values: list) -> float | None:
  return max(values) - min(values) if len(values) > 1 else None


def centred_means(losses: list[dict], window: int) -> dict:
  """The mean loss over the `window` steps around each step, the mean over the
  repeats (`losses`, step to loss) that have every one of those steps; steps
  whose window reaches past the runs are left out."""
  steps, half = sorted(set().union(*losses)), window // 2
  out = {}
  for s in steps:
    lo = s - half
    means = [m for m in (_mean(loss, lo, lo + window) for loss in losses) if m is not None]
    if lo >= 0 and means:
      out[s] = statistics.fmean(means)
  return out


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
  first = configs[0]["runs"][0]
  windows = _windows(first["loss"], window)
  eval_steps = sorted(first["validation"])
  last = windows[-1][1] - 1 if windows else 0
  for c in configs:
    runs = c["runs"]
    means = [[m for m in (_mean(r["loss"], lo, hi) for r in runs) if m is not None]
             for lo, hi in windows]
    c["window_loss"] = [statistics.fmean(m) if m else None for m in means]
    c["repeats_differ"] = [_spread(m) for m in means]
    vals = [[r["validation"][s] for r in runs if s in r["validation"]] for s in eval_steps]
    c["validation"] = [statistics.fmean(v) if v else None for v in vals]
    c["validation_repeats_differ"] = [_spread(v) for v in vals]
    speeds = [r["metrics"].get("tokens_per_s_per_gpu") for r in runs]
    speeds = [s for s in speeds if s]
    c["tokens_per_s_per_gpu"] = statistics.median(speeds) if speeds else None
    c["speed_spread"] = (_spread(speeds) / statistics.median(speeds)
                         if len(speeds) > 1 else None)
    hbm = [r["metrics"]["peak_hbm_gib"] for r in runs if r["metrics"].get("peak_hbm_gib")]
    c["peak_hbm_gib"] = max(hbm, key=lambda h: h["max"]) if hbm else None
  return {
      "window": window,
      "skip": skip,
      "windows": windows,
      "eval_steps": eval_steps,
      "last_step": last,
      "short_runs": [r["run"] for c in configs for r in c["runs"]
                     if not r["loss"] or max(r["loss"]) < last],
      "configurations": [
          {k: v for k, v in c.items() if k != "runs"} | {"runs": [r["run"] for r in c["runs"]]}
          for c in configs],
      "_loss": [[r["loss"] for r in c["runs"]] for c in configs],
  }


def _fmt(x, spec: str) -> str:
  return "-" if x is None else format(x, spec)


def _hbm(peak: dict | None) -> str:
  """A run's peak HBM per GPU, or its range where the GPUs differ by 1 GiB or more."""
  if peak is None:
    return "-"
  if peak["max"] - peak["min"] < 1:
    return f"{peak['max']:.1f}"
  return f"{peak['min']:.1f} to {peak['max']:.1f}"


def _argument(key: str) -> str:
  return "GPUs" if key == "num_devices" else f"`--{key}`"


def markdown(report: dict) -> str:
  """The report as Markdown tables."""
  configs = report["configurations"]
  ref = configs[0]
  differs = sorted({k for c in configs for k in c["differs"] if k != "model"})
  also = "also " if any("model" in c["differs"] for c in configs) else ""
  lines = [f"Reference: `{ref['name']}`." + ("" if len(configs) == 1 else (
      " The configurations differ in `--model` alone." if not differs else
      f" The configurations {also}differ in " + ", ".join(map(_argument, differs)) + ":"))]

  def table(title, columns, rows):
    lines.extend(["", f"| {title} | " + " | ".join(columns) + " |",
                  "|---" * (len(columns) + 1) + "|"])
    lines.extend(f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows)

  def named(c):
    return f"`{c['name']}`" + (" (reference)" if c is ref else "")

  def against_reference(c, key):
    return [_fmt(v, ".4f") + ("" if c is ref or v is None or b is None else f" ({v - b:+.4f})")
            for v, b in zip(c[key], ref[key])]

  def repeats(key, n):
    values = [max((c[key][i] for c in configs if c[key][i] is not None), default=None)
              for i in range(n)]
    return [("Repeats differ by up to", [_fmt(v, ".4f") for v in values])] if any(
        v is not None for v in values) else []

  if differs:
    table("Configuration", [_argument(k) for k in differs],
          [(named(c), [f"`{json.dumps(c['differs'].get(k))}`" for k in differs])
           for c in configs])
  columns = [f"{lo}-{hi - 1}" for lo, hi in report["windows"]]
  table("Training loss, mean over steps", columns,
        [(named(c), against_reference(c, "window_loss")) for c in configs]
        + repeats("repeats_differ", len(columns)))
  if report["eval_steps"]:
    table("Validation loss, after step", [str(s) for s in report["eval_steps"]],
          [(named(c), against_reference(c, "validation")) for c in configs]
          + repeats("validation_repeats_differ", len(report["eval_steps"])))
  rows = []
  for c in configs:
    speed, base, spread = c["tokens_per_s_per_gpu"], ref["tokens_per_s_per_gpu"], c["speed_spread"]
    rows.append((named(c), [
        str(c["args"]["num_devices"]), str(len(c["runs"])), _fmt(speed, ",.0f"),
        "-" if spread is None else f"{100 * spread:.1f}%",
        _fmt(speed / base if speed and base else None, ".3f"), _hbm(c["peak_hbm_gib"])]))
  table("Speed", ["GPUs", "Runs", "Tokens/s per GPU", "Spread over runs", "Ratio to reference",
                  "Peak HBM per GPU (GiB)"], rows)
  if report["short_runs"]:
    lines.append(f"\nRuns that end before step {report['last_step']}: "
                 + ", ".join(f"`{r}`" for r in report["short_runs"]) + ".")
  return "\n".join(lines) + "\n"


def figure(report: dict, path) -> None:
  """Each configuration's loss, the mean of its repeats, against the step, with
  its mean over the window around each step, and the difference of that mean
  to the reference's."""
  from jax_titan.observability import plot  # pylint: disable=g-import-not-at-top

  plt = plot._pyplot()  # pylint: disable=protected-access
  window, configs = report["window"], report["configurations"]
  if len(configs) > 1:
    fig, (loss_ax, diff_ax) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2, 1]})
  else:
    fig, loss_ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)

  means, colors = [], []
  for c, losses in zip(configs, report["_loss"]):
    steps = sorted(set().union(*losses))
    line, = loss_ax.plot(
        steps, [statistics.fmean(loss[s] for loss in losses if s in loss) for s in steps],
        linewidth=0.5, alpha=0.35)
    colors.append(line.get_color())
    means.append(centred_means(losses, window))
    loss_ax.plot(list(means[-1]), list(means[-1].values()), color=colors[-1], linewidth=1.5,
                 label=c["name"])
  loss_ax.set(ylabel="training loss",
              title=f"Training loss, the mean of the repeats, and its mean over the {window} "
              "steps around each step")
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
  if args.json:
    out = {k: v for k, v in report.items() if not k.startswith("_")}
    pathlib.Path(args.json).write_text(json.dumps(out, indent=1) + "\n")
  if args.figure:
    figure(report, args.figure)
    print(f"\nwrote {args.figure}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
