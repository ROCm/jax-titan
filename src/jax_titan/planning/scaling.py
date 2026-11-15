"""Model quality from parameters and tokens, and the compute-optimal size: scaling laws.

  python -m jax_titan.planning.scaling --compute=8.4e24
  python -m jax_titan.planning.scaling --model=llama3.1-70b --tokens=20e12 --target_loss=2.0
  python -m jax_titan.planning.scaling --fit runs/a runs/b points.csv --out law.json
  python -m jax_titan.planning.scaling --fit runs/* --fix=alpha,beta --out law.json

A law predicts the final loss of a dense model with N parameters trained on D tokens,

  L(N, D) = E + A / N^alpha + B / D^beta        (Hoffmann et al. 2022, arXiv:2203.15556)

with C = 6ND training FLOPs. From it follow the loss of a planned run; the
compute-optimal model for a budget, which minimizes L along 6ND = C,

  N = G (C/6)^a and D = C / (6N), where G = (alpha A / (beta B))^(1 / (alpha + beta))
  and a = beta / (alpha + beta);

the tokens a model needs to reach a target loss; and the model that reaches it with the
least compute. A law's coefficients hold for the data, tokenizer and recipe it was fit on,
so its losses compare only with losses measured the same way:

- `chinchilla_refit` (the default): Besiroglu et al.'s 2024 refit of Chinchilla's runs
  (arXiv:2404.10102). Its optimum, about 20 tokens per parameter, is how Chinchilla 70B was
  trained and what Hoffmann et al.'s other two methods found.
- `chinchilla_2022`: Hoffmann et al.'s own fit (approach 3, unrounded), whose optimum is 50
  to 90 tokens per parameter from 1e23 to 1e26 FLOPs.

Both are in nats per token on MassiveText with a 32k vocabulary and count all parameters,
embeddings included. `--fit` fits a law to JAX-Titan runs or to (parameters, tokens, loss)
points as Hoffmann et al. did: a Huber loss on the log loss, minimized from a grid of starts.
"""

import argparse
import csv
import dataclasses
import itertools
import json
import math
import pathlib

from jax_titan.planning import models

TOKENS_PER_PARAM = 20  # Chinchilla's rule of thumb (Hoffmann et al. 2022, Table 3)
HUBER_DELTA = 1e-3
MASSIVETEXT = "nats per token on MassiveText with a 32k vocabulary"
FIT_KEYS = ("A", "B", "E", "alpha", "beta")


@dataclasses.dataclass(frozen=True)
class Law:
  """L(N, D) = E + A / N^alpha + B / D^beta, with N in parameters and D in tokens."""
  E: float
  A: float
  B: float
  alpha: float
  beta: float
  params: str = "total"  # "total" counts the embeddings, as Chinchilla does; "non_embedding" does not
  source: str = ""
  units: str = ""

  def loss(self, n: float, d: float) -> float:
    return self.E + self.A / n**self.alpha + self.B / d**self.beta

  def floor(self, n: float) -> float:
    """The loss of `n` parameters with unlimited tokens."""
    return self.E + self.A / n**self.alpha

  def optimal(self, compute: float) -> tuple[float, float]:
    """The (N, D) with the lowest loss for C = 6ND."""
    s = self.alpha + self.beta
    g = (self.alpha * self.A / (self.beta * self.B)) ** (1 / s)
    n = g * (compute / 6) ** (self.beta / s)
    return n, compute / (6 * n)

  def tokens_for(self, n: float, target: float) -> float | None:
    """Tokens `n` parameters need to reach `target`; None at or below their floor."""
    gap = target - self.floor(n)
    return (self.B / gap) ** (1 / self.beta) if gap > 0 else None

  def cheapest(self, target: float) -> tuple[float, float] | None:
    """The (N, D) that reach `target` with the least compute 6ND; None at or below E."""
    gap = target - self.E
    if gap <= 0:
      return None
    s = self.alpha + self.beta
    return ((self.A * s / (self.beta * gap)) ** (1 / self.alpha),
            (self.B * s / (self.alpha * gap)) ** (1 / self.beta))


LAWS = {
    "chinchilla_refit": Law(1.8172, 482.01, 2085.43, 0.3478, 0.3658, units=MASSIVETEXT,
                            source="Besiroglu et al. 2024, a refit of Chinchilla's runs"),
    "chinchilla_2022": Law(1.6934, 406.4, 410.7, 0.3392, 0.2849, units=MASSIVETEXT,
                           source="Hoffmann et al. 2022, approach 3"),
}
DEFAULT = "chinchilla_refit"


def load(spec: str) -> tuple[str, Law]:
  """A preset by name, or a law from a JSON file such as `--fit --out` writes."""
  if spec in LAWS:
    return spec, LAWS[spec]
  data = json.loads(pathlib.Path(spec).read_text())
  fields = {f.name for f in dataclasses.fields(Law)}
  return pathlib.Path(spec).name, Law(**{k: v for k, v in data.items() if k in fields})


def params_of(model: models.Model, law: Law) -> float:
  """The parameters a token uses, counted as `law` counts them."""
  n = model.params(active=True)
  return n - model.embedding_params() if law.params == "non_embedding" else n


def rule_of_thumb(compute: float) -> tuple[float, float]:
  """Chinchilla's 20 tokens per parameter at C = 6ND: N = sqrt(C / 120), D = 20N."""
  n = math.sqrt(compute / (6 * TOKENS_PER_PARAM))
  return n, TOKENS_PER_PARAM * n


def assess(model: models.Model, tokens: float, law: Law, name: str, *, custom: bool = False,
           target_loss: float | None = None, batch_tokens: float | None = None,
           critical_batch_tokens: float | None = None, epochs: float = 1.0) -> dict:
  """What `law` says about training `model` on `tokens`: the loss, the compute-optimal size
  for the same compute, and with a target loss the tokens to reach it and the cheapest
  model that does. The presets are fit to dense models, so they predict nothing for a
  mixture of experts unless `custom` says the law was fit for it."""
  n = params_of(model, law)
  compute = 6 * n * tokens
  rule_n, rule_d = rule_of_thumb(compute)
  out = {"law": name, "source": law.source, "units": law.units, "params": n, "tokens": tokens,
         "compute_6nd": compute, "dense": not model.moe_layers, "epochs": epochs,
         "rule_of_thumb": {"params": rule_n, "tokens": rule_d}}
  if out["dense"] or custom:
    opt_n, opt_d = law.optimal(compute)
    out.update(loss=law.loss(n, tokens), floor=law.floor(n),
               milestones={f: law.loss(n, tokens * f) for f in (0.5, 2.0)},
               optimal={"params": opt_n, "tokens": opt_d, "loss": law.loss(opt_n, opt_d)})
    if target_loss is not None:
      cheapest = law.cheapest(target_loss)
      out["target"] = {"loss": target_loss, "tokens": law.tokens_for(n, target_loss),
                       "cheapest": cheapest and {"params": cheapest[0], "tokens": cheapest[1],
                                                 "compute_6nd": 6 * cheapest[0] * cheapest[1]}}
  if batch_tokens and critical_batch_tokens:
    # Kaplan et al. 2020, Eq. 5.1: reaching a loss at batch B takes E_min (1 + B / B_crit) tokens.
    out["batch"] = {"tokens": batch_tokens, "critical": critical_batch_tokens,
                    "token_factor": 1 + batch_tokens / critical_batch_tokens}
  return out


def _count(x: float) -> str:
  for scale, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
    if x >= scale:
      return f"{x / scale:.3g}{unit}"
  return f"{x:.0f}"


def lines(a: dict) -> list[tuple[str, str]]:
  """`assess`'s result as (label, value) report lines."""
  out = [("law", f"{a['law']} ({a['source']})" if a["source"] else a["law"])]
  if a["units"]:
    out.append(("loss units", a["units"]))
  if "loss" in a:
    half, double = a["milestones"][0.5], a["milestones"][2.0]
    out.append(("loss", f"{a['loss']:.3f} at {_count(a['tokens'])} tokens ({half:.3f} at "
                f"{_count(a['tokens'] / 2)}, {double:.3f} at {_count(a['tokens'] * 2)})"))
    out.append(("loss floor", f"{a['floor']:.3f} with unlimited tokens"))
  else:
    out.append(("loss", "not predicted: the laws are fit to dense models (fit one to MoE runs "
                "with planning.scaling --fit)"))
  out.append(("compute (6ND)", f"{a['compute_6nd']:.3e} FLOPs over {_count(a['params'])} parameters"))
  if "optimal" in a:
    o = a["optimal"]
    out.append(("compute-optimal", f"{_count(o['params'])} parameters on {_count(o['tokens'])} "
                f"tokens ({o['tokens'] / o['params']:.0f} per parameter), loss {o['loss']:.3f}"))
  r = a["rule_of_thumb"]
  out.append(("20 tokens/parameter", f"{_count(r['params'])} parameters on {_count(r['tokens'])} "
              f"tokens" + ("" if a["dense"] else " (a dense rule, on active parameters)")))
  t = a.get("target")
  if t:
    out.append(("target loss", f"{t['loss']:.3f}: " + (
        f"{_count(t['tokens'])} tokens for this model" if t["tokens"] else
        f"out of reach for this model, whose floor is {a['floor']:.3f}")))
    c = t["cheapest"]
    out.append(("cheapest for target", f"{_count(c['params'])} parameters on {_count(c['tokens'])} "
                f"tokens, {c['compute_6nd']:.2e} FLOPs (6ND), x{c['compute_6nd'] / a['compute_6nd']:.2f} "
                "this run's" if c else "none: the target is at or below the law's irreducible loss"))
  b = a.get("batch")
  if b:
    out.append(("batch", f"{_count(b['tokens'])} tokens against a critical {_count(b['critical'])}: "
                f"x{b['token_factor']:.2f} the fewest tokens to a given loss (x2 at the critical batch)"))
  if a["epochs"] > 1:
    out.append(("repeated data", f"{a['epochs']:g} epochs; the laws assume every token is new, "
                "so the loss reads low"))
  return out


def _huber(r, np):
  a = np.abs(r)
  return np.where(a <= HUBER_DELTA, 0.5 * r * r, HUBER_DELTA * (a - 0.5 * HUBER_DELTA))


def fit(points, base: Law | None = None, fixed: tuple[str, ...] = (), units: str = "") -> Law:
  """The law that best fits [(N, D, loss)]: Hoffmann et al.'s Huber loss (delta 1e-3) on
  the log loss, minimized by L-BFGS from a grid of starts. Coefficients named in `fixed`
  keep `base`'s values, so that a few points can still fit the rest."""
  import numpy as np  # pylint: disable=g-import-not-at-top
  from scipy import optimize  # pylint: disable=g-import-not-at-top

  base = base or LAWS[DEFAULT]
  free = [k for k in FIT_KEYS if k not in fixed]
  if len(points) < len(free):
    raise ValueError(f"{len(points)} points cannot fit {len(free)} coefficients; fix some")
  n, d, loss = (np.asarray(v, dtype=float) for v in zip(*points))
  log_n, log_d, log_loss = np.log(n), np.log(d), np.log(loss)
  # A, B and E are fit as their logarithms, as Hoffmann et al. did.
  start = {"A": math.log(base.A), "B": math.log(base.B), "E": math.log(base.E),
           "alpha": base.alpha, "beta": base.beta}

  def objective(x):
    p = {**start, **dict(zip(free, x))}
    terms = np.stack([p["A"] - p["alpha"] * log_n, p["B"] - p["beta"] * log_d,
                      np.full_like(log_n, p["E"])])
    top = terms.max(axis=0)
    weights = np.exp(terms - top)
    total = weights.sum(axis=0)
    r = top + np.log(total) - log_loss
    weights /= total
    slope = np.clip(r, -HUBER_DELTA, HUBER_DELTA)
    grads = {"A": weights[0], "B": weights[1], "E": weights[2],
             "alpha": -weights[0] * log_n, "beta": -weights[1] * log_d}
    return float(_huber(r, np).sum()), np.array([(slope * grads[k]).sum() for k in free])

  grid = {"A": (0, 5, 10, 15, 20), "B": (0, 5, 10, 15, 20), "E": (-0.5, 0, 0.5, 1),
          "alpha": (0.1, 0.3, 0.5, 1.0), "beta": (0.1, 0.3, 0.5, 1.0)}
  bounds = [(0, 3) if k in ("alpha", "beta") else (None, None) for k in free]
  best = None
  for x0 in itertools.chain([[start[k] for k in free]], itertools.product(*(grid[k] for k in free))):
    res = optimize.minimize(objective, np.array(x0, dtype=float), jac=True, method="L-BFGS-B",
                            bounds=bounds)
    if best is None or res.fun < best.fun:
      best = res
  p = {**start, **dict(zip(free, best.x))}
  return Law(E=math.exp(p["E"]), A=math.exp(p["A"]), B=math.exp(p["B"]), alpha=float(p["alpha"]),
             beta=float(p["beta"]), params=base.params, units=units,
             source=f"fit to {len(points)} points" + (f", {', '.join(fixed)} fixed" if fixed else ""))


def points_from(paths, law: Law | None = None) -> list[tuple[float, float, float]]:
  """(N, D, loss) points from JAX-Titan run directories and from CSV or JSON files.

  A run gives its model's parameters, counted as `law` counts them; the tokens trained when
  its last validation pass ran; and that pass's loss, or without one the mean loss of its
  last 10 steps. Fit final losses of runs whose learning-rate schedule ended with them, as
  Hoffmann et al. did: a midway loss of a longer schedule reads high. A file holds rows of
  params, tokens and loss, as CSV with that header or as JSON lists."""
  from jax_titan.observability import records  # pylint: disable=g-import-not-at-top

  law = law or LAWS[DEFAULT]
  out = []
  for path in map(pathlib.Path, paths):
    if path.is_dir():
      log = records.read(path / "train.jsonl")
      args = records.events(log, "start")[0]["args"]
      if args.get("model_overrides"):
        raise ValueError(f"{path}: --model_overrides changed the model; give its point in a file")
      n = params_of(models.from_name(args["model"]), law)
      tokens_per_step = records.events(log, "layout")[-1]["tokens_per_step"]
      evals = records.events(log, "eval")
      if evals:
        step, loss = evals[-1]["step"], evals[-1]["loss"]
      else:
        steps = records.steps(log)
        step, loss = steps[-1]["step"] + 1, sum(s["loss"] for s in steps[-10:]) / len(steps[-10:])
      out.append((n, tokens_per_step * step, loss))
    elif path.suffix == ".csv":
      with open(path) as f:
        out += [(float(r["params"]), float(r["tokens"]), float(r["loss"])) for r in csv.DictReader(f)]
    else:
      out += [tuple(float(v) for v in row) for row in json.loads(path.read_text())]
  return out


def _metric(label: str, value: str) -> None:
  print(f"  {label + ':':<24}{value}")


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--law", default=DEFAULT,
                      help=f"{' or '.join(LAWS)}, or a JSON file such as --out writes.")
  parser.add_argument("--compute", type=float, help="Training FLOPs (6ND) to allocate.")
  parser.add_argument("--model", default="", help="A model the planner knows (planning.budget).")
  parser.add_argument("--tokens", type=float, help="Tokens to train --model on.")
  parser.add_argument("--target_loss", type=float)
  parser.add_argument("--fit", nargs="+", default=[],
                      help="Run directories, and CSV or JSON files of params, tokens and loss.")
  parser.add_argument("--fix", default="",
                      help="Coefficients the fit keeps from --law, e.g. alpha,beta.")
  parser.add_argument("--units", default="nats per token on the runs' data",
                      help="What the fitted losses measure.")
  parser.add_argument("--out", default="", help="Write the fitted law here, for --law.")
  args = parser.parse_args(argv)
  name, law = load(args.law)

  if args.fit:
    fixed = tuple(k for k in args.fix.split(",") if k)
    unknown = set(fixed) - set(FIT_KEYS)
    if unknown:
      parser.error(f"--fix takes {', '.join(FIT_KEYS)}, not {', '.join(sorted(unknown))}")
    points = points_from(args.fit, law)
    law = fit(points, law, fixed, args.units)
    name = args.out or "fit"
    print(f"L(N, D) = {law.E:.4f} + {law.A:.4g} / N^{law.alpha:.4f} + {law.B:.4g} / D^{law.beta:.4f}"
          f"  ({law.source}; {law.units})")
    print(f"  {'params':>10} {'tokens':>10} {'loss':>8} {'fit':>8}")
    for n, d, loss in points:
      print(f"  {_count(n):>10} {_count(d):>10} {loss:8.4f} {law.loss(n, d):8.4f}")
    if args.out:
      pathlib.Path(args.out).write_text(json.dumps(dataclasses.asdict(law), indent=1) + "\n")
    if not (args.compute or args.model):
      return 0

  print("--------- Scaling Law Report ---------")
  if args.model:
    model = models.from_name(args.model)
    tokens = args.tokens or (args.compute / (6 * params_of(model, law)) if args.compute else None)
    if tokens is None and args.target_loss:
      tokens = law.tokens_for(params_of(model, law), args.target_loss)
    if not tokens:
      parser.error("give --tokens, --compute or a --target_loss the model can reach")
    a = assess(model, tokens, law, name, custom=name not in LAWS, target_loss=args.target_loss)
    for label, value in lines(a):
      _metric(label, value)
    return 0
  if not args.compute:
    parser.error("give --compute, --model or --fit")
  opt_n, opt_d = law.optimal(args.compute)
  rule_n, rule_d = rule_of_thumb(args.compute)
  _metric("law", f"{name} ({law.source})" if law.source else name)
  _metric("compute (6ND)", f"{args.compute:.3e} FLOPs")
  _metric("compute-optimal", f"{_count(opt_n)} parameters on {_count(opt_d)} tokens "
          f"({opt_d / opt_n:.0f} per parameter), loss {law.loss(opt_n, opt_d):.3f}")
  _metric("20 tokens/parameter", f"{_count(rule_n)} parameters on {_count(rule_d)} tokens")
  if args.target_loss:
    c = law.cheapest(args.target_loss)
    _metric("cheapest for target", f"{_count(c[0])} parameters on {_count(c[1])} tokens, "
            f"{6 * c[0] * c[1]:.2e} FLOPs (6ND)" if c else "none: at or below the irreducible loss")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
