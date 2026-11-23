"""Pretrains a model with Shardy MPMD pipeline parallelism.

  python -m jax_titan.trainer.train --model=llama3.1-8b --num_stages=2

`--model` is a JAX-Titan Llama flavor, `ideogram4`, `command-a-plus`, FLUX.1
(`flux-dev`, `flux-schnell`), or a model of another repository; FLUX.1 and the
models of other repositories take `--parallel=dp` or `fsdp`:
`maxtext:<name>`, `maxdiffusion:<name>`, `transformers:<name>` or
`diffusers:<name>` (see `jax_titan.trainer.adapters` and docs/adapters).

`--parallel=dp` and `--parallel=fsdp` train the same model on the same batches
without the pipeline, as one `jax.jit` program over all devices.

With `--checkpoint_dir`, training resumes from the latest checkpoint there.
"""

import argparse
import dataclasses
import functools
import json
import os
import socket
import time

from jax_titan import paths
from jax_titan.trainer import adapters
from jax_titan.trainer.layout import EVAL_STEPS, Layout
from jax_titan.trainer.layout import Workload
from jax_titan.trainer.layout import abstract
from jax_titan.trainer.layout import placement as layout_placement

_NORM_PARAMS = ("attn_norm", "mlp_norm", "final_norm", "attn_out_norm", "mlp_out_norm", "q_norm",
                "k_norm", "text_norm")
# Matched against a parameter's own name only: MaxDiffusion's FLUX.1 keeps kernels under a
# module named `norm`.
_NORM_LEAVES = ("norm", "norm1", "norm2", "norm1_bias", "norm2_bias", "final_norm_bias")


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--model", default="llama3.1-8b",
      help="A Llama flavor (llama3.1-8b, llama3-test), ideogram4, command-a-plus, or, with "
      "--parallel=dp or fsdp, flux-dev, flux-schnell or a model of another repository: "
      "maxtext:<MaxText model name>, maxdiffusion:flux-dev, maxdiffusion:flux-schnell or "
      "maxdiffusion:wan2.1-t2v-1.3b, transformers:<directory or model type>, "
      "diffusers:<denoiser class or directory>. maxdiffusion:ideogram4 is the pipeline of "
      "MaxDiffusion's Ideogram branch.",
  )
  parser.add_argument(
      "--model_overrides", default="",
      help="JSON object of model config overrides for ideogram4, command-a-plus, FLUX.1 and "
      "for the models of other repositories, e.g. '{\"base_num_decoder_layers\": 4}'.",
  )
  parser.add_argument(
      "--ops", default="",
      help="Operator providers per op, e.g. 'attention=te,rms_norm=aiter:te' "
      "(see jax_titan.ops); 'stageN.op=...' overrides one stage. Default: "
      "native JAX everywhere.",
  )
  parser.add_argument(
      "--parallel", default="mpmd", choices=["mpmd", "dp", "fsdp"],
      help="mpmd: Shardy MPMD pipeline, data parallel inside each stage. "
      "dp, fsdp: no pipeline; one jax.jit program over all devices with "
      "replicated or sharded parameters and optimizer state. FLUX.1 and a model of "
      "another repository take dp or fsdp only.",
  )
  parser.add_argument(
      "--num_stages", type=int, default=None,
      help="Pipeline stages, 2 by default. dp and fsdp start from the same weights as the "
      "pipeline with this many stages. FLUX.1 and a model of another repository take none.",
  )
  parser.add_argument(
      "--schedule", default="ZERO_BUBBLE_H1",
      help="Shardy pipeline schedule name, e.g. ONE_FWD_ONE_BWD, GPIPE or "
      "CIRCULAR, or empty for the default order.",
  )
  parser.add_argument(
      "--stage_layers", default="",
      help="Comma-separated decoder layers per stage, e.g. 17,15; default "
      "even. dp and fsdp start from the weights of this split.",
  )
  parser.add_argument(
      "--stages_per_mesh", type=int, default=1,
      help="Pipeline stages on each mesh; above 1 for circular schedules, "
      "which put stage i on mesh i %% (num_stages / stages_per_mesh).",
  )
  parser.add_argument(
      "--mpmd_options", default="",
      help="JSON object of extra Shardy MPMD partitioning options, e.g. "
      '\'{"mpmd_merge_inferred_after_scheduling": true}\'.',
  )
  parser.add_argument(
      "--reduce_grads", default=None, choices=["microbatch", "step"],
      help="When mpmd stages sum gradients across their data-parallel ranks: "
      "in every microbatch's backward pass, or once per step from per-rank "
      "parameter copies. Default: step for Llama flavors, microbatch for "
      "the other models, which support only that.",
  )
  parser.add_argument(
      "--per_device", default="vmap", choices=["vmap", "shard_map"],
      help="With --reduce_grads=step, how each rank's parameter copy runs: "
      "vmap over the copies, or shard_map over the data axis, which also runs "
      "operators without a batching rule (JAX-AITER).",
  )
  parser.add_argument(
      "--stage_fsdp", action="store_true",
      help="mpmd: shard parameters and optimizer state across each stage's "
      "data-parallel ranks.",
  )
  parser.add_argument(
      "--tensor_parallel", type=int, default=1,
      help="Tensor-parallel degree, inside each stage for mpmd.",
  )
  parser.add_argument("--microbatches", type=int, default=8)
  parser.add_argument(
      "--microbatch_size", type=int, default=4,
      help="Sequences per microbatch, sharded across a stage's data-parallel "
      "ranks. dp and fsdp run all microbatches as one batch.",
  )
  parser.add_argument(
      "--seq_len", type=int, default=2048,
      help="Tokens per sequence. A cache of conversations (--data) sets its own.",
  )
  parser.add_argument(
      "--resolution", type=int, default=1024,
      help="Image models: image side in pixels; the model sees (resolution / 16)^2 "
      "latent tokens.",
  )
  parser.add_argument(
      "--frames", type=int, default=81,
      help="Video models: frames of a sample, of side --resolution; Wan sees "
      "(frames - 1) / 4 + 1 latent frames of (resolution / 16)^2 tokens each.",
  )
  parser.add_argument(
      "--text_tokens", type=int, default=0,
      help="Image models: text tokens per sample; default 512 for flux-dev and "
      "256 for flux-schnell and Ideogram 4. A cache of samples (--data) sets its own.",
  )
  parser.add_argument("--steps", type=int, default=50)
  parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate.")
  parser.add_argument("--warmup_steps", type=int, default=10)
  parser.add_argument(
      "--lr_schedule", default="cosine", choices=["cosine", "nemo_cosine"],
      help="cosine: optax's, from 0 at the first update to --lr over the warm-up, then a "
      "cosine to a tenth of it. nemo_cosine: NeMo's CosineAnnealing, as the MLPerf "
      "reference runs it: --lr * (s + 1) / (warm-up + 1) for update s up to the warm-up, "
      "then the same cosine (jax_titan.trainer.schedules).",
  )
  parser.add_argument(
      "--lr_decay_steps", type=int, default=0,
      help="Updates of the cosine after the warm-up; default: the rest of --steps. MLPerf "
      "fixes it at 1,200,000 minus the warm-up.",
  )
  parser.add_argument("--adam_b1", type=float, default=0.9, help="AdamW beta1.")
  parser.add_argument("--adam_b2", type=float, default=0.95, help="AdamW beta2.")
  parser.add_argument(
      "--adam_eps", type=float, default=1e-8,
      help="AdamW epsilon, added to the square root of the second moment.")
  parser.add_argument(
      "--weight_decay", type=float, default=0.1,
      help="AdamW's decoupled weight decay, for every parameter but the norms' weights.")
  parser.add_argument(
      "--max_grad_norm", type=float, default=1.0,
      help="Clip by the gradient norm over all stages together.",
  )
  parser.add_argument(
      "--data", default="",
      help="Token models: a uint32 token file (default: the WikiText-103 train file in "
      "Llama 3's tokens; a model with another tokenizer takes its own file, from "
      "python -m jax_titan.data.text --tokenizer), the prefix of a Megatron indexed "
      "dataset (PREFIX.bin and PREFIX.idx, read in the order Megatron's GPTDataset gives "
      "--seed, --seq_len and --steps; jax_titan.data.megatron) or "
      "'synthetic'. Image models train on synthetic batches; ideogram4 also on a cache "
      "directory that jax_titan.data.ideogram wrote. command-a-plus also takes a cache of "
      "conversations that jax_titan.data.command_a wrote.",
  )
  parser.add_argument("--remat", action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument(
      "--peak_tflops_per_device", type=float, default=2500.0,
      help="Dense BF16 peak used for MFU; 2500 for MI355X.",
  )
  parser.add_argument(
      "--log_file", default="",
      help="JSONL of step records and timeline events (jax_titan.observability.records).",
  )
  parser.add_argument(
      "--run_dir", default="",
      help="Writes train.jsonl (unless --log_file is given), env.json, power.jsonl and, at "
      "the end, results.json here, the ROCm scaling book's runs/<cell>/ layout; see "
      "docs/metrics.md.",
  )
  parser.add_argument(
      "--eval_every", type=int, default=0,
      help="Steps between validation passes, and one after the last step; 0 turns them off.",
  )
  parser.add_argument(
      "--eval_batches", type=int, default=4,
      help="Batches of --eval_batch_size per validation pass, the same every pass.",
  )
  parser.add_argument(
      "--eval_samples", type=int, default=0,
      help="Token models: sequences per validation pass, the same ones every pass, in "
      "batches of --eval_batch_size; default --eval_batches batches.",
  )
  parser.add_argument(
      "--eval_batch_size", type=int, default=0,
      help="Token models: sequences per validation batch, a multiple of the data-parallel "
      "ranks; default one step's batch, the only size a pipeline takes.",
  )
  parser.add_argument(
      "--target_eval_loss", type=float, default=0.0,
      help="Stop after the first validation whose loss is at or below this, as MLPerf runs "
      "stop at their quality target; 0 trains all --steps.",
  )
  parser.add_argument(
      "--synthetic_warmup", type=int, default=0,
      help="Llama flavors: train this many steps, and validate once, on synthetic tokens "
      "before the run starts, so both programs are compiled and warm; the state is then "
      "initialized (or restored) again. The run_start event, where MLPerf's clock starts, "
      "follows it, and the first tokens of a Megatron dataset are read after it.",
  )
  parser.add_argument(
      "--eval_data", default="",
      help="uint32 token file to validate on (default: WikiText-103's validation split in "
      "Llama 3's tokens; a model with another tokenizer takes its own file), or the prefix "
      "of a Megatron indexed dataset, whose first sequences in file order are validated on. "
      "Synthetic training data validates on held-out synthetic batches, and a cache on the "
      "samples it holds out.",
  )
  parser.add_argument(
      "--power_sample_s", type=float, default=1.0,
      help="With --run_dir, seconds between GPU power, energy, clock and throttle samples "
      "into power.jsonl; 0 turns sampling off.",
  )
  parser.add_argument(
      "--rccl_inspector", action="store_true",
      help="Load the RCCL Inspector (scripts/build_rccl_inspector.sh, or "
      "$JAX_TITAN_RCCL_INSPECTOR) to record every collective into <run_dir>/rccl_inspector.",
  )
  parser.add_argument(
      "--checkpoint_dir", default="",
      help="Saves training state here, and resumes from its latest checkpoint.",
  )
  parser.add_argument(
      "--init_from", default="",
      help="maxtext: models start from this MaxText parameter checkpoint, e.g. "
      "pretrained weights converted with MaxText's checkpoint_conversion/to_maxtext.py, "
      "transformers: models from a directory of published weights, "
      "and ideogram4 and command-a-plus from published weights (a directory of safetensors "
      "files) or from parameters that --save_params wrote, instead of a random initialization. "
      "A checkpoint in --checkpoint_dir wins for the trained parameters; the parameters a model "
      "keeps frozen are not checkpointed and are read from here again, so a run that resumes "
      "keeps this flag.",
  )
  parser.add_argument(
      "--save_params", default="",
      help="Writes the model's parameters here after the last step, as --init_from reads "
      "them (ideogram4, command-a-plus). A model with frozen parameters writes the trained "
      "ones alone, which command_a_generate --params reads and --init_from does not. The "
      "directory must not exist.",
  )
  parser.add_argument(
      "--checkpoint_every", type=int, default=0,
      help="Steps between checkpoints; 0 saves only after the last step.",
  )
  parser.add_argument(
      "--profile_steps", default="",
      help="START:END traces steps [START, END) with jax.profiler into "
      "--profile_dir.",
  )
  parser.add_argument(
      "--profile_dir", default=str(paths.work_dir() / "profiles"),
  )
  parser.add_argument(
      "--step_timeout", type=float, default=300.0,
      help="Exit when the first step runs longer than this many seconds, or a "
      "later step longer than this or 10x the median step; 0 turns it off.",
  )
  parser.add_argument(
      "--fail_at_step", type=int, default=0,
      help="Exit at once with code 75 after this many steps, without finishing saves or "
      "logging an end event, to rehearse a restart (ETTR, restart cost, lost work).",
  )
  parser.add_argument("--platform", default="", help="'cpu' runs on 8 CPU devices.")
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args(argv)
  if args.num_stages is None:
    args.num_stages = adapters.default_num_stages(args.model)
  return args


def _decay_mask(params):
  """Weight decay for every parameter but the norms' (MaxText's and
  MaxDiffusion's norms hold their weight as `scale`)."""
  import jax  # pylint: disable=g-import-not-at-top

  def decays(path, _):
    keys = [getattr(k, "key", None) for k in path]
    norm = any(k in _NORM_PARAMS or k == "scale" for k in keys) or keys[-1] in _NORM_LEAVES
    return not norm

  return jax.tree_util.tree_map_with_path(decays, params)


def _stage_layers(args):
  return tuple(int(n) for n in args.stage_layers.split(",")) if args.stage_layers else None


def _build_mpmd(args, cfg, devices, optimizer) -> Layout:
  """Shardy MPMD pipeline: one mesh per stage, DP, FSDP and TP inside each."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan import ops  # pylint: disable=g-import-not-at-top
  from jax_titan import parallel  # pylint: disable=g-import-not-at-top
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top
  from shardy.integrations.python.jax import mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  num_stages = args.num_stages
  num_meshes = num_stages // args.stages_per_mesh
  topology = jt_mpmd.make_topology(
      num_meshes, devices, tensor_parallel=args.tensor_parallel
  )
  options = json.loads(args.mpmd_options) if args.mpmd_options else None
  config = jt_mpmd.make_config(
      topology, args.schedule or None, options, num_stages=num_stages
  )
  meshes = [topology[jt_mpmd.mesh_name(i % num_meshes)] for i in range(num_stages)]
  data_parallel = meshes[0].shape[jt_mpmd.DATA_AXIS]
  if args.microbatch_size % data_parallel:
    raise ValueError(
        f"--microbatch_size={args.microbatch_size} does not split across "
        f"{data_parallel} data-parallel ranks per stage"
    )
  activation_sharding = NamedSharding(meshes[0], P(jt_mpmd.DATA_AXIS))

  def constrain(h):
    return jax.lax.with_sharding_constraint(h, activation_sharding)

  per_step = args.reduce_grads == "step"
  stage_fns = []
  for i in range(num_stages):
    fn = functools.partial(
        llama.stage_forward, cfg=cfg, stage=i, num_stages=num_stages,
        remat=args.remat, constrain=None if per_step else constrain,
    )
    if i < num_stages - 1:
      fn = functools.partial(lambda f, p, x: f(p, x, None), fn)
    stage_fns.append(ops.for_stage(fn, i))

  layers = _stage_layers(args)

  def init_stage(i, key):
    params = llama.init_stage_params(
        cfg, key, stage=i, num_stages=num_stages, layers=layers
    )
    return params, optimizer.init(params)

  keys = jax.random.split(jax.random.key(args.seed), num_stages)
  shapes = [
      jax.eval_shape(functools.partial(init_stage, i), keys[i])
      for i in range(num_stages)
  ]
  specs = [
      parallel.tree_specs(
          shapes[i], llama.logical_axes, meshes[i].shape, fsdp=args.stage_fsdp
      )
      for i in range(num_stages)
  ]
  shardings = [
      jax.tree.map(lambda s, m=meshes[i]: NamedSharding(m, s), specs[i])
      for i in range(num_stages)
  ]

  def init():
    states = [
        jax.jit(functools.partial(init_stage, i), out_shardings=shardings[i])(keys[i])
        for i in range(num_stages)
    ]
    return tuple(s[0] for s in states), tuple(s[1] for s in states)

  param_shardings = tuple(s[0] for s in shardings)
  opt_shardings = tuple(s[1] for s in shardings)
  abstract_state = (
      tuple(abstract(shapes[i][0], param_shardings[i]) for i in range(num_stages)),
      tuple(abstract(shapes[i][1], opt_shardings[i]) for i in range(num_stages)),
  )

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  token_sharding = NamedSharding(meshes[0], P(None, jt_mpmd.DATA_AXIS, None))
  target_sharding = NamedSharding(meshes[-1], P(None, jt_mpmd.DATA_AXIS, None))
  loss_sharding = NamedSharding(meshes[-1], P())

  train_step = jt_mpmd.make_train_step(
      stage_fns, optimizer, max_grad_norm=args.max_grad_norm, mesh=meshes[0],
      copy_axes=[llama.device_copy_axes(s[0]) for s in shapes] if per_step else None,
      param_specs=[s[0] for s in specs], per_device=args.per_device,
  )
  jitted = mpmd.jit(
      train_step,
      config,
      in_shardings=(param_shardings, opt_shardings, token_sharding, target_sharding),
      out_shardings=(param_shardings, opt_shardings, loss_sharding, loss_sharding),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    batch = batch.reshape(num_mb, mb, seq + 1)
    return (
        jax.device_put(batch[..., :-1], token_sharding),
        jax.device_put(batch[..., 1:], target_sharding),
    )

  eval_step = mpmd.jit(
      functools.partial(jt_mpmd.pipeline_loss, stage_fns),
      jt_mpmd.make_config(topology, None, options, num_stages=num_stages),
      in_shardings=(param_shardings, token_sharding, target_sharding),
      out_shardings=loss_sharding,
  )
  batch_spec = jax.ShapeDtypeStruct((num_mb, mb, seq), jnp.int32)
  return Layout(
      init, abstract_state, jitted, batch_spec, batch_spec, put_batch, f"mpmd:{num_stages}",
      placement=layout_placement(
          topology, [jt_mpmd.mesh_name(i % num_meshes) for i in range(num_stages)]),
      eval_step=eval_step,
  )


def _build_spmd(args, cfg, devices, optimizer) -> Layout:
  """No pipeline: one `jax.jit` program over all devices."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan import parallel  # pylint: disable=g-import-not-at-top
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  NamedSharding = jax.sharding.NamedSharding  # pylint: disable=invalid-name

  tp = args.tensor_parallel
  mesh = jax.sharding.Mesh(
      np.asarray(devices).reshape(len(devices) // tp, tp),
      (parallel.DATA_AXIS, parallel.TENSOR_AXIS),
  )
  batch_sharding = NamedSharding(mesh, P(parallel.DATA_AXIS))

  layers = _stage_layers(args)

  def init_params(keys):
    return llama.merge_stage_params([
        llama.init_stage_params(
            cfg, keys[i], stage=i, num_stages=args.num_stages, layers=layers
        )
        for i in range(args.num_stages)
    ])

  def init_state(keys):
    params = init_params(keys)
    return params, optimizer.init(params)

  keys = jax.random.split(jax.random.key(args.seed), args.num_stages)
  shapes = jax.eval_shape(init_state, keys)
  shardings = jax.tree.map(
      lambda s: NamedSharding(mesh, s),
      parallel.tree_specs(
          shapes, llama.logical_axes, mesh.shape, fsdp=args.parallel == "fsdp"
      ),
  )

  def init():
    return jax.jit(init_state, out_shardings=shardings)(keys)

  loss_fn = functools.partial(
      llama.stage_forward, cfg=cfg, stage=0, num_stages=1, remat=args.remat,
      constrain=lambda h: jax.lax.with_sharding_constraint(h, batch_sharding),
  )

  def train_step(params, opt_state, tokens, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens, targets)
    norm = jt_mpmd.global_grad_norm([grads])
    params, opt_state = jt_mpmd.clipped_update(
        optimizer, params, grads, opt_state, norm, args.max_grad_norm)
    return params, opt_state, jnp.where(jnp.isfinite(norm), loss, jnp.nan), norm

  param_shardings, opt_shardings = shardings
  scalar = NamedSharding(mesh, P())
  jitted = jax.jit(
      train_step,
      in_shardings=(param_shardings, opt_shardings, batch_sharding, batch_sharding),
      out_shardings=(param_shardings, opt_shardings, scalar, scalar),
      donate_argnums=(0, 1),
  )

  def put_batch(batch):
    return (
        jax.device_put(batch[:, :-1], batch_sharding),
        jax.device_put(batch[:, 1:], batch_sharding),
    )

  eval_step = jax.jit(
      loss_fn, in_shardings=(param_shardings, batch_sharding, batch_sharding),
      out_shardings=scalar,
  )
  batch = args.microbatches * args.microbatch_size
  batch_spec = jax.ShapeDtypeStruct((batch, args.seq_len), jnp.int32)
  return Layout(
      init, abstract(shapes, shardings), jitted, batch_spec, batch_spec, put_batch, "spmd",
      placement=layout_placement({"mesh": mesh}, ["mesh"]), eval_step=eval_step,
  )


def _mkdir(*parts):
  import pathlib  # pylint: disable=g-import-not-at-top

  path = pathlib.Path(*parts)
  path.mkdir(parents=True, exist_ok=True)
  return path


def step_programs(lowered) -> dict[str, int]:
  """The compiled programs one step runs, by HLO module name, and how often:
  each MPMD fragment once per call in the IFRT program, or the one jit program."""
  from jax_titan.observability import gemms  # pylint: disable=g-import-not-at-top

  return gemms.program_calls(lowered.as_text())


def _enable_rccl_inspector(run_dir: str):
  """Loads the RCCL Inspector into RCCL, which reads these before JAX starts it."""
  plugin = os.environ.get("JAX_TITAN_RCCL_INSPECTOR") or str(
      paths.work_dir() / "lib/librccl-profiler-inspector.so")
  if not os.path.exists(plugin):
    raise FileNotFoundError(f"{plugin} is missing; build it with scripts/build_rccl_inspector.sh")
  out = os.path.join(run_dir or ".", "rccl_inspector")
  os.makedirs(out, exist_ok=True)
  os.environ.setdefault("NCCL_PROFILER_PLUGIN", plugin)
  os.environ.setdefault("NCCL_INSPECTOR_ENABLE", "1")
  os.environ.setdefault("NCCL_INSPECTOR_DUMP_DIR", out)
  os.environ.setdefault("NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS", "1000000")


def main(argv=None, listeners=()):
  """Trains; `listeners` get every step record and event as the log is written."""
  main_start = time.time()
  args = parse_args(argv)
  if args.platform == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
  if args.run_dir:
    os.makedirs(args.run_dir, exist_ok=True)
  if args.rccl_inspector:
    _enable_rccl_inspector(args.run_dir)

  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top

  if args.platform == "cpu":
    jax.config.update("jax_num_cpu_devices", 8)

  import jax_titan  # noqa: F401  pylint: disable=g-import-not-at-top,unused-import
  from jax_titan import ops  # pylint: disable=g-import-not-at-top
  from jax_titan.data import megatron  # pylint: disable=g-import-not-at-top
  from jax_titan.data.text import WindowSampler  # pylint: disable=g-import-not-at-top
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top
  from jax_titan.observability import records  # pylint: disable=g-import-not-at-top
  from jax_titan.trainer import schedules  # pylint: disable=g-import-not-at-top

  devices = jax.devices()
  jax_ready = time.time()
  if args.run_dir and not args.log_file:
    index = jax.process_index()
    args.log_file = os.path.join(args.run_dir, f"train.p{index}.jsonl" if index else "train.jsonl")
  log = records.Log(args.log_file, listeners)
  launch = os.environ.get("JAX_TITAN_LAUNCH_TIME")
  log.event(
      "start", launch_time=float(launch) if launch else None,
      process_start=records.process_start_time(), main_start=main_start, jax_ready=jax_ready,
      host=socket.gethostname(), pid=os.getpid(), process_index=jax.process_index(),
      process_count=jax.process_count(), num_devices=len(devices),
      device_kind=devices[0].device_kind, jax_version=jax.__version__, args=vars(args),
  )

  ops.set_policy(args.ops)
  # A staged model, which jax_titan.trainer.adapters builds a layout for.
  staged = ":" in args.model or args.model in adapters.MODELS
  if args.reduce_grads is None:
    args.reduce_grads = "microbatch" if staged else "step"
  tp = f", tensor parallel {args.tensor_parallel}" if args.tensor_parallel > 1 else ""
  if args.parallel == "mpmd":
    description = (
        f"{args.num_stages} stages{' with FSDP' if args.stage_fsdp else ''}{tp} | "
        f"schedule={args.schedule or 'default'} | "
        f"gradients reduced per {args.reduce_grads}"
    )
    if args.reduce_grads == "step" and args.per_device != "vmap":
      description += f" (copies via {args.per_device})"
    if args.mpmd_options:
      description += f" | {args.mpmd_options}"
  else:
    description = f"{args.parallel} over {len(devices)} devices{tp}, no pipeline"
  if args.ops:
    description += f" | ops {args.ops}"

  decay_steps = (args.lr_decay_steps
                 or max(args.steps, args.warmup_steps + 1) - args.warmup_steps)
  lr = schedules.SCHEDULES[args.lr_schedule](args.lr, args.warmup_steps, decay_steps,
                                             args.lr * 0.1)

  def make_optimizer(decays=None):
    """AdamW; `decays(path, leaf)` says which parameters take weight decay."""
    mask = _decay_mask if decays is None else (
        lambda params: jax.tree_util.tree_map_with_path(decays, params))
    # Each step clips by the global gradient norm (--max_grad_norm) before this.
    return optax.adamw(lr, b1=args.adam_b1, b2=args.adam_b2, eps=args.adam_eps,
                       weight_decay=args.weight_decay, mask=mask)

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  eval_batch = args.eval_batch_size or num_mb * mb
  eval_count = args.eval_samples or args.eval_batches * eval_batch
  if args.target_eval_loss and not args.eval_every:
    raise ValueError("--target_eval_loss needs validation passes: set --eval_every")
  if eval_count % eval_batch:
    raise ValueError(f"--eval_samples={eval_count} is not whole batches of {eval_batch}")
  if (args.eval_samples or args.eval_batch_size) and staged:
    raise NotImplementedError("--eval_samples and --eval_batch_size are for Llama flavors")
  if eval_batch != num_mb * mb and args.parallel == "mpmd":
    raise ValueError(f"a pipeline validates batches of one step's {num_mb * mb} sequences, "
                     f"not --eval_batch_size={eval_batch}")
  if args.synthetic_warmup and staged:
    raise NotImplementedError("--synthetic_warmup is for Llama flavors")
  if staged:
    layout, work = adapters.build(args, devices, make_optimizer)
  else:
    if args.init_from:
      raise NotImplementedError(
          "--init_from loads parameters of maxtext: and transformers: models, ideogram4 and "
          "command-a-plus")
    cfg = llama.FLAVORS[args.model]
    build = _build_mpmd if args.parallel == "mpmd" else _build_spmd
    layout = build(args, cfg, devices, make_optimizer())
    from jax_titan.observability import flops as flops_lib  # pylint: disable=g-import-not-at-top

    tokens = num_mb * mb * seq
    work = Workload(
        num_params=cfg.num_params(),
        tokens_per_step=tokens,
        flops_per_step=cfg.flops_per_token(seq) * tokens,
        vocab_size=cfg.vocab_size,
        hardware_flops_per_step=(cfg.flops_per_token(seq) + flops_lib.recompute_flops_per_token(
            cfg, seq, args.remat)) * tokens,
    )
  print(
      f"jax {jax.__version__} | {len(devices)} x {devices[0].device_kind} | "
      f"{args.model} ({work.num_params / 1e9:.2f}B params) | {description}",
      flush=True,
  )
  if layout.frozen:
    trained = sum(x.size for x in jax.tree.leaves(layout.abstract_state[0]))
    print(f"{trained / 1e9:.2f}B of the parameters are trained, the others are frozen", flush=True)
  log.event(
      "layout", structure=layout.structure, num_params=work.num_params,
      tokens_per_step=work.tokens_per_step, flops_per_step=work.flops_per_step,
      hardware_flops_per_step=work.hardware_flops_per_step, placement=layout.placement,
  )
  if args.run_dir and jax.process_index() == 0:
    from jax_titan.observability import env as env_lib  # pylint: disable=g-import-not-at-top

    env_lib.write(os.path.join(args.run_dir, "env.json"), args=args, devices=devices,
                  placement=layout.placement)
  if args.eval_every and layout.eval_step is None:
    raise ValueError(f"{layout.structure} has no validation step")
  if args.save_params:
    if layout.export is None:
      raise NotImplementedError(
          "--save_params writes the parameters of ideogram4 and command-a-plus only")
    if os.path.exists(args.save_params):
      raise FileExistsError(f"--save_params={args.save_params} exists")

  # Frozen parameters are not checkpointed; a run that resumes reads them again.
  frozen_from = os.path.abspath(args.init_from) if args.init_from else f"seed {args.seed}"
  ckpt = None
  start = 0
  if args.checkpoint_dir:
    from jax_titan.trainer import checkpoint  # pylint: disable=g-import-not-at-top

    ckpt = checkpoint.Checkpointer(args.checkpoint_dir, save_every=args.checkpoint_every)
    start = ckpt.latest_step() or 0
  t0 = time.time()
  if start:
    (params, opt_states), meta = ckpt.restore(start, layout.abstract_state)
    if meta["structure"] != layout.structure or meta["model"] != args.model:
      raise ValueError(
          f"checkpoint holds {meta['model']} as {meta['structure']}, not "
          f"{args.model} as {layout.structure}"
      )
    if layout.frozen and meta.get("frozen_from") != frozen_from:
      raise ValueError(
          f"the checkpoint's run took its frozen parameters from {meta.get('frozen_from')}, "
          f"and this one would take them from {frozen_from}: resume with the same --init_from "
          "and --seed"
      )
    state_s = time.time() - t0
    print(f"restored step {start} from {args.checkpoint_dir} in {state_s:.1f}s", flush=True)
  else:
    params, opt_states = layout.init()
    state_s = time.time() - t0
    print(f"initialized params and optimizer state in {state_s:.1f}s", flush=True)
  log.event("state", seconds=state_s, restored_step=start)
  frozen = (layout.frozen(),) if layout.frozen else ()

  t0 = time.time()
  ops.reset_selections()
  lowered = layout.step.lower(params, opt_states, *frozen, layout.input_spec, layout.target_spec)
  t1 = time.time()
  compiled = lowered.compile()
  t2 = time.time()
  print(f"lowered in {t1 - t0:.1f}s, compiled in {t2 - t1:.1f}s", flush=True)
  log.event("compile", lower_s=t1 - t0, compile_s=t2 - t1, programs=step_programs(lowered))
  selected = [dataclasses.asdict(s) for s in ops.selections()]
  print("ops:\n  " + ops.summary().replace("\n", "\n  "), flush=True)

  t0 = time.time()
  eval_batches = []
  if work.next_batch is not None:
    next_batch = work.next_batch
    print(f"data: {work.data}", flush=True)
  elif args.data == "synthetic":

    def next_batch(step):
      rng = np.random.default_rng((args.seed, step))
      return rng.integers(0, work.vocab_size, (num_mb * mb, seq + 1), dtype=np.int32)

  else:
    path = args.data or str(paths.work_dir() / "data/wikitext103_train.llama3.bin")
    prefix = megatron.prefix_of(path)
    if prefix:
      # Megatron's order for a run of --steps global batches. The tokens are read batch by
      # batch, so their ids are checked batch by batch too.
      sampler = megatron.Sampler(prefix, seq, num_samples=args.steps * num_mb * mb,
                                 seed=args.seed, step=start)
      print(f"data: {prefix} (Megatron indexed dataset, {sampler.num_tokens / 1e6:.1f}M "
            f"tokens, {len(sampler.samples)} samples of {seq + 1})", flush=True)
    else:
      sampler = WindowSampler(path, seq, seed=args.seed, step=start)
      print(f"data: {path} ({sampler.num_tokens / 1e6:.1f}M tokens)", flush=True)
      if sampler.max_token() >= work.vocab_size:
        raise ValueError(
            f"{path} has token ids up to {sampler.max_token()}, beyond "
            f"{args.model}'s vocabulary of {work.vocab_size}"
        )

    def next_batch(step):
      assert sampler.step == step, (sampler.step, step)
      batch = sampler.next(num_mb * mb)
      if prefix and batch.max() >= work.vocab_size:
        raise ValueError(f"{prefix} has token ids up to {batch.max()} in step {step}, beyond "
                         f"{args.model}'s vocabulary of {work.vocab_size}")
      return batch

  load_eval = None
  if args.eval_every:
    num_eval = eval_count // eval_batch
    if work.next_batch is not None:
      eval_batches = [next_batch(EVAL_STEPS + i) for i in range(num_eval)]
    elif args.data == "synthetic":
      eval_batches = [
          np.random.default_rng((args.seed, EVAL_STEPS + i)).integers(
              0, work.vocab_size, (eval_batch, seq + 1), dtype=np.int32)
          for i in range(num_eval)]
    else:
      eval_path = args.eval_data or str(
          paths.work_dir() / "data/wikitext103_validation.llama3.bin")
      if megatron.prefix_of(eval_path):
        # The index is read here; the tokens at the first validation, after run_start.
        held_out = megatron.validation_samples(megatron.prefix_of(eval_path), seq, eval_count)

        def load_eval():
          tokens = held_out.batch(0, eval_count)
          if tokens.max() >= work.vocab_size:
            raise ValueError(f"{eval_path} has token ids beyond the vocabulary")
          return list(tokens.reshape(num_eval, eval_batch, seq + 1))
      else:
        held_out = WindowSampler(eval_path, seq, seed=args.seed, step=0)
        if held_out.max_token() >= work.vocab_size:
          raise ValueError(f"{eval_path} has token ids beyond the vocabulary")
        eval_batches = [held_out.next(eval_batch) for _ in range(num_eval)]
  log.event("data", seconds=time.time() - t0)

  def meta(step):
    return {
        "step": step, "model": args.model, "structure": layout.structure,
        "data": {"seed": args.seed, "step": step}, "ops": selected,
        **({"frozen_from": frozen_from} if layout.frozen else {}),
    }

  profile = [int(s) for s in args.profile_steps.split(":")] if args.profile_steps else None
  tokens_per_step, flops_per_step = work.tokens_per_step, work.flops_per_step
  peak_flops = len(devices) * args.peak_tflops_per_device * 1e12
  watchdog = None
  if args.step_timeout > 0:
    from jax_titan.observability import watchdog as wd  # pylint: disable=g-import-not-at-top

    watchdog = wd.StepWatchdog(args.step_timeout)
  power_sampler = None
  if args.run_dir and args.power_sample_s > 0 and jax.process_index() == 0:
    from jax_titan.observability import power  # pylint: disable=g-import-not-at-top

    power_sampler = power.Sampler(
        os.path.join(args.run_dir, "power.jsonl"), args.power_sample_s).start()

  compiled_eval = None
  if args.synthetic_warmup:
    t0 = time.time()
    rng = np.random.default_rng(0)

    def synthetic(batch):
      return layout.put_batch(rng.integers(0, work.vocab_size, (batch, seq + 1), dtype=np.int32))

    for _ in range(args.synthetic_warmup):
      params, opt_states, _, _ = compiled(params, opt_states, *frozen, *synthetic(num_mb * mb))
    if load_eval or eval_batches:
      placed = synthetic(eval_batch)
      compiled_eval = layout.eval_step.lower(params, *frozen, *placed).compile()
      jax.block_until_ready(compiled_eval(params, *frozen, *placed))
    jax.block_until_ready((params, opt_states))
    # The warm-up changed the state: start again from the initial or restored one.
    params = opt_states = None
    if start:
      (params, opt_states), _ = ckpt.restore(start, layout.abstract_state)
    else:
      params, opt_states = layout.init()
    print(f"warmed up on {args.synthetic_warmup} synthetic steps in {time.time() - t0:.1f}s",
          flush=True)
    log.event("warmup", seconds=time.time() - t0, steps=args.synthetic_warmup)

  log.event("run_start", step=start)
  status, done, reached = "completed", start, False
  try:
    for step in range(start, args.steps):
      if profile and step == profile[0]:
        jax.block_until_ready((params, opt_states))
        jax.profiler.start_trace(args.profile_dir)
      tokens, targets = layout.put_batch(next_batch(step))
      t0 = time.time()
      if watchdog:
        watchdog.start(step)
      with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
        params, opt_states, loss, grad_norm = compiled(
            params, opt_states, *frozen, tokens, targets)
        loss, grad_norm = float(loss), float(grad_norm)
      if watchdog:
        watchdog.stop()
      dt = time.time() - t0
      if profile and step == profile[1] - 1:
        jax.block_until_ready((params, opt_states))
        jax.profiler.stop_trace()
        print(f"traced steps {profile[0]}-{step} into {args.profile_dir}", flush=True)
        if args.run_dir:
          from jax_titan.observability import profile as profile_lib  # pylint: disable=g-import-not-at-top

          profile_lib.write_json(args.profile_dir, profile[1] - profile[0],
                                 os.path.join(args.run_dir, "profile.json"))
      record = {
          "step": step,
          "loss": loss,
          "grad_norm": grad_norm,
          "step_time_s": dt,
          "tokens_per_s": tokens_per_step / dt,
          "tokens_per_s_per_gpu": tokens_per_step / dt / len(devices),
          **({"mfu": flops_per_step / dt / peak_flops} if flops_per_step else {}),
          "time": time.time(),
      }
      print(
          f"step {step:4d} | loss {loss:8.4f} | grad norm {grad_norm:8.3f} | {dt * 1e3:8.1f} ms | "
          f"{record['tokens_per_s']:10.0f} tok/s"
          + (f" | MFU {100 * record['mfu']:5.1f}%" if flops_per_step else ""),
          flush=True,
      )
      log.write(record)
      if step == start:
        peaks = [
            (d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30
            for d in devices
        ]
        print("peak HBM per device (GiB): " + " ".join(f"{p:.1f}" for p in peaks), flush=True)
        log.event("memory", step=step, peak_hbm_gib=peaks)
      if not np.isfinite(loss):
        raise FloatingPointError(
            f"non-finite loss or gradient norm ({grad_norm}) at step {step}; the step "
            "made no update, and nothing was saved after it")
      done = step + 1
      if (eval_batches or load_eval) and (done % args.eval_every == 0 or done == args.steps):
        t0 = time.time()
        if load_eval:
          eval_batches, load_eval = load_eval(), None
        placed = [layout.put_batch(b) for b in eval_batches]
        compile_s = 0.0
        if compiled_eval is None:
          compiled_eval = layout.eval_step.lower(params, *frozen, *placed[0]).compile()
          compile_s = time.time() - t0
        t1 = time.time()
        # Every batch has eval_batch sequences of seq tokens, so the mean of the batch means
        # is the mean loss over every validation token.
        eval_loss = float(np.mean([float(compiled_eval(params, *frozen, x, t)) for x, t in placed]))
        eval_s = time.time() - t1
        reached = bool(args.target_eval_loss) and eval_loss <= args.target_eval_loss
        print(f"step {done:4d} | validation loss {eval_loss:.4f} over {len(placed)} batches "
              f"in {eval_s:.1f}s" + (f" | at or below {args.target_eval_loss}" if reached else ""),
              flush=True)
        log.event("eval", step=done, loss=eval_loss, seconds=time.time() - t0,
                  compile_s=compile_s, samples=done * num_mb * mb, eval_samples=eval_count)
      if ckpt:
        t0 = time.time()
        force = step + 1 == args.steps or reached
        if ckpt.save(step + 1, (params, opt_states), meta(step + 1), force=force):
          blocking = time.time() - t0
          print(f"saved step {step + 1} in {blocking:.1f}s", flush=True)
          log.event("save", step=step + 1, blocking_s=blocking)
      if done == args.fail_at_step:
        print(f"--fail_at_step: exiting after step {step}", flush=True)
        os._exit(75)
      if reached:
        break
    if ckpt:
      t0 = time.time()
      ckpt.close()
      log.event("checkpoint_closed", seconds=time.time() - t0)
    if args.save_params:
      from jax_titan.trainer import checkpoint  # pylint: disable=g-import-not-at-top

      t0 = time.time()
      checkpoint.save_params(args.save_params, layout.export(params))
      print(f"saved parameters to {args.save_params} in {time.time() - t0:.1f}s", flush=True)
  except BaseException as e:
    status = f"{type(e).__name__}: {e}"[:500]
    raise
  finally:
    if power_sampler:
      power_sampler.stop()
    log.event("end", status=status, steps_done=done - start,
              **({"target_reached": reached} if args.target_eval_loss else {}))
    log.close()
  if args.run_dir and jax.process_index() == 0:
    from jax_titan.observability import summarize  # pylint: disable=g-import-not-at-top

    try:
      summarize.write_results(args.run_dir, skip=min(5, (args.steps - start) // 2))
      print(f"results: {os.path.join(args.run_dir, 'results.json')}", flush=True)
    except (ValueError, KeyError, OSError) as e:
      print(f"no results.json: {type(e).__name__}: {e}", flush=True)
    try:
      from jax_titan.observability import plot  # pylint: disable=g-import-not-at-top

      figures = plot.plot_run(plot.series(args.run_dir), _mkdir(args.run_dir, "figures"))
      print(f"figures: {', '.join(map(str, figures))}", flush=True)
    except ImportError:
      pass
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
