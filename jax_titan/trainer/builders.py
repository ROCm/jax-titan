"""The trainer's builders: one function per JAX-Titan model and per repository
of models. A builder makes, from the flags, the model that `--model` names and
gives it to a layout function; it returns the `Layout` and the `Workload`
(`jax_titan.trainer.layout`). `build` finds the builder in `BUILDERS` by the
module that implements the model (`jax_titan.models`).

A model of another repository trains as one stage, the repository's model
called whole, with `--parallel=dp` or `fsdp`:
`--model=maxtext:<name>` a MaxText model config (e.g. `maxtext:llama3.1-8b`
or `maxtext:qwen3-8b`) on the token data, `--model=maxdiffusion:<flavor>`
FLUX.1 (`flux-dev`, `flux-schnell`) or Wan 2.1 (`wan2.1-t2v-1.3b`) on
synthetic rectified-flow batches, `--model=transformers:<name>` a decoder-only
causal language model of Hugging Face transformers, a directory with a
`config.json`, a model type or a model on the Hub, and
`--model=diffusers:<name>` a denoiser of diffusers on synthetic batches. The
last two run through torchax (`models/adapters/torch_bridge.py`) and
rematerialize through the library's own gradient-checkpointing switch.
MaxText's and MaxDiffusion's FLUX.1 attention is their own JAX
implementation unless the ops policy names attention.

`--model=ideogram4` trains JAX-Titan's own Ideogram 4
(`jax_titan.models.ideogram`) on synthetic rectified-flow batches or on a
cache of image tokens and text features (`jax_titan.data.ideogram`), and
`--model=maxdiffusion:ideogram4` the model of MaxDiffusion's Ideogram branch
on the same batches. They run as a Shardy MPMD pipeline with data parallelism
inside each stage, reducing gradients in every microbatch's backward pass.
Ideogram 4's attention does not follow the ops policy: JAX-Titan's model passes
a mask, which the providers decline, and the branch's model runs its own
attention. `--model=command-a-plus` trains JAX-Titan's Command A+
(`jax_titan.models.command_a`) on token windows or on a cache of conversations
(`jax_titan.data.command_a`), as one stage or, with its embedding frozen, as a
pipeline. `--model=flux-dev` and `flux-schnell` train JAX-Titan's own FLUX.1
(`jax_titan.models.flux`) on the synthetic batches of the two FLUX.1 adapters,
as one stage, and `--model=llama3.1-8b` JAX-Titan's Llama
(`jax_titan.models.llama`) on token windows, as one stage. For these,
`--parallel=dp` runs the same stages, weights and batches as one data-parallel
`jax.jit` program, the no-pipeline reference, and `--parallel=fsdp` also shards
parameters and optimizer state.
"""

import dataclasses
import functools
import json
import os

from jax_titan import models
from jax_titan.trainer.layout import EVAL_STEPS
from jax_titan.trainer.layout import Layout
from jax_titan.trainer.layout import mpmd_layout
from jax_titan.trainer.layout import spmd_layout
from jax_titan.trainer.layout import stage_topology
from jax_titan.trainer.layout import Workload


def _overrides(args) -> dict:
  return json.loads(args.model_overrides) if args.model_overrides else {}


def _stage_layers(args):
  return tuple(int(n) for n in args.stage_layers.split(",")) if args.stage_layers else None


def _num_params(shapes) -> int:
  import jax  # pylint: disable=g-import-not-at-top

  return sum(x.size for x in jax.tree.leaves(shapes))


def _token_batches(args):
  """The spec of a step's token windows, inputs and targets alike, and the
  function that splits a host batch of windows of `seq_len + 1` into them."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top

  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  tokens = (jax.ShapeDtypeStruct((num_mb, mb, seq), jnp.int32),
            jax.sharding.PartitionSpec(None, jt_mpmd.DATA_AXIS, None))

  def split_batch(batch):
    batch = batch.reshape(num_mb, mb, seq + 1)
    return batch[..., :-1], batch[..., 1:]

  return tokens, split_batch


def _needs_own_tokens(args, vocab_size: int):
  """Refuses the default token files, which hold Llama 3's tokens, for another vocabulary."""
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top

  own_tokens = vocab_size != llama.FLAVORS["llama3.1-8b"].vocab_size
  held_out = args.eval_every and args.data != "synthetic" and not args.eval_data
  if own_tokens and (not args.data or held_out):
    raise ValueError(
        f"{args.model} does not read Llama 3's tokens, which the default --data and --eval_data "
        "hold: name its own token files (python -m jax_titan.data.text --tokenizer)")


def _one_stage(args, ops: bool = False):
  """Refuses what a model without pipeline stages cannot run as, and `--ops`
  for one whose operators `jax_titan.ops` cannot reach."""
  if args.parallel == "mpmd" or args.num_stages != 1:
    raise NotImplementedError(
        f"{args.model} has no pipeline: --parallel=dp or --parallel=fsdp, without --num_stages")
  if args.ops and not ops:
    raise NotImplementedError(f"{args.model} runs its own operators: no --ops")


def _no_init_from(args):
  """Refuses `--init_from` for a model that reads no parameters."""
  if args.init_from:
    raise NotImplementedError(
        "--init_from loads parameters of maxtext: and transformers: models, ideogram4 and "
        "command-a-plus")


def _synthetic_only(args):
  """Refuses `--data` for a model that trains on synthetic batches alone."""
  if args.data not in ("", "synthetic"):
    raise NotImplementedError(f"{args.model} trains on synthetic batches only")


def _data_mesh(devices):
  import jax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top

  return jax.sharding.Mesh(
      np.asarray(devices).reshape(len(devices), 1), (jt_mpmd.DATA_AXIS, jt_mpmd.TENSOR_AXIS))


def _keep_batch_sharded(mesh):
  """A constraint that keeps a batch sharded along the data axis of `mesh`, so
  that FSDP gathers the weights and not the activations."""
  import jax  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top

  return functools.partial(
      jax.lax.with_sharding_constraint,
      shardings=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(jt_mpmd.DATA_AXIS)))


def _routed_attention():
  """`jax_titan.ops.attention` when the ops policy names attention, else None.

  Unrouted, a reference model keeps its own attention implementation.
  """
  from jax_titan import ops  # pylint: disable=g-import-not-at-top

  policy = ops.current_policy()
  routed = "attention" in policy.prefs or any("attention" in s for s in policy.stages.values())
  return ops.attention if routed else None


def _maxtext(args, devices, make_optimizer):
  from jax_titan.models.adapters import maxtext as mt  # pylint: disable=g-import-not-at-top

  name = args.model.partition(":")[2]
  _one_stage(args, ops=True)
  if not args.remat:
    raise NotImplementedError(
        f"{args.model}: MaxText rematerializes its scanned layers under every remat_policy")
  cfg = mt.config(
      name, **{"remat_policy": "full", **_overrides(args), "max_target_length": args.seq_len})
  _needs_own_tokens(args, cfg.vocab_size)
  mesh = mt.mesh(cfg, devices)
  model = mt.AdapterModel(cfg, mesh, attention=_routed_attention(), fsdp=args.parallel == "fsdp")
  tokens, split_batch = _token_batches(args)
  layout = spmd_layout(args, mesh, model, make_optimizer(model.decays), tokens, tokens,
                       split_batch)
  tokens_per_step = args.microbatches * args.microbatch_size * args.seq_len
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=tokens_per_step,
      flops_per_step=model.flops_per_token() * tokens_per_step,
      vocab_size=cfg.vocab_size,
  )
  return layout, work


def _transformers(args, devices, make_optimizer):
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import transformers as hf  # pylint: disable=g-import-not-at-top

  name = args.model.partition(":")[2]
  _one_stage(args)
  overrides = _overrides(args)
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  cfg = hf.config(name, **overrides)
  if cfg.model_type not in hf.VERIFIED:
    raise NotImplementedError(
        f"{args.model} is of transformers' model type {cfg.model_type!r}, which was not "
        f"compared with PyTorch through torchax: {', '.join(hf.VERIFIED)}")
  vocab_size = hf.vocab_size(cfg)
  _needs_own_tokens(args, vocab_size)
  model = hf.causal_lm(cfg, seed=args.seed, dtype=dtype, remat=args.remat,
                       init_from=args.init_from)
  tokens, split_batch = _token_batches(args)
  layout = spmd_layout(args, _data_mesh(devices), model, make_optimizer(model.decays),
                       tokens, tokens, split_batch)
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=args.microbatches * args.microbatch_size * args.seq_len,
      flops_per_step=None,
      vocab_size=vocab_size,
  )
  return layout, work


def _batch_specs(batches):
  """The specs of a family's batches (`jax_titan.data.diffusion`): inputs and target."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name
  x, target = batches.next_batch(0)

  def spec(name, a):
    return (jax.ShapeDtypeStruct(a.shape, jnp.float32),
            P() if name in batches.constants else P(None, jt_mpmd.DATA_AXIS))

  return {k: spec(k, v) for k, v in x.items()}, spec("", target)


def _flux_batches(args, *, in_channels, joint_attention_dim, pooled_projection_dim,
                  guidance_embeds):
  """FLUX.1's batches at `--resolution` and `--text_tokens`, and its text tokens."""
  from jax_titan.data import diffusion  # pylint: disable=g-import-not-at-top

  text_tokens = args.text_tokens or (512 if guidance_embeds else 256)
  batches = diffusion.flux(
      args.seed, (args.microbatches, args.microbatch_size), image_side=args.resolution // 16,
      text_tokens=text_tokens, in_channels=in_channels, joint_attention_dim=joint_attention_dim,
      pooled_projection_dim=pooled_projection_dim, guidance_embeds=guidance_embeds)
  return batches, text_tokens


def _diffusers(args, devices, make_optimizer):
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import diffusers as hfd  # pylint: disable=g-import-not-at-top

  name = args.model.partition(":")[2]
  _no_init_from(args)
  _synthetic_only(args)
  _one_stage(args)
  overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in _overrides(args).items()}
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  cfg = hfd.config(name, **overrides)
  if cfg["_class_name"] not in hfd.FAMILIES:
    raise NotImplementedError(
        f"{args.model} is diffusers' {cfg['_class_name']}, which was not compared with PyTorch "
        f"through torchax: {', '.join(hfd.FAMILIES)}")
  model = hfd.denoiser(cfg, seed=args.seed, dtype=dtype, remat=args.remat)
  c = model.module.config
  batches, _ = _flux_batches(
      args, in_channels=c.in_channels, joint_attention_dim=c.joint_attention_dim,
      pooled_projection_dim=c.pooled_projection_dim, guidance_embeds=c.guidance_embeds)
  inputs, targets = _batch_specs(batches)
  layout = spmd_layout(args, _data_mesh(devices), model, make_optimizer(model.decays),
                       inputs, targets, lambda batch: batch)
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=args.microbatches * args.microbatch_size * batches.tokens,
      flops_per_step=None,
      next_batch=batches.next_batch,
  )
  return layout, work


def _maxdiffusion(args, devices, make_optimizer):
  flavor = args.model.partition(":")[2]
  _no_init_from(args)
  if flavor == "ideogram4":
    return _ideogram(args, devices, make_optimizer, adapter=True)
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.data import diffusion  # pylint: disable=g-import-not-at-top
  from jax_titan.models.adapters import maxdiffusion as md  # pylint: disable=g-import-not-at-top

  _synthetic_only(args)
  # FLUX.1's attention can run through jax_titan.ops; Wan 2.1's cannot.
  _one_stage(args, ops=flavor not in md.WAN)
  kwargs = {
      "dtype": jnp.bfloat16,
      "weights_dtype": jnp.float32,
      "remat_policy": "FULL" if args.remat else "NONE",
      **{k: tuple(v) if isinstance(v, list) else v for k, v in _overrides(args).items()},
  }
  for k in ("dtype", "weights_dtype"):
    kwargs[k] = jnp.dtype(kwargs[k])
  mesh, fsdp = _data_mesh(devices), args.parallel == "fsdp"
  samples = args.microbatches * args.microbatch_size
  if flavor in md.WAN:
    cfg = {**md.WAN[flavor], **kwargs}
    model = md.whole_wan(mesh, fsdp=fsdp, **cfg)
    batches = diffusion.wan(
        args.seed, (args.microbatches, args.microbatch_size),
        frames=(args.frames - 1) // 4 + 1, height=args.resolution // 8, width=args.resolution // 8,
        patch=cfg["patch_size"], text_tokens=args.text_tokens or 512,
        in_channels=cfg["in_channels"], text_dim=cfg["text_dim"])
    flops = None
  else:
    flux = md.transformer(flavor, **kwargs)
    model = md.whole_flux(flux, mesh, fsdp=fsdp, attention=_routed_attention())
    batches, text_tokens = _flux_batches(
        args, in_channels=flux.in_channels, joint_attention_dim=flux.joint_attention_dim,
        pooled_projection_dim=flux.pooled_projection_dim, guidance_embeds=flux.guidance_embeds)
    flops = samples * md.flux_flops_per_sample(flux, batches.tokens - text_tokens, text_tokens)
  inputs, targets = _batch_specs(batches)
  layout = spmd_layout(args, mesh, model, make_optimizer(model.decays), inputs, targets,
                       lambda batch: batch)
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=samples * batches.tokens,
      flops_per_step=flops,
      next_batch=batches.next_batch,
  )
  return layout, work


def _flux(args, devices, make_optimizer):
  """JAX-Titan's FLUX.1 on the batches of `_flux_batches`. Besides config
  fields, `--model_overrides` takes `dtype`."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models import flux  # pylint: disable=g-import-not-at-top

  _no_init_from(args)
  _synthetic_only(args)
  _one_stage(args, ops=True)
  overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in _overrides(args).items()}
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  cfg = dataclasses.replace(flux.FLAVORS[args.model], **overrides)
  mesh = _data_mesh(devices)
  constrain = _keep_batch_sharded(mesh) if args.parallel == "fsdp" else None
  model = flux.Model(cfg, dtype=dtype, remat=args.remat, constrain=constrain)
  batches, text_tokens = _flux_batches(
      args, in_channels=cfg.in_channels, joint_attention_dim=cfg.joint_attention_dim,
      pooled_projection_dim=cfg.pooled_projection_dim, guidance_embeds=cfg.guidance_embeds)
  inputs, targets = _batch_specs(batches)
  layout = spmd_layout(args, mesh, model, make_optimizer(model.decays), inputs, targets,
                       lambda batch: batch)
  samples = args.microbatches * args.microbatch_size
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=samples * batches.tokens,
      flops_per_step=samples * cfg.flops_per_sample(batches.tokens - text_tokens, text_tokens),
      next_batch=batches.next_batch,
  )
  return layout, work


def _llama(args, devices, make_optimizer):
  """JAX-Titan's Llama on token windows. Besides config fields,
  `--model_overrides` takes `dtype`."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax_titan.models import llama  # pylint: disable=g-import-not-at-top
  from jax_titan.observability import flops as flops_lib  # pylint: disable=g-import-not-at-top

  _no_init_from(args)
  _one_stage(args, ops=True)
  overrides = _overrides(args)
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  cfg = dataclasses.replace(llama.FLAVORS[args.model], **overrides)
  _needs_own_tokens(args, cfg.vocab_size)
  mesh = _data_mesh(devices)
  model = llama.Model(cfg, dtype=dtype, remat=args.remat, constrain=_keep_batch_sharded(mesh))
  tokens, _ = _token_batches(args)
  # One microbatch of any size: a validation batch need not have the step's.
  layout = spmd_layout(args, mesh, model, make_optimizer(model.decays), tokens, tokens,
                       lambda batch: (batch[None, :, :-1], batch[None, :, 1:]))
  seq = args.seq_len
  tokens_per_step = args.microbatches * args.microbatch_size * seq
  work = Workload(
      num_params=cfg.num_params(),
      tokens_per_step=tokens_per_step,
      flops_per_step=cfg.flops_per_token(seq) * tokens_per_step,
      vocab_size=cfg.vocab_size,
      hardware_flops_per_step=(cfg.flops_per_token(seq) + flops_lib.recompute_flops_per_token(
          cfg, seq, args.remat)) * tokens_per_step,
  )
  return layout, work


def _ideogram(args, devices, make_optimizer, adapter: bool = False):
  """Ideogram 4: JAX-Titan's model or, with `adapter`, the one of MaxDiffusion's
  Ideogram branch (`--model=maxdiffusion:ideogram4`).

  Besides config fields, `--model_overrides` takes `dtype` and, for JAX-Titan's
  model, `timestep_mean` and `timestep_std` of the training timesteps. That
  model's config starts from the `config.json` of published weights that
  `--init_from` names. It also trains on a cache of real samples
  (`--data`), with a tenth of its images, at least one, held out for
  validation when `--eval_every` is set, so a run that resumes with that flag
  changed trains on other images.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan.models import ideogram, pretrained  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  overrides = _overrides(args)
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  schedule = {}
  if adapter:
    from jax_titan.models.adapters import ideogram as ig  # pylint: disable=g-import-not-at-top

    config, model_class = ig.Ideogram4Config, ig.AdapterModel
  else:
    ig, config, model_class = ideogram, ideogram.IdeogramConfig, ideogram.Model
    schedule = {k: overrides.pop(f"timestep_{k}") for k in ("mean", "std")
                if f"timestep_{k}" in overrides}
  fields = {k: tuple(v) if isinstance(v, list) else v for k, v in overrides.items()}
  if not adapter and args.init_from and pretrained.is_safetensors(args.init_from):
    cfg = dataclasses.replace(ideogram.pretrained_config(args.init_from), **fields)
  else:
    cfg = config(**fields)

  mesh = _data_mesh(devices)
  constrain = _keep_batch_sharded(mesh) if args.parallel == "fsdp" else None
  model = model_class(cfg, num_stages=args.num_stages, blocks=_stage_layers(args), dtype=dtype,
                      remat=args.remat, constrain=constrain)

  side = args.resolution // 16
  image_tokens = side * side
  text_tokens = args.text_tokens or 256
  cache, held_out = None, 0
  if args.data not in ("", "synthetic"):
    if adapter:
      raise NotImplementedError("--data reads a cache of samples for ideogram4 only")
    from jax_titan.data import ideogram as data  # pylint: disable=g-import-not-at-top

    cache = data.Cache.open(args.data)
    if cache.resolution != args.resolution:
      raise ValueError(
          f"{args.data} holds images of {cache.resolution} pixels, not {args.resolution}")
    text_tokens = cache.text_tokens
    held_out = max(1, len(cache) // 10) if args.eval_every else 0
    if held_out >= len(cache):
      raise ValueError(f"{args.data} has {len(cache)} images, too few to hold one out")
  ids = ig.sequence_ids(side, text_tokens)
  mb, num_mb = args.microbatch_size, args.microbatches
  b = (num_mb, mb)
  latent_shape = b + (image_tokens, cfg.in_channels)
  text_shape = b + (text_tokens, cfg.llm_features_dim)
  f32 = jnp.float32
  batched = P(None, jt_mpmd.DATA_AXIS)
  inputs = {
      "img": (jax.ShapeDtypeStruct(latent_shape, f32), batched),
      "txt": (jax.ShapeDtypeStruct(text_shape, f32), batched),
      "timestep": (jax.ShapeDtypeStruct(b, f32), batched),
      **{k: (jax.ShapeDtypeStruct(b + v.shape, v.dtype), batched) for k, v in ids.items()},
  }
  targets = (jax.ShapeDtypeStruct(latent_shape, f32), batched)
  # The synthetic text features are the largest input by far, and the same every step.
  txt = None if cache else np.random.default_rng(args.seed).standard_normal(text_shape, np.float32)

  def next_batch(step):
    """Noised latents and velocity targets with their text features: random ones,
    or the cache's samples, the held-out images for a validation step."""
    rng = np.random.default_rng((args.seed, step))
    if cache is None:
      latents = rng.standard_normal(latent_shape, np.float32)
      text, batch_ids = txt, {k: np.broadcast_to(v, b + v.shape) for k, v in ids.items()}
    else:
      train = len(cache) - held_out
      if step >= EVAL_STEPS:
        images = train + rng.integers(0, held_out, b)
      else:
        images = rng.integers(0, train, b)
      latents, text, valid = (x.reshape(b + x.shape[1:]) for x in cache.samples(images.reshape(-1)))
      batch_ids = ig.padded_sequence_ids(side, valid)
    noise = rng.standard_normal(latent_shape, np.float32)
    t = ig.timesteps(rng.standard_normal(b), args.resolution, **schedule)
    img, target = ideogram.noised(latents, noise, t)
    x = {"img": img, "txt": text, "timestep": t, **batch_ids}
    return x, target

  optimizer = make_optimizer(model.decays)
  if args.parallel == "mpmd":
    layout = mpmd_layout(args, stage_topology(args, devices, _data_mesh), model, optimizer,
                         inputs, targets, lambda batch: batch)
  else:
    layout = spmd_layout(args, mesh, model, optimizer, inputs, targets, lambda batch: batch)
  samples = num_mb * mb
  work = Workload(
      num_params=_num_params(model.abstract(args.seed)),
      tokens_per_step=samples * (image_tokens + text_tokens),
      flops_per_step=model.flops_per_sample(image_tokens, text_tokens) * samples,
      next_batch=next_batch,
      data="synthetic" if cache is None else
      f"{args.data} ({len(cache) - held_out} images, {held_out} more held out)",
  )
  return layout, work


def _command_a(args, devices, make_optimizer):
  """Command A+ on token windows, or on a cache of conversations.

  Besides config fields, `--model_overrides` takes `dtype`, `image_tiles` and
  `freeze`. With tiles, every sequence of a synthetic batch opens with that
  many random image tiles, and the vision tower and the projector train with
  the language model; without, they are not part of the trained state.
  `freeze` lists what is not trained, of `experts` and `embed`; the model runs
  as more than one stage only with its embedding frozen. The config starts
  from the `config.json` of the published checkpoint that `--init_from` names.

  `--data`, when a directory, is a cache that `jax_titan.data.command_a`
  wrote: a step trains on random conversations of it, and with `--eval_every`
  its last tenth, at least one, is held out and validated on in turn.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top
  from jax_titan import mpmd as jt_mpmd  # pylint: disable=g-import-not-at-top
  from jax_titan.data import command_a as data  # pylint: disable=g-import-not-at-top
  from jax_titan.models import command_a, pretrained  # pylint: disable=g-import-not-at-top

  P = jax.sharding.PartitionSpec  # pylint: disable=invalid-name

  overrides = _overrides(args)
  dtype = jnp.dtype(overrides.pop("dtype", "bfloat16"))
  tiles = overrides.pop("image_tiles", 0)
  freeze = overrides.pop("freeze", ())
  if args.init_from and pretrained.is_safetensors(args.init_from):
    cfg = dataclasses.replace(command_a.pretrained_config(args.init_from), **overrides)
  else:
    cfg = command_a.CommandAConfig(**overrides)

  mesh = _data_mesh(devices)
  constrain = _keep_batch_sharded(mesh) if args.parallel == "fsdp" else None
  mb, num_mb, seq = args.microbatch_size, args.microbatches, args.seq_len
  cache = data.Cache.open(args.data) if os.path.isdir(args.data) else None
  if cache is not None:
    seq = cache.tokens.shape[1]
  # Conversations give microbatches different numbers of targets: the layouts without a
  # pipeline then take the pipeline's mean of the microbatches' means.
  flat = cache is not None and args.parallel != "mpmd"
  model = command_a.Model(cfg, num_stages=args.num_stages, layers=_stage_layers(args),
                          dtype=dtype, remat=args.remat, constrain=constrain, images=tiles > 0,
                          freeze=freeze, microbatches=num_mb if flat else 1)
  b = (num_mb, mb)
  batched = P(None, jt_mpmd.DATA_AXIS)
  tokens = (jax.ShapeDtypeStruct(b + (seq,), jnp.int32), batched)
  source = "synthetic"

  def whole(batch):
    return batch

  def windows(batch):
    batch = batch.reshape(num_mb, mb, seq + 1)
    return batch[..., :-1], batch[..., 1:]

  split_batch = whole
  if cache is not None and not tiles:
    held_out = max(1, len(cache) // 10) if args.eval_every else 0
    train = len(cache) - held_out
    if train < 1:
      raise ValueError(f"{args.data} has {len(cache)} conversations, too few to hold one out")
    inputs = tokens
    source = f"{args.data} ({train} conversations, {held_out} more held out)"

    def next_batch(step):
      """Conversations of the cache: random ones, or for a validation step the
      held-out ones in turn."""
      if step >= EVAL_STEPS:
        picks = train + (np.arange(num_mb * mb) + (step - EVAL_STEPS) * num_mb * mb) % held_out
      else:
        picks = np.random.default_rng((args.seed, step)).integers(0, train, num_mb * mb)
      picks = picks.reshape(b)
      return cache.tokens[picks], cache.targets[picks]
  elif not tiles:
    inputs, next_batch, split_batch = tokens, None, windows
  else:
    if args.data not in ("", "synthetic"):
      raise NotImplementedError("command-a-plus trains with image tiles on synthetic batches only")
    image_tokens, size = tiles * cfg.tile_tokens, cfg.vision_image_size
    if image_tokens >= seq:
      raise ValueError(f"{tiles} tiles are {image_tokens} tokens, more than --seq_len={seq} holds")
    pixel_shape = b + (tiles, cfg.vision_num_channels, size, size)
    inputs = {"tokens": tokens, "pixels": (jax.ShapeDtypeStruct(pixel_shape, jnp.float32), batched)}

    def next_batch(step):
      """Random tiles, their image tokens and random text; the text is the target."""
      rng = np.random.default_rng((args.seed, step))
      text = rng.integers(0, cfg.vocab_size, b + (seq + 1 - image_tokens,), dtype=np.int32)
      text[text == cfg.image_token_id] = cfg.pad_token_id
      image = np.full(b + (image_tokens,), cfg.image_token_id, np.int32)
      window = np.concatenate([image, text], axis=-1)
      x = {"tokens": window[..., :-1], "pixels": rng.standard_normal(pixel_shape, np.float32)}
      return x, np.where(window == cfg.image_token_id, -1, window)[..., 1:]

  optimizer = make_optimizer(model.decays)
  if args.parallel == "mpmd":
    layout = mpmd_layout(args, stage_topology(args, devices, _data_mesh), model, optimizer,
                         inputs, tokens, split_batch)
  else:
    layout = spmd_layout(args, mesh, model, optimizer, inputs, tokens, split_batch)
  tokens_per_step = num_mb * mb * seq
  # Not the stages' parameters: with several stages the embedding is held twice.
  shapes = jax.eval_shape(lambda: command_a.init_params(cfg, jax.random.key(0), tiles > 0))
  work = Workload(
      num_params=_num_params(shapes),
      tokens_per_step=tokens_per_step,
      flops_per_step=model.flops_per_token(seq) * tokens_per_step,
      vocab_size=cfg.vocab_size,
      next_batch=next_batch,
      data=source,
  )
  return layout, work


# The module of a model (`jax_titan.models`) -> its builder.
BUILDERS = {
    "llama": _llama,
    "flux": _flux,
    "ideogram": _ideogram,
    "command_a": _command_a,
    "adapters.maxtext": _maxtext,
    "adapters.maxdiffusion": _maxdiffusion,
    "adapters.transformers": _transformers,
    "adapters.diffusers": _diffusers,
}


def build(args, devices, make_optimizer) -> tuple[Layout, Workload]:
  """The layout and workload of `--model`, from the builder of its module.

  `make_optimizer(decays)` gives the optimizer for a model's `decays(path,
  leaf)`, which says which parameters take weight decay.
  """
  if args.stage_fsdp or args.tensor_parallel != 1:
    raise NotImplementedError(f"{args.model} takes neither --stage_fsdp nor --tensor_parallel")
  if args.parallel == "mpmd" and args.reduce_grads != "microbatch":
    raise NotImplementedError(f"{args.model} reduces gradients per microbatch only")
  if args.parallel != "mpmd" and (args.microbatches * args.microbatch_size) % len(devices):
    raise ValueError(
        f"--parallel={args.parallel} splits all {args.microbatches * args.microbatch_size} "
        f"sequences of a step across {len(devices)} devices"
    )
  return BUILDERS[models.module_of(args.model)](args, devices, make_optimizer)
