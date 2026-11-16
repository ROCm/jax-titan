"""Pipeline stages of Ideogram 4's transformer, from MaxDiffusion's Ideogram branch.

The branch (`yaomingamd/maxdiffusion`, `ymmu/ideogram_v4_training`, pinned at
`6db5c172`) supplies the model, its blocks and its parameter layout;
JAX-Titan supplies the stages, meshes and train step. Ideogram 4 runs one
sequence, the text tokens followed by the image tokens, through 34 identical
blocks. A stage holds a contiguous run of blocks; stage 0 also holds the
input projections and the timestep embedding, and the last stage the output
layer. Stage parameters keep the branch's names, with a stage's blocks
numbered from 0.

Stage 0 takes the text features of the text positions only and pads them to
the sequence after projecting them, which is what the branch's transformer
computes from features zero-padded over the image positions. Every stage
boundary carries the hidden states, the segment ids, the rotary embeddings
and the conditioning, and the last stage returns the branch's flow-matching
loss over the image tokens.
"""

from collections.abc import Callable, Sequence
import functools
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from maxdiffusion.models.ideogram import constants
from maxdiffusion.models.ideogram import scheduler
from maxdiffusion.models.ideogram.transformer_ideogram import Ideogram4Config
from maxdiffusion.models.ideogram.transformer_ideogram import Ideogram4MRoPE
from maxdiffusion.models.ideogram.transformer_ideogram import Ideogram4Transformer

PINNED_COMMIT = "6db5c17257ea6a37b444ef5669ebc730350e94b0"
_FIRST = ("input_proj", "llm_cond_norm", "llm_cond_proj", "t_embedding", "adaln_proj",
          "embed_image_indicator")
_LAST = ("final_layer",)


def flow_matching_loss(pred: jax.Array, target: jax.Array) -> jax.Array:
  """The branch's loss: mean squared error to latents minus noise, over the image tokens."""
  pred = pred[:, -target.shape[1]:]
  return jnp.mean((target.astype(jnp.float32) - pred.astype(jnp.float32)) ** 2)


def timesteps(z: np.ndarray, resolution: int) -> np.ndarray:
  """The branch's training timesteps for standard normal `z`: its logit-normal
  schedule for square images of `resolution` pixels. 0 is noise and 1 is data."""
  s = scheduler.get_schedule_for_resolution((resolution, resolution), known_mean=0.5)
  t = 1.0 / (1.0 + np.exp(s.mean + s.std * z))
  return np.clip(t, 1.0 / (1.0 + math.exp(0.5 * s.logsnr_max)),
                 1.0 / (1.0 + math.exp(0.5 * s.logsnr_min))).astype(np.float32)


def sequence_ids(image_side: int, text_tokens: int) -> dict[str, np.ndarray]:
  """`position_ids` [tokens, 3], `segment_ids` and `indicator` [tokens] of one
  sample whose text fills all `text_tokens` positions, as the branch builds them."""
  rows, columns = np.divmod(np.arange(image_side * image_side), image_side)
  image = np.stack([np.zeros_like(rows), rows, columns], axis=1) + constants.IMAGE_POSITION_OFFSET
  text = np.repeat(np.arange(text_tokens)[:, None], 3, axis=1)
  tokens = text_tokens + image_side * image_side
  return {
      "position_ids": np.concatenate([text, image]).astype(np.int32),
      "segment_ids": np.ones(tokens, np.int32),
      "indicator": np.concatenate([
          np.full(text_tokens, constants.LLM_TOKEN_INDICATOR),
          np.full(tokens - text_tokens, constants.OUTPUT_IMAGE_INDICATOR),
      ]).astype(np.int32),
  }


class Model:
  """The branch's Ideogram 4 transformer split into pipeline stages.

  `blocks` gives the blocks per stage; the default splits them as evenly as
  possible, earlier stages taking the extra blocks. `constrain`, when given,
  is applied to the conditioning, and to the hidden states after the embedding
  and after every block.
  """

  def __init__(self, config: Ideogram4Config, *, num_stages: int,
               blocks: Sequence[int] | None = None, dtype=jnp.bfloat16, remat: bool = True,
               constrain: Callable | None = None):
    total = config.num_layers
    if blocks is None:
      blocks = tuple(total // num_stages + (i < total % num_stages) for i in range(num_stages))
    if len(blocks) != num_stages or sum(blocks) != total or min(blocks) < 1:
      raise ValueError(f"stage blocks {tuple(blocks)} do not split {total} blocks")
    self.config = config
    self.num_stages = num_stages
    self.blocks = tuple(blocks)
    self.dtype = dtype
    self._remat = remat
    self._constrain = constrain or (lambda h: h)
    abstract = nnx.eval_shape(lambda: Ideogram4Transformer(nnx.Rngs(0), config, dtype=dtype))
    self._graphs = {name: nnx.graphdef(getattr(abstract, name)) for name in _FIRST + _LAST}
    self._block_graph = nnx.graphdef(abstract.layers[0])
    self._rope = Ideogram4MRoPE(
        config.emb_dim // config.num_heads, config.rope_theta, config.mrope_section)

  def _init_full(self, seed: int) -> dict:
    transformer = Ideogram4Transformer(nnx.Rngs(seed), self.config, dtype=self.dtype)
    return nnx.state(transformer, nnx.Param).to_pure_dict()

  def split(self, params: dict) -> tuple[dict, ...]:
    """Stage parameter dicts from the parameters of `Ideogram4Transformer`."""
    stages, start = [], 0
    for i, n in enumerate(self.blocks):
      p = {"layers": {j: params["layers"][start + j] for j in range(n)}}
      if i == 0:
        p.update({k: params[k] for k in _FIRST})
      if i == self.num_stages - 1:
        p.update({k: params[k] for k in _LAST})
      stages.append(p)
      start += n
    return tuple(stages)

  def init(self, seed: int = 0) -> tuple[dict, ...]:
    """Stage parameters, initialized as the branch initializes its transformer."""
    return self.split(jax.jit(lambda: self._init_full(seed))())

  def abstract(self, seed: int = 0) -> tuple[dict, ...]:
    """ShapeDtypeStructs of `init(seed)`, without initializing."""
    return jax.eval_shape(lambda: self.split(self._init_full(seed)))

  def _embed(self, params: dict, x: dict):
    """`Ideogram4Transformer.__call__` up to the first block."""
    def layer(name):
      return nnx.merge(self._graphs[name], params[name])

    dtype = self.dtype
    indicator = x["indicator"].astype(jnp.int32)
    text_tokens = x["txt"].shape[1]
    image = indicator == constants.OUTPUT_IMAGE_INDICATOR
    image_mask = image.astype(dtype)[..., None]
    llm = indicator[:, :text_tokens] == constants.LLM_TOKEN_INDICATOR
    llm_mask = llm.astype(dtype)[..., None]

    img = jnp.pad(x["img"].astype(dtype), ((0, 0), (text_tokens, 0), (0, 0))) * image_mask
    h = layer("input_proj")(img) * image_mask
    txt = layer("llm_cond_norm")(x["txt"].astype(dtype) * llm_mask)
    txt = layer("llm_cond_proj")(txt) * llm_mask
    h = h + jnp.pad(txt, ((0, 0), (0, h.shape[1] - text_tokens), (0, 0)))
    h = h + layer("embed_image_indicator")(image.astype(jnp.int32))

    t_cond = layer("t_embedding")(x["timestep"].astype(dtype))[:, None]
    adaln = jax.nn.silu(layer("adaln_proj")(t_cond))
    cos, sin = self._rope(x["position_ids"])
    return h, x["segment_ids"], cos.astype(dtype), sin.astype(dtype), adaln

  def forward(self, stage: int, params: dict, x):
    """Runs one stage.

    Stage 0 takes a dict of `img` noised latent tokens [batch, image_tokens,
    in_channels], `txt` Qwen3-VL features of the text positions [batch,
    text_tokens, llm_features_dim], `timestep` [batch], and `position_ids`
    [batch, tokens, 3], `segment_ids` and `indicator` [batch, tokens] over the
    text tokens followed by the image tokens. The last stage returns the
    predicted velocity [batch, tokens, in_channels].
    """
    h, segment_ids, cos, sin, adaln = self._embed(params, x) if stage == 0 else x
    h, adaln = self._constrain(h), self._constrain(adaln)

    def block(p, h, segment_ids, cos, sin, adaln):
      return nnx.merge(self._block_graph, p)(h, segment_ids, cos, sin, adaln)

    if self._remat:
      block = jax.checkpoint(block, prevent_cse=True)
    for j in range(self.blocks[stage]):
      h = self._constrain(block(params["layers"][j], h, segment_ids, cos, sin, adaln))
    if stage < self.num_stages - 1:
      return h, segment_ids, cos, sin, adaln
    return nnx.merge(self._graphs["final_layer"], params["final_layer"])(h, c=adaln)

  def stage_fns(self) -> list[Callable[..., jax.Array]]:
    """Stage functions for `jax_titan.mpmd.make_train_step`.

    The last stage also takes the velocity target of the image tokens and
    returns `flow_matching_loss`.
    """
    fns = [functools.partial(self.forward, i) for i in range(self.num_stages - 1)]
    last = self.num_stages - 1
    fns.append(lambda p, x, t: flow_matching_loss(self.forward(last, p, x), t))
    return fns

  @staticmethod
  def logical_axes(path, leaf) -> Sequence[str | None]:
    """The axes of a parameter for `jax_titan.parallel.tree_specs`. None is
    named yet: nothing is tensor-parallel, and FSDP shards each array's first
    dimension that splits evenly."""
    del path
    return (None,) * len(leaf.shape)

  def flops_per_sample(self, image_tokens: int, text_tokens: int) -> float:
    """Training FLOPs per sample: 6 per weight and token, plus attention."""
    cfg = self.config
    d, tokens = cfg.emb_dim, image_tokens + text_tokens
    per_token = 4 * d * d + 3 * d * cfg.intermediate_size
    weights = per_token * tokens * cfg.num_layers + d * (
        cfg.llm_features_dim * text_tokens + 2 * cfg.in_channels * image_tokens)
    attention = 2 * tokens * tokens * d * cfg.num_layers
    return 6.0 * (weights + attention)
