# Adding a model

A model is a module of `jax_titan/models`, its names in one table and a
builder in the trainer. The trainer's loop names no model: it finds the
builder by the model's module, and the builder gives the model to a layout.
[design/model_definition.md](design/model_definition.md) has the design and
an example of every kind of model; `models/llama.py` is the smallest model to
read.

## The module

A JAX-Titan model is `jax_titan/models/<name>.py`. It holds:

- **The configuration:** a frozen dataclass, `<Name>Config`, whose fields are
  the structural keys of the model's source under the source's names, and
  `CONFIGS`, the configurations by the names `--model` takes. A value that
  follows from other fields is a property, and `__post_init__` refuses what
  the module does not implement.
- **The parameters and the forward:** functions over a tree of fp32 arrays.
  Operators come from `jax_titan.ops`, never from a kernel library, and the
  loss from `jax_titan/models/losses.py`.
- **`class Model`:** what the trainer reads, with the members of
  `jax_titan/models/contract.py`.

| Member | What it gives |
|---|---|
| `config` | the configuration, which a run records |
| `init(seed)` | the parameters of each stage |
| `abstract(seed)` | their shapes and dtypes, without initializing |
| `stage_fns()` | one function per stage; the last takes the targets and returns the loss |
| `logical_axes(path, leaf)` | the logical axes of a parameter, which FSDP shards along |
| `decays(path, leaf)` | whether a parameter takes weight decay |

A layout uses four more members of a model that has them: `load(path)` for
`--init_from`, `merge(stage_params)` for `--save_params`, and, for a model
that trains only part of its parameters, `is_frozen(path)` with `dtype` and
`place(path, targets)`.

A model with published weights or with generation is a package,
`jax_titan/models/<name>/`: `model.py` for the above, `weights.py` for its
published weights (their config and their tensors as the model's parameters)
and `generate.py`, run as `python -m jax_titan.models.<name>.generate`. Its
`__init__.py` exports the names of `model.py` and `weights.py`, so the
package is used as a module is. `jax_titan/models/weights.py` reads
safetensors files and saves and restores parameters for every model. A
model's module imports no other model's.

## The names

Add the names of `CONFIGS` to `MODELS` in `jax_titan/models/__init__.py`,
each with the module that implements it:

```python
MODELS = {
    "llama3.1-8b": "llama",
    "llama3-test": "llama",
    ...
}
```

`tests/test_models.py` holds the two together, and the help of `--model` is
made from the table.

## The builder

Add a builder to `jax_titan/trainer/builders.py` and to `BUILDERS`, under the
module's name. A builder takes the flags, the devices and the trainer's
`make_optimizer`, and returns a `Layout` and a `Workload`
(`jax_titan/trainer/layout.py`). Llama's is the pattern:

```python
def _llama(args, devices, make_optimizer):
  _options(args, "dtype")                 # refuses the run options of other models
  _no_init_from(args)
  _one_stage(args, ops=True)              # no pipeline: --parallel=dp or fsdp
  cfg = config.resolve(llama, args.model, overrides=_overrides(args))
  mesh = _data_mesh(devices)
  model = llama.Model(cfg, dtype=_dtype(args), remat=args.remat,
                      constrain=_keep_batch_sharded(mesh))
  tokens, _ = _token_batches(args)        # the spec of a step's inputs and targets
  layout = spmd_layout(args, mesh, model, make_optimizer(model.decays), tokens, tokens, ...)
  work = Workload(num_params=..., tokens_per_step=..., flops_per_step=..., vocab_size=...)
  return layout, work
```

- **The configuration** comes from `config.resolve`: the entry of `CONFIGS`,
  or the config of the published weights that `--init_from` names, with the
  fields of `--model_overrides`. A key that is not a field is refused.
- **The layout** is `spmd_layout` without a pipeline and `mpmd_layout` with
  one. Both read the model through the contract alone.
- **The workload** says what a step processes: its tokens and FLOPs, and
  `next_batch` for a model that does not train on token windows.
- **A run option** that is no field of the configuration, such as the compute
  dtype, is a flag of the trainer, and `_options` names the ones the model
  takes.

## A model of another repository

An adapter is a module of `jax_titan/models/adapters`, named in `SOURCES`
under its source, with a builder under the adapter's module. Its class is
`AdapterModel`, with the same members; its `config` is an `AdapterConfig`,
the source, the model's name there and the overrides the source gets, and the
source's own configuration stays inside the adapter.

## Tests and pages

- `tests/test_models.py` checks the names, the builders, the contract and the
  imports of every model without a change.
- The model's own tests set it against its reference implementation, check
  that every parameter is sharded under FSDP and that only its norms take no
  weight decay, and train it through the trainer and resume it.
- A model with stages gets a case in `tests/test_mpmd_parity.py`.
- A recipe page in `docs/recipes/` says how to train it and what was
  measured.
