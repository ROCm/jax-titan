"""Published weights read from safetensors files, without PyTorch, and the
parameters a run saves (`--save_params`) and starts from (`--init_from`).

A safetensors file is a JSON header of names, dtypes, shapes and byte ranges,
followed by the tensors. Ideogram's fp8 checkpoints store a Linear weight as
float8 e4m3 with one fp32 scale per output row, `<name>.weight_scale`; the
weight is their product.
"""

import dataclasses
import json
import pathlib
import struct

import ml_dtypes
import numpy as np

_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16,
    "F8_E4M3": ml_dtypes.float8_e4m3fn, "F8_E5M2": ml_dtypes.float8_e5m2,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
    "BOOL": np.bool_,
}
_NAMES = {np.dtype(v): k for k, v in _DTYPES.items()}


def is_safetensors(path) -> bool:
  """Whether `path` is a safetensors file or a directory that holds some."""
  path = pathlib.Path(path)
  return path.suffix == ".safetensors" or (path.is_dir() and any(path.glob("*.safetensors")))


def config(cls, path, select=lambda values: values):
  """`cls`, a dataclass, from the fields it shares with the `config.json` beside
  the published weights at `path`; `select` picks the part of the file to read."""
  path = pathlib.Path(path)
  with open((path if path.is_dir() else path.parent) / "config.json", encoding="utf-8") as f:
    values = select(json.load(f))
  names = {f.name for f in dataclasses.fields(cls)}
  return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items() if k in names})


def to_device(tree, dtype=np.float32):
  """`tree` on the default device in `dtype`, cast there: a host cast of fp32
  weights to bf16 is slow."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  return jax.tree.map(lambda x: jnp.asarray(x).astype(dtype), tree)


def read(path) -> dict[str, np.ndarray]:
  """The tensors by name, mapped read-only into memory, of a safetensors file or
  of every safetensors file in a directory."""
  path = pathlib.Path(path)
  files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
  if not files:
    raise FileNotFoundError(f"no safetensors file in {path}")
  tensors = {}
  for file in files:
    with open(file, "rb") as f:
      (size,) = struct.unpack("<Q", f.read(8))
      header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    data = np.memmap(file, dtype=np.uint8, mode="r", offset=8 + size)
    for name, entry in header.items():
      if name in tensors:
        raise ValueError(f"{name} is in more than one file of {path}")
      start, end = entry["data_offsets"]
      tensors[name] = data[start:end].view(_DTYPES[entry["dtype"]]).reshape(entry["shape"])
  return tensors


def write(path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
  """Writes `tensors` as one safetensors file, with `metadata` in its header."""
  header, offset, blobs = ({"__metadata__": metadata} if metadata else {}), 0, []
  for name, value in tensors.items():
    value = np.asarray(value, order="C")
    blob = value.tobytes()
    header[name] = {"dtype": _NAMES[value.dtype], "shape": list(value.shape),
                    "data_offsets": [offset, offset + len(blob)]}
    offset += len(blob)
    blobs.append(blob)
  text = json.dumps(header, separators=(",", ":")).encode()
  text += b" " * (-len(text) % 8)
  with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(text)))
    f.write(text)
    for blob in blobs:
      f.write(blob)


def weight(tensors: dict, name: str) -> np.ndarray:
  """Tensor `name` in fp32, times its row scales when `<name>_scale` exists."""
  w = np.asarray(tensors[name]).astype(np.float32)
  scale = tensors.get(name + "_scale")
  if scale is not None:
    w *= np.asarray(scale, np.float32).reshape((-1,) + (1,) * (w.ndim - 1))
  return w


def save_params(path, params) -> None:
  """Writes a pytree of parameters, for a model to start from (`--init_from`)."""
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top

  checkpointer = ocp.StandardCheckpointer()
  checkpointer.save(pathlib.Path(path).absolute(), params)
  checkpointer.wait_until_finished()
  checkpointer.close()


def restore_params(path, target):
  """Returns the parameters `save_params` wrote; `target` is a pytree of arrays
  or ShapeDtypeStructs giving the shapes, dtypes and shardings to restore to."""
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top

  checkpointer = ocp.StandardCheckpointer()
  params = checkpointer.restore(pathlib.Path(path).absolute(), target)
  checkpointer.close()
  return params


def saved_tree(path):
  """The tree of the parameters `save_params` wrote, its leaves their shapes
  and dtypes, without reading an array."""
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top

  checkpointer = ocp.StandardCheckpointer()
  tree = checkpointer.metadata(pathlib.Path(path).absolute()).item_metadata.tree
  checkpointer.close()
  return tree


def restore_saved(path):
  """Returns the parameters `save_params` wrote, whatever their tree, on the
  first local device."""
  import jax  # pylint: disable=g-import-not-at-top

  device = jax.sharding.make_single_device_sharding(jax.local_devices()[0])
  target = jax.tree.map(
      lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=device),
      saved_tree(path))
  return restore_params(path, target)
