"""Enough of torch's namespace for FlyDSL 0.5.0's compiler to import without PyTorch.

FlyDSL registers torch types when it loads; JAX-Titan hands it CPU placeholders
through DLPack, which never touches torch. Only the FlyDSL export subprocess
puts this directory on sys.path: other packages may take a `torch` module as a
sign that PyTorch is installed.
"""

__version__ = "0.0.0+stub"


class dtype:  # noqa: N801
  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return f"torch.{self.name}"


for _name in ("float16", "bfloat16", "float32", "float64", "bool", "uint8", "int8", "int16",
              "int32", "int64", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
  globals()[_name] = dtype(_name)


class Tensor:
  pass


class _Cuda:

  class Stream:
    pass

  @staticmethod
  def is_available():
    return False


cuda = _Cuda()
