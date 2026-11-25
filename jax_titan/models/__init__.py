"""Model definitions.

`MODELS` and `SOURCES` tie every name that `--model` takes to the module that
implements it: a JAX-Titan model by its name, a model of another repository
as `<source>:<name>` by its source. The modules are named relative to this
package and imported where they are used.
"""

# --model=<name> -> the module of the JAX-Titan model.
MODELS = {
    "llama3.1-8b": "llama",
    "llama3-test": "llama",
    "flux-dev": "flux",
    "flux-schnell": "flux",
    "ideogram4": "ideogram",
    "command-a-plus": "command_a",
}
# --model=<source>:<name> -> the module of the source's adapter.
SOURCES = {
    "maxtext": "adapters.maxtext",
    "maxdiffusion": "adapters.maxdiffusion",
    "transformers": "adapters.transformers",
    "diffusers": "adapters.diffusers",
}


def module_of(name: str) -> str:
  """The module that implements `--model=name`."""
  if name in MODELS:
    return MODELS[name]
  source, colon, _ = name.partition(":")
  if colon and source in SOURCES:
    return SOURCES[source]
  raise ValueError(
      f"unknown model {name!r}; known: {', '.join(MODELS)}, and <source>:<name> for a source of "
      f"{', '.join(SOURCES)}")
