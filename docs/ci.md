# Continuous integration

What CI runs on every pull request and on `main`, how a test joins the CPU or
the GPU job, how to run the same tests locally, and how to update the packages
CI installs. The workflow is
[.github/workflows/ci.yml](../.github/workflows/ci.yml).

## The two jobs

The workflow `CI` runs on every pull request and on every push to `main`. A
new push to a pull request cancels the run of the older one, and
`gh workflow run ci.yml --ref BRANCH` starts a run by hand.

| | CPU tests | GPU tests (MI350) |
|---|---|---|
| Runner | GitHub's `ubuntu-24.04`, with Python 3.12 | Self-hosted, with one gfx950 GPU, in the TheRock 10.1 container |
| JAX | jax, jaxlib and Shardy's MPMD package, built from source | The same, with the two ROCm plugin wheels of that build |
| JAX-Titan | With MaxText and MaxDiffusion | Without them |
| Command | `python -m pytest tests/` | `python -m pytest tests/ --gpus=1` |
| Tests | Every test not marked `gpu`, on 8 CPU devices | The tests marked `gpu` that need one GPU |
| Time | About 25 minutes | About 2 minutes |

Both jobs test one stack: the wheels of one release, and every other package
at the version [.github/ci-constraints.txt](../.github/ci-constraints.txt)
names. A pull request also runs the workflow `Security scan (PR)`, which this
page does not cover.

## CPU tests and GPU tests

A test in `tests/` is a CPU test unless it is marked `gpu`:

```python
def test_on_cpu():  # no marker: a CPU test
  ...


@pytest.mark.gpu  # needs one GPU
def test_on_gpu():
  ...


@pytest.mark.gpu(4)  # needs four GPUs
def test_on_4_gpus():
  ...
```

The marker alone decides which job runs a test, so a new test needs no change
to `ci.yml`. [tests/conftest.py](../tests/conftest.py) makes each pytest
command run one kind:

- **`python -m pytest tests/`** runs the CPU tests, with JAX on 8 CPU devices,
  and never a `gpu` test.
- **`python -m pytest tests/ --gpus=N`** runs only the `gpu` tests, with JAX
  on ROCm, and skips those that need more than N GPUs. When JAX sees fewer
  than N GPUs, it stops with an error instead of running on CPU.

## Running the tests locally

The environment of [getting_started.md](getting_started.md) runs the same two
commands:

```bash
python -m pytest tests/             # the CPU job's tests
python -m pytest tests/ --gpus=1    # the GPU job's tests; needs a free GPU
```

When a test fails in CI and passes locally, two things can differ:

- **Skipped tests.** The MaxText and MaxDiffusion tests are skipped where
  those packages are not installed, and they run in CI. Ideogram 4's
  comparisons with diffusers', transformers' and the Ideogram branch's models
  and Command A+'s comparisons with transformers' model and processor are
  skipped in CI, which has neither `torch` nor that branch: they run only
  where these are installed. pytest ends with a line for every skipped test
  and its reason.
- **Package versions.** The step `Show installed versions` of each job prints
  every package it installed, to compare with `python -m pip freeze`.

For the CPU job's exact packages, run its `run:` lines in a new Python 3.12
environment, with `WHEELS` (the release's name), `GITHUB_REPOSITORY`
(`ROCm/jax-titan`), `GITHUB_WORKSPACE` (the checkout) and `RUNNER_TEMP` (an
empty directory) set.

## The packages CI installs

Every package CI installs has its version fixed in the repository:

| Package | Fixed by | To update |
|---|---|---|
| jax, jaxlib, the ROCm plugin wheels and Shardy's MPMD package | `WHEELS` in `ci.yml`, which names a release of this repository, and the checksums in `.github/ci-wheels.sha256` | Publish new wheels (below) |
| MaxText | `maxtext==0.2.5` in the `maxtext` extra of `pyproject.toml` | Change the pin, and its line in the constraints file |
| MaxDiffusion | `PINNED_COMMIT` in `src/jax_titan/models/adapters/maxdiffusion.py`, with `third_party/patches/maxdiffusion-*.patch` | Change the commit or the patches: every run applies them to a new checkout |
| Every other Python package | `.github/ci-constraints.txt` | Replace its lines, as its first lines say |
| ROCm 10.1 and the rest of the GPU job's system | The digest of the container image in `ci.yml` | Change the digest |
| GitHub Actions | Commit hashes in `ci.yml` | Dependabot opens the pull requests |

### New wheels

The wheels hold jaxlib and Shardy with the patches of
[third_party/patches](../third_party/patches), built at the revisions of
`scripts/setup_sources.sh`. After a change to one of those patches or
revisions, CI keeps testing the old build until `WHEELS` names a new one:

1. Build the wheels with `scripts/build_jax_rocm.sh`
   ([getting_started.md](getting_started.md#3-build)).
2. Run `scripts/publish_ci_wheels.sh` on the host. It uploads the wheels as a
   draft release and writes their checksums to `.github/ci-wheels.sha256`.
3. Review the draft, and publish it with the `gh release edit` command the
   script prints.
4. Put the release's name into `WHEELS` in `ci.yml`, and commit it with the
   checksum file.

## Pitfalls

- **Old wheels.** A change to a jaxlib or Shardy patch, or to a revision in
  `scripts/setup_sources.sh`, passes CI untested until `WHEELS` names a build
  that has it.
- **A package without a line.** pip installs a package that
  `.github/ci-constraints.txt` does not name at its newest version. Add its
  line together with a new dependency.
- **One GPU.** The runner gives a job one GPU, so CI skips a test marked
  `gpu(2)` or higher. Run it by hand on a node with that many GPUs, with
  `--gpus=N`.
- **No autotuning.** Unless `XLA_FLAGS` is set, `--gpus` makes XLA run
  hipBLASLt's first choice for every GEMM
  (`--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false`). With
  XLA's defaults, compiling takes the GPU tests from 90 seconds to 20 minutes.
- **Forks.** The GPU job does not run for a pull request from a fork: the
  self-hosted runner never runs a fork's code.
- **Memory.** The CPU runner has 16 GB of memory, and the tests peak near
  10 GB.
