#!/usr/bin/env bash
# Publishes the wheels that CI installs, as a draft GitHub release: jax, jaxlib
# and the two ROCm plugin wheels from ${JAX_TITAN_ROOT}/src/jax/dist
# (scripts/build_jax_rocm.sh) and Shardy's MPMD package, built here from the
# patched tree. Run it after a new build, when a revision of
# scripts/setup_sources.sh or a patch of third_party/patches changed. It writes
# the checksums to .github/ci-wheels.sha256; then publish the draft and put its
# name into .github/workflows/ci.yml. Needs gh, logged in with write access.
#
# Environment: JAX_TITAN_ROOT, REPOSITORY (default ROCm/jax-titan), TAG (default
# jax-wheels-<build date>-<jax commit>; set it to publish one build twice).
set -euo pipefail

REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
JAX_TITAN_ROOT=${JAX_TITAN_ROOT:-$(dirname "$REPO")/.jax_titan}
REPOSITORY=${REPOSITORY:-ROCm/jax-titan}
SRC=$JAX_TITAN_ROOT/src
MPMD=shardy/integrations/python/jax/mpmd

jaxlib=$(ls "$SRC"/jax/dist/jaxlib-*.whl)
built=$(date -r "$jaxlib" +%F)
rev() { git -C "$SRC/$1" rev-parse --short=10 HEAD; }
tag=${TAG:-jax-wheels-$built-$(git -C "$SRC/jax" rev-parse --short=7 HEAD)}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/wheels"
cp "$SRC"/jax/dist/*.whl "$work/wheels/"
# The ROCm plugin wheels carry no copy of their license; the release does.
cp "$SRC/jax/LICENSE" "$work/LICENSE"
# A copy, with Shardy's license: pip writes build files into the tree it builds.
git -C "$SRC/shardy" ls-files -z -- "$MPMD" |
  rsync -a --from0 --files-from=- "$SRC/shardy/" "$work/shardy/"
cp "$SRC/shardy/LICENSE" "$work/shardy/$MPMD/"
python3 -m venv "$work/venv"
# -I, and not from inside the package: its types.py shadows the standard library's.
"$work/venv/bin/python" -I -m pip wheel --quiet --no-deps -w "$work/wheels" "$work/shardy/$MPMD"
(cd "$work" && sha256sum wheels/*.whl) > "$REPO/.github/ci-wheels.sha256"

cat > "$work/notes.md" <<EOF
Wheels that JAX-Titan's CI installs. This is not a release of JAX-Titan.

Built on $built with \`scripts/build_jax_rocm.sh\` from the trees of \`scripts/setup_sources.sh\`:

| Wheel | Built from |
|---|---|
| \`jax\`, \`jaxlib\`, \`jax_rocm10_plugin\`, \`jax_rocm10_pjrt\` | jax-ml/jax \`$(rev jax)\` with \`third_party/patches/jaxlib-*.patch\`, against openxla/xla \`$(rev xla)\` and the Shardy below |
| \`shardy_mpmd\` | \`$MPMD\` of openxla/shardy \`$(rev shardy)\` with \`third_party/patches/shardy-*.patch\` |

\`jaxlib\` runs on CPU. The two \`jax_rocm10\` wheels add AMD GPUs: they contain no ROCm libraries and need ROCm 10 from TheRock, as the container of CI's GPU job has it.

The wheels contain code of JAX, XLA and Shardy as those patches modify it, under the Apache License 2.0: the file \`LICENSE\` of this release.

\`\`\`
$(cat "$REPO/.github/ci-wheels.sha256")
\`\`\`
EOF

# A pre-release, so that GitHub never shows it as JAX-Titan's latest release.
gh release create "$tag" --repo "$REPOSITORY" --draft --prerelease \
  --title "JAX wheels for CI (jax $(rev jax), xla $(rev xla), shardy $(rev shardy))" \
  --notes-file "$work/notes.md" "$work"/wheels/*.whl "$work/LICENSE"
cat "$REPO/.github/ci-wheels.sha256"
echo "Review the draft, then: gh release edit $tag --repo $REPOSITORY --draft=false"
