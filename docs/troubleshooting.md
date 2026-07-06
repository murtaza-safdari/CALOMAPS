# Troubleshooting

Infrastructure-level quirks that bite when running CALOMAPS on the Fermilab Elastic Analysis Facility (EAF) and its SSHFS-mounted `/nashome` shared home. Each entry is a known pattern with a workaround.

For **project-level errors** (your `ddsim` exited non-zero, a notebook cell threw a NameError, etc.), see [handbook.md §14](handbook.md#14-common-gotchas) instead.

---

## SSHFS occasionally writes zero-byte files

**Symptom.** A `cp` (or git internal write, or `python -m venv`, or any small-file write) onto the `/nashome`-via-SSHFS mount succeeds, but the resulting file is **zero bytes** even though `ls` shows the correct size. Reading the file later reveals it's all NULs.

Hit during this session on: `elements.xml` (33 KB → 33 KB of NULs after `cp`), `analysis/dashboard.py` (Write tool wrote 0 bytes), `.git/objects/<hash>` (git's internal store, hit twice during `git commit`).

**Workaround.**

- For copying files: prefer `rsync -a` over `cp -r`. Rsync writes atomically (write-to-tempfile, rename) which dodges the issue on most attempts.
- For git operations: don't do `git init` / `git add` / `git commit` on the `/nashome`-mounted tree. Instead, rsync the working tree to a local-FS path (`/tmp/CALOMAPS-push/`), run git there, push from there, then `git clone` back from github to `/nashome` if you want a local checkout.
- After git operations on SSHFS, `git fsck --full` is cheap insurance against zero-byte objects.

**Diagnose.** `file path/to/suspect.xml` reports `data` (binary) instead of `ASCII text`; `xxd <file> | head` shows `00 00 00 …`.

---

## CVMFS Key4hep ships a CPU-only PyTorch

**Symptom.** In any environment that has the CVMFS Key4hep 2026-02-01 stack loaded, `torch.cuda.is_available()` returns `False` even on a GPU node. Worse, `torch.cuda.get_device_name(0)` raises `AssertionError: Torch not compiled with CUDA enabled` — the CVMFS torch was built without CUDA support at all.

**Workaround.** Install a CUDA-enabled torch into a venv that's *first* in your `sys.path`. See [handbook.md §11.2](handbook.md#112-training-new-models-on-the-gpu) for the full recipe. The short version:

```bash
/cvmfs/sw.hsf.org/key4hep/releases/2026-02-01/x86_64-almalinux9-gcc14.2.0-opt/python/3.13.8-z2dydk/bin/python3.13 \
    -m venv --system-site-packages /tmp/cu_torch_env
/tmp/cu_torch_env/bin/pip install --force-reinstall torch \
    --index-url https://download.pytorch.org/whl/cu121
```

Then to actually use the cu121 torch (see next entry).

---

## CVMFS PYTHONPATH shadows your venv's torch

**Symptom.** After `pip install --force-reinstall torch+cu121` into a venv with `--system-site-packages = true`, `import torch` *still* loads the CVMFS CPU-only build. Even from the venv's own Python interpreter.

**Root cause.** When `source ~/setup_calomaps.sh` (or directly `source /cvmfs/.../setup.sh`) runs, it pre-pends ~30 paths to `PYTHONPATH`, including `/cvmfs/.../py-torch/.../site-packages`. That comes **earlier** in `sys.path` than the venv's own site-packages, so `import torch` finds CVMFS first.

**Workaround.** Drop CVMFS torch paths from `sys.path` and prepend the venv's site-packages before `import torch`:

```python
import sys
sys.path = [p for p in sys.path if "py-torch" not in p]
sys.path.insert(0, "/tmp/cu_torch_env/lib/python3.13/site-packages")
import torch    # now resolves to your venv-installed cu121 build
```

**Exception.** Inside a JupyterLab notebook using the `Key4hep + GPU` kernel, the kernel spawns without `setup_calomaps.sh` having been sourced — so the venv's torch wins naturally. No shim required in that path.

---

## JupyterHub `/api/kernels` POST returns 500 mid-session

**Symptom.** Spawning a Jupyter kernel via the JupyterHub REST API (`POST /user/<u>/api/kernels`) returns `HTTP 500 Unhandled error` with an empty traceback. The user-server itself is healthy: GET on `/api/contents` works, terminals work, files are accessible — only kernel-spawn fails.

**Workaround.**

- If you're driving things programmatically and need code execution: open a **terminal** via `POST /user/<u>/api/terminals` (which stays healthy), then connect via WebSocket and run shell commands. A Python invocation as a subprocess of the terminal shell works for code execution without a Jupyter kernel.
- If you can open JupyterLab in a browser: just open a notebook normally — the kernel-spawn path through the UI is different and usually unaffected.
- Last resort: restart your user-server from the JupyterHub control panel ("Stop My Server" → "Start My Server"). This wipes container-local paths like `/tmp/cu_torch_env`, so re-do any installs.

---

## JupyterHub `/api/contents` PUT returns 500

**Symptom.** `PUT /user/<u>/api/contents/<path>` (used to programmatically write files) returns `HTTP 500` while `GET` on the same endpoint works. The home directory may be quota-pressured (see next entry).

**Workaround.** Write the file via a terminal session instead: `POST /user/<u>/api/terminals` to create a terminal, connect over WebSocket, send a command like `base64 -d <<< '<base64-of-content>' > /path/to/file`.

---

## `/home/<username>` on EAF fills up

**Symptom.** `df -h /home/$USER` shows 100% used, 0 bytes free. Subsequent writes anywhere under `/home` fail with `disk quota exceeded`, including small ones (a 1 KB symlink update).

**Diagnose.** `du -sh ~/.cache/* ~/.conda/* /home/$USER/* 2>/dev/null | sort -hr | head -10`

**Workaround.** The usual large-and-disposable culprits:
- `~/.cache/pip` — pip's wheel cache. Routinely 2-3 GB. Safe to delete: `rm -rf ~/.cache/pip`.
- `~/.conda/pkgs` — conda's package cache. Hundreds of MB. Safe to delete (`conda clean --packages` or `rm -rf ~/.conda/pkgs`).
- Old project data — large `.root` / `.npz` files left behind by previous experiments. Move to `/exp/...` (huge persistent CephFS) or delete.

If you want a persistent-but-isolated install of something big (e.g., a CUDA torch venv) and `/home` is tight, install to `/tmp/<name>` instead — EAF's `/tmp` is an overlay filesystem with several TB free. The cost: `/tmp` is wiped on container restart.

---

## macOS resource-fork files (`._*`) leak into `.git/`

**Symptom.** After `git clone` of a CALOMAPS repo onto a `/nashome` SSHFS mount, `git fsck --full` complains:

```
error: refs/heads/._main: badRefName: invalid refname format
error: refs/heads/._main: badRefContent:
error: refs/._heads: badRefName: invalid refname format
... etc
```

**Root cause.** macOS creates `._<name>` shadow files to store extended attributes on filesystems that don't support them natively. SSHFS surfaces these through the mount, and git's ref scanner picks them up as malformed ref names.

**Workaround.** They're harmless noise. Either ignore them, or sweep them out periodically:

```bash
find .git -name '._*' -type f -delete
```

(Same applies to any `._*` you find scattered through the work tree. The `.gitignore` already excludes them from being tracked.)

---

## `Key4hep + GPU` Jupyter kernel won't spawn via REST API

**Symptom.** `POST /user/<u>/api/kernels` with `{"name": "my_gpu_env"}` either returns 500 or returns 200 but the WebSocket connection drops immediately when you try to execute code.

**Root cause.** The `my_gpu_env` kernel.json points at `/home/<user>/my_gpu_env/bin/python` directly. That python expects CVMFS to be visible (because `my_gpu_env` inherits `--system-site-packages` from a CVMFS Python). When JupyterLab UI spawns the kernel, CVMFS is pre-set in the spawn environment. When the bare REST API spawns it, CVMFS is **not** pre-set, and the kernel fails on `import` of CVMFS-provided modules.

**Workaround.**

- For interactive notebook work: open the notebook in **JupyterLab UI**, pick the `Key4hep + GPU` kernel from the kernel selector. This path works.
- For programmatic code execution: use the **`Python (Key4hep)`** kernel (different kernelspec, sources CVMFS in its own launcher) — it spawns cleanly from the REST API. Or use a terminal + subprocess, as in the API-500 workaround above.

---

## File mode bits (the `+x` executable flag) get lost via SSHFS on clone

**Symptom.** Immediately after `git clone` into `/nashome` you have an unstaged "modified" set:

```
modified:   sim/generate_batched.sh
modified:   sim/generate_dataset.sh
```

`git diff` shows no content change, just `old mode 100755 → new mode 100644`. The clone failed to preserve the executable bit.

**Workaround.** Re-add it manually after clone:

```bash
chmod +x sim/*.sh
```

Then `git status` is clean again. Permanent fix: `git config core.fileMode false` if you don't want git tracking file modes on this checkout at all — but that hides the issue rather than fixing it.

---

## "Key4hep + GPU" kernel dies on launch from nbconvert / freshly-spawned shells

**Symptom.** A GPU kernel whose `kernel.json` runs the venv python directly
(`~/my_gpu_env/bin/python -m ipykernel_launcher`) starts fine from the JupyterLab
UI but **dies immediately** when launched by `nbconvert` (or any non-interactive
spawn), with `No module named ipykernel_launcher` /
`Kernel died before replying to kernel_info`.

**Root cause.** `ipykernel` (and numpy/scipy/...) live in CVMFS prefixes that are
only added to `PYTHONPATH` by `source setup_calomaps.sh`. An interactive terminal
that sourced CVMFS earlier *looks* fine, but a kernel spawned with a clean
environment never gets those paths — and the venv (created `--system-site-packages`
from the CVMFS python, which does **not** carry ipykernel in its base
site-packages) can't find `ipykernel_launcher`.

**Workaround.** Make the GPU kernel **self-contained** and CVMFS-independent:
install everything the notebook needs *into* the venv with a clean `PYTHONPATH`,
and launch via a wrapper that clears `PYTHONPATH` so the venv's cu121 torch wins:

```bash
env -u PYTHONPATH "$VENV/bin/pip" install --no-cache-dir --ignore-installed \
    numpy scipy matplotlib ipykernel
env -u PYTHONPATH "$VENV/bin/pip" install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cu121
# wrapper.sh:   unset PYTHONPATH PYTHONHOME; exec "$VENV/bin/python" -m ipykernel_launcher "$@"
```

`setup/setup_gpu_kernel.sh` does exactly this. Notebook 03 needs only
torch/numpy/scipy/matplotlib (no ROOT/uproot), so a self-contained venv covers it
without CVMFS at all. This supersedes the older `~/my_gpu_env` + `sys.path`-shim
recipe for the *notebook* path (the shim is still fine for headless scripts).

---

## ddsim EDM4hep output crashes if the steering file contains non-ASCII characters

Writing EDM4hep ROOT output with `ddsim` can abort with
`cppyy.gbl.dd4hep.unrelated_value_error: ... std::map<string,string> ... is not defined`
(a traceback, exit 1) right after `++++ Setting up EDM4hep ROOT Output ++++`, with no
output file. (If you try to "fix" it by monkeypatching the RunHeader assignment, it then
turns into a *silent* C++ `exit(0)` with no traceback at the next metadata assignment,
which is easy to misread as an OOM kill or a hang.)

**Root cause:** a **non-ASCII character in the steering file** (an em-dash `-` typed as the
Unicode em-dash, an en-dash, or a "smart quote" in a comment/docstring). ddsim copies the
entire steering-file text into the run metadata (`SteeringFileContent`), then hands the
metadata dict to the EDM4hep writer's `RunHeader` (a `std::map<string,string>` property).
cppyy cannot convert a dict whose values contain non-ASCII bytes and reports it as the map
type being "not defined". It is **not** a Key4hep build bug, **not** memory/OOM, and **not**
release-specific (reproduced identically on 2026-02-01 and 2026-04-08; an ASCII steering
works on both).

**Fix:** keep steering files **pure ASCII** -- use `-` not the em-dash, and straight quotes
`'` `"` not curly ones. One-liner to clean a file:

```bash
python -c "import sys,io; p=sys.argv[1]; io.open(p,'w').write(io.open(p,encoding='utf-8').read().encode('ascii','ignore').decode())" sim/run_sim.py
```

(`SIM.outputConfig.forceDD4HEP = True` also sidesteps the crash by writing DD4hep-native
ROOT instead of EDM4hep, but that's a *different output format* — not what the uproot-based
extractors here read — so it is not the EDM4hep fix.)

## Geant4 shower cascade is dropped unless you disable the user particle handler

`part.keepAllParticles = True` **alone does not** persist a calorimeter EM shower's
secondaries. ddsim's default `Geant4TCUserParticleHandler` restricts MC-truth to the inner
*tracking* region; particles born outside it (ECal shower secondaries — at r > 1267 mm in the
DECAL barrel) get merged into their parent and never written, even with `keepAllParticles`.
The symptom is "only the primary survived" (1 MCParticle, 0 daughters) despite thousands of
silicon hits proving the shower happened.

Fix: disable the user particle handler so the region cut is removed, and keep all particles:

```python
SIM.part.userParticleHandler = ""
SIM.part.keepAllParticles    = True
```

With both, the complete cascade is retained (~78k particles for a single 50 GeV photon). See
`sim/run_sim_fullcascade.py`.

## EDM4hep podio writer crashes (free(): invalid pointer) on very large events

On Key4hep 2026-02-01, writing a *high-multiplicity* EDM4hep event -- e.g. a full shower
cascade with `keepAllParticles=True` + `userParticleHandler=""` (~78k MCParticles) -- crashes
at the END of the run with `free(): invalid pointer` inside `podio::ROOTWriter::finish()` ->
`TFile::Close`. The event data is written first ("Saving EDM4hep event 0"); the abort happens
during file *finalization*. Standard 1-particle samples are unaffected.

Cause: a podio/ROOT memory bug in the 2026-02-01 podio (1.7), triggered when finalizing a
file with this many objects/relations. It is **fixed in Key4hep 2026-04-08** (newer podio).

Fix: generate large-cascade EDM4hep samples under 2026-04-08:

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08
export LD_LIBRARY_PATH="$HOME/lib_hack:$LD_LIBRARY_PATH"
```

Reading the output with uproot is release-agnostic, so the analysis notebooks can stay on the
pinned 2026-02-01. (This is a *different* failure from the non-ASCII-steering EDM4hep crash
documented above.)

---

## Training the ensembles on a CPU-only EAF session thrashes across all cores

Training the quantile ensembles without a GPU (a CPU-only EAF session, or `cuda=False`
because CVMFS ships CPU-only torch) can stall: PyTorch parallelizes each tiny full-batch
step across every core, and for a model this small (~4,400 parameters, full-batch) the
threading overhead dominates. You will see ~2000% CPU with *no* per-model progress for
many minutes. Pin it to a single thread and it runs far faster (~15-30 s per model,
~30 min for all four ensembles on CPU):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python analysis/train_ensembles.py --particle gamma
```

A GPU session is still faster end-to-end; this is the fallback when none is available.

---

## Where else to check

- **handbook.md §14** — code-level errors and project-specific gotchas (CUDA torch, cell-19/26 bugs, scripts that wipe data dirs, etc.)
- **`~/lpc-tools/lpc-setup.html`** — the user's personal LPC/EAF onboarding handbook on their laptop, with broader-than-CALOMAPS Fermilab notes
- The DD4hep, Geant4, and Key4hep upstream issue trackers if you're hitting something deeper.
