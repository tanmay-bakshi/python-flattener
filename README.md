# python-flattener

Builds a **single** self-extracting Python file (one massive `.py`) from:

- an app entry (a `.py` file or a directory)
- a `requirements.txt`
- a target platform spec (Rust-like target triple *or* pip `--platform` tag)

The generated bundle is runnable on a fresh Python install on the target machine (it will extract embedded files to a cache directory at runtime; no `pip install` required).

## Usage

```bash
python3 -m python_flattener build \
  path/to/app.py \
  -r path/to/requirements.txt \
  -o dist/bundle.py \
  --target native
```

Cross-platform example (downloads wheels for Linux x86_64):

```bash
python3 -m python_flattener build \
  path/to/app_dir \
  -r path/to/requirements.txt \
  -o dist/bundle_linux.py \
  --target x86_64-unknown-linux-gnu \
  --python-version 3.12 \
  --implementation cp \
  --abi cp312
```

## Entrypoint Rules

- If input is a **file**: that file is executed.
- If input is a **directory**:
  - If it has `__init__.py` (looks like a package):
    - default is `python -m <package_dir_name>` if `__main__.py` exists
    - otherwise it tries `<package>/main.py`
  - If it is not a package:
    - it tries `__main__.py`, then `main.py`
  - You can override with `--entry <path>` (relative to the input directory) or `--module <module.name>`.

## Runtime Behavior

- Extracts embedded `app/` + `deps/` to a cache directory:
  - default: `tempfile.gettempdir()/python_flattener_cache/<sha256>`
  - override with: `PYTHON_FLATTENER_CACHE_DIR=/some/path`
- Adds `app/` then `deps/` to `sys.path`
- Runs the entrypoint

## Notes / Limitations

- Cross-platform builds require **wheels** for the target (pip is invoked with `--only-binary=:all:` when constraints are set).
- Some binary wheels rely on OS-level shared libraries; a “fresh Python install” on the target OS still needs those OS libraries present.

