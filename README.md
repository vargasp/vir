# vir

Shared Python library intended for reuse across multiple projects.

## Repository structure

#```text
common_lib_a/
├── README.md
├── pyproject.toml
├── environment.yml
├── src/
│   └── common_lib_a/
│       ├── __init__.py
│       ├── common_utils.py
│       └── ...
├── tests/
├── data/
└── notebooks/
#```plaintext

## Purpose of folders

- `src/common_lib_a/`: importable library code
- `tests/`: automated tests for the library
- `data/`: small sample or test data that should move with the repo
- `notebooks/`: Jupyter notebooks for exploration and trying new ideas

## Create the environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate common-lib-a
```plaintext

## Install the library in editable mode

From the repository root:

```bash
pip install -e .
```plaintext

Editable install means source changes are picked up without reinstalling.

## Verify installation

```bash
python -c "import common_lib_a; print(common_lib_a.__file__)"
```plaintext

## Run tests

From the repository root:

```bash
pytest
```plaintext

## Spyder setup

This environment includes:

- `spyder-kernels`

If Spyder is installed in a separate `dev-tools` environment, you can still point Spyder to this environment as the interpreter/kernel.

## Jupyter setup

This environment includes:

- `ipykernel`

Register it as a Jupyter kernel:

```bash
python -m ipykernel install --user --name common-lib-a --display-name "Python (common-lib-a)"
```plaintext

Then select `Python (common-lib-a)` inside Jupyter.

## Using this library from a project

Install the library first in the target project environment:

```bash
conda activate project1
pip install -e /path/to/common_lib_a
```plaintext

Then project code can import it like:

```python
from common_lib_a import common_utils
```plaintext

## Notebook imports

Notebooks should import package code normally, for example:

```python
from common_lib_a import common_utils
```plaintext

Avoid manual path modifications like `sys.path.append(...)`.

## Notes

- Keep reusable logic in `src/common_lib_a/`
- Use notebooks for exploration, not as the main home of production code
- Keep only small, appropriate files in `data/`