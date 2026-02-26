# navigoquest

Path and navigation analysis for the [SeaHeroQuest](https://seaheroquest.com/) dataset.

## Installation

The package is published to [PyPI](https://pypi.org/) and can be installed directly using

```bash
$ pip install navigoquest
```

## User guide

Coming soon.

## Contributing

The package is written in Python (minimal version: 3.12).
We recommend that the installation is made inside a virtual environment.
While you can use `conda` or Python's built-in `venv`, we recommend `uv` ([https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)).

### Local install with `uv`

If you don't have `uv` installed, follow the instructions on their [documentation](https://docs.astral.sh/uv/getting-started/installation/).
Once `uv` is installed, navigate to the project's root directory and run:

```bash
$ uv sync
```

This command creates a virtual environment in the `.venv` directory and installs all project dependencies.

You have two options for using the virtual environment:

1. Activate the environment using:

```bash
$ source .venv/bin/activate
```

2. Execute scripts or programs directly within the virtual environment by prefixing your commands with `uv run`:

```bash
$ uv run python my_script.py
$ uv run pytest tests/
```

For more information on using `uv`, please refer to the [official documentation](https://docs.astral.sh/uv/).
