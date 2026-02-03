# PyPSA Energy Crisis Hindcast -- DTU Special Course

This repository contains the analysis workflow used in the **DTU Special Course** on modeling the European energy crisis with **PyPSA**.  
The project focuses on hindcasting the **2021–2024** period using historical data, PyPSA network files, and post-processing scripts to analyze system behavior, dispatch, prices, and CO₂ emissions.

## Quickstart
Clone this repository by running the command:
```bash
git clone https://github.com/marco-saretta/pypsa-network-analyzer.git
cd pypsa-network-analyzer
```

## Environment Setup

### Using `uv` (recommended)

`uv` reads dependencies directly from `pyproject.toml`.

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh                     # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create virtual environment
uv venv

# Install dependencies from pyproject.toml
uv sync
```


### Using `conda`

`conda` cannot create environments directly from `pyproject.toml`.

```bash
# Create and activate environment
conda create -n pypsa-course python=3.11
conda activate pypsa-course

# Install project and dependencies
pip install -e .
```

## Repository Structures

The directory structure of the project looks like this:
```text
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
│
├── configs/                         # Hydra configuration files
│   ├── default_config.yaml          # Main configuration entry point
│   └── config_results_concat/
│       └── config_results_concat.yaml
│
├── data/                     # Data directory
│   ├── benchmark
│   └── network_files         # Place network files here
│
├── docs/                     # Dcoumentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
│                           
├── src/                      # Source code
│   └── pypsa_network_analyzer/
│       ├── scripts/                 # Analysis and plotting scripts
│       ├── utils/                   # Utilities and helpers
│       └── __init__.py
│
├── tests/                    # Unit tests
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model.py
│   └── __init__.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Project configuration and dependencies
├── README.md                 # This file
```

## Objective

The repository is designed to **analyze PyPSA hindcast simulations** using pre-downloaded historical data and pre-built PyPSA networks.

It supports:

- Comparison of dispatch across countries and buses
- Energy-mix breakdowns by carrier
- CO2 emissions analysis
- Electricity price diagnostics
- Installed capacity and generation statistics
- Script-based, reproducible analysis (no notebooks required)

## Configuration

The project uses **Hydra** for configuration management.  
The main entry point is:

```text
configs/default_config.yaml
```

### Default configuration structure

```yaml
defaults:
  - config_results_concat: config_results_concat
  - _self_

network_files:
  - "YOUR_NETWORK_FILE_HERE.nc"
  - "YOUR_NETWORK_FILE_HERE_1.nc"
  
paths:
  root: "${hydra:runtime.cwd}"
  log: "./logs"

api_key: YOUR-ENTSOE-API-KEY

plot_export_format: "pdf"

metrics:
  - mae
  - rmse
  - smape
```

### Key fields

- **`network_files`**  
  List of PyPSA `.nc` files to analyze.  
  Comment or uncomment blocks to switch between hindcast variants.

- **`paths.root`**  
  Base directory for relative paths. By default, this is the runtime working directory.

- **`paths.log`**  
  Directory for log files.

- **`api_key`**  
  Optional ENTSO-E API key. Required only for scripts that fetch live data.

- **`exclude_countries`**  
  Countries to omit from analysis due to data quality or model issues.

- **`metrics`**  
  Error metrics computed when comparing simulations.

## Workflow

### Using `uv`

Run scripts inside the managed environment without manual activation:

```bash
uv run main.py
```

or directly:

```bash
uv run src/pypsa_network_analyzer/scripts/network_analyzer.py
```

---

### Using `conda`

Activate the environment first:

```bash
conda activate pypsa-course
```

Then run:

```bash
python main.py
```

or:

```bash
python src/pypsa_network_analyzer/scripts/network_analyzer.py
```

Each script is self-contained and produces figures and summaries relevant to a specific analysis task.

## Data

All datasets required for the hindcast analyses  
(load, generation, prices, capacities, emissions) are either:

- already included, or
- expected to be locally available in preprocessed form.

**ENTSO-E API tokens are not included.**  
Scripts that require live ENTSO-E access are optional and clearly separated in `utils/`.

## Presentations

The `presentation/` directory contains course slides documenting the
modeling setup, data collection, and interpretation of results.

## Authors

- **Lukas Karkossa** and **Marco Saretta**  
  Technical University of Denmark  
  `{lalka, mcsr}@dtu.dk`

- **Frederik Erhard Gullach**  
  Aarhus University  
  `fegu@cowi.com`

## Template

Created using [DTU_mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).