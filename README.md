# PyPSA Network Analyzer
 
A post-processing and analysis toolkit for PyPSA hindcast simulations, developed to support the paper:
 
> *"Can Optimal Dispatch Models Recreate Reality? A Retrospective Analysis of Europe's Energy Crisis Using PyPSA-Eur"*  
> Presented at EEM26, Trondheim - [https://www.ntnu.edu/eem26](https://www.ntnu.edu/eem26)

The tool compares simulated dispatch, prices, generation mix, and CO₂ emissions against historical benchmarks. While the repository is set up to reproduce the paper results, it is designed to work with **any PyPSA `.nc` network file**.

## Quickstart
 
Clone the repository locally: 
```bash
git clone https://github.com/marco-saretta/pypsa-network-analyzer.git
cd pypsa-network-analyzer
```
Setup the environment using **`uv` (recommended):**
```bash
uv sync
```
or using **`conda`:**
```bash
conda create -n pypsa-course python=3.11 && conda activate pypsa-course
pip install -e .
```

## Reproducing the Paper Results
 
The paper uses 15 network files - one per year (2020–2024) across three scenarios:
 
| Scenario | Fuel prices | Solving approach |
|---|---|---|
| Hindcast standard | Static | Full horizon at once |
| Hindcast dynamic | Dynamic | Full horizon at once |
| Hindcast dynamic rolling horizon | Dynamic | Two-week rolling horizon |
 
### 1. Download the network files

The 15 solved PyPSA network files are available at:
 
> [**DTU Data**](https://data.dtu.dk/account/articles/31248568)


Place the downloaded `.nc` files in:

```
data/network_files/     <-- Place the 15 ".nc" files downloaded here
```

### 2. Configure the run
 
`configs/default_config.yaml` is pre-configured with the correct file names. To run a subset of scenarios, comment/uncomment the relevant entries under `network_files`:
 
```yaml
network_files:
  - "hindcast_standard_2020.nc"
  - "hindcast_standard_2021.nc"
  # ...
```
 
### 3. Run the analysis
 
First, run `main.py` to process the network files and produce the results CSVs:
 
```bash
# with uv
uv run main.py          

# with conda
python main.py          
```
 
Then generate the paper figures:
 
```bash
# with uv
uv run src/pypsa_network_analyzer/plot_figures_paper.py     

# with conda
python src/pypsa_network_analyzer/plot_figures_paper.py     
```
 
Figures are saved to `figures_paper/`.

## Repository Structure
 
```text
├── .github/                        # GitHub Actions and Dependabot
│
├── configs/                        # Hydra configuration files
│   ├── default_config.yaml         # Main configuration entry point
│   └── config_results_concat/
│       └── config_results_concat.yaml
│
├── data/
│   ├── benchmark/                  # Benchmark data for scoring
│   └── network_files/              # Place PyPSA .nc files here
│
├── docs/                           # Documentation
├── figures_paper/                  # Output figures for the paper
├── logs/                           # Log files
├── notebooks/                      # Jupyter notebooks for exploration
├── outputs/                        # General output files
├── results/                        # Analysis results (CSV)
├── results_concat/                 # Concatenated multi-year results
│
├── src/
│   └── pypsa_network_analyzer/
│       ├── utils/                  # Utilities and helpers
│       ├── __init__.py             # Import packages
│       ├── entsoe_retrieval.py     # ENTSO-E data fetching
│       ├── network_analyzer.py     # Core analysis per network file
│       ├── plot_figures_paper.py   # Figure generation for the paper
│       └── score_analyzer.py       # Scoring vs benchmark
│
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── LICENSE
├── main.py                         # Main entry point
├── pyproject.toml                  # Project config and dependencies
├── README.md                       # This file :-)
└── uv.lock
```

## Other Repository Functionalities
 
### Running the network analyzer on a single file
 
To analyze a single PyPSA network file directly, without going through `main.py`:
 
```bash
uv run src/pypsa_network_analyzer/network_analyzer.py      # with uv
python src/pypsa_network_analyzer/network_analyzer.py      # with conda
```
 
### Using the tool with your own networks
 
Place your `.nc` files in `data/network_files/` and update `configs/default_config.yaml`:
 
```yaml
network_files:
  - "your_network_file.nc"
```
 
Then run `main.py` as above. The tool will produce dispatch comparisons, price diagnostics, energy-mix breakdowns, CO₂ emissions analysis, and benchmark scores for any valid PyPSA network.
 
### Fetching live ENTSO-E data
 
Scripts that retrieve live data from ENTSO-E are available in `utils/` via `entsoe_retrieval.py`. An API key is required and can be set in `configs/default_config.yaml`:
 
```yaml
api_key: YOUR-ENTSOE-API-KEY
```


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

## Data

All datasets required for the hindcast analyses  
(load, generation, prices, capacities, emissions) are either:

- already included, or
- expected to be locally available in preprocessed form.

**ENTSO-E API tokens are not included.**  
Scripts that require live ENTSO-E access are optional and clearly separated in `utils/`.


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
