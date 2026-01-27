# PyPSA Energy Crisis Hindcast -- DTU Special Course

This repository contains the analysis workflow used in the DTU Special
Course on modeling the European energy crisis with **PyPSA**. The
project focuses on hindcasting the 2021--2024 period by collecting
historical data, running PyPSA networks, and producing comparative
analytics on system behavior, dispatch, prices, and emissions.

## Repository Structure

    data/
        api_output/
        capacity/
        co2_emissions/
        fuel_prices/
        generation/
        generators_list/
        load/
        marginal_cost/
        networks/                <- place your PyPSA network files here
        prices/
        solar_pv_datasets/
        wind_farms_datasets/

    presentation/
        *.pptx                    <- course presentation material

    results/
        benchmark/
        co2_comparison_plots/
        hindcast_dynamic_*        <- year-specific hindcast runs
        hindcast_standard_*       <- year-specific hindcast runs
        scripts_output/

    scripts/
        dataframe_merger.py
        dispatch_compare.py
        ENTSOE_api_raw.py
        entsoe_api.py
        gas_prices_retrieve.py
        network_analyzer.py
        plot_co2_emissions.py
        plot_prices.py
        result_scorer.py
        ...

    utils/
        ppt_comparison/




The directory structure of the project looks like this:
```bash
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


## Objective

The repository is designed to **analyze PyPSA hindcast simulations**
using pre-downloaded historical data. It supports:

-   Comparing dispatch across countries and buses\
-   Generating energy-mix breakdowns\
-   Visualizing CO₂ emissions and marginal costs\
-   Benchmarking dynamic vs. standard hindcast runs\
-   Producing diagnostic plots for prices, fuels, loads, and more

## Workflow

1.  Place your PyPSA network files inside:

``` bash
data/networks/
```

2.  Create the environment:

``` bash
conda env create -f environment.yaml
conda activate pypsa-course
```

3.  Run:

``` bash
python main.py
```

4.  The pipeline generates:

``` bash
results/simulations/
```
containing:

-   country-level energy mixes\
-   price and dispatch comparisons\
-   CO₂ emissions summaries\
-   diagnostic figures

## Data

All datasets required for hindcasting (load, generation, prices,
capacities, emissions, etc.) are already included or preprocessed.
External tokens (e.g., ENTSO-E) are **not included** and not required
for running the existing workflow.

## Presentations

The `presentation/` directory contains course slides documenting the
modeling setup, data collection, and interpretation of results.

## Authors

Authors of this repository are:
-  Lukas Karkossa and Marco Saretta from Technical University of Denmark,  {lalka, mcsr}@dtu.dk
-  Frederik Erhard Gullach from university of Aarhus, fegu@cowi.com.

## Template

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).