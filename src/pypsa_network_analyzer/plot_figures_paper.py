import hydra
from omegaconf import DictConfig
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@hydra.main(
    version_base=None,
    config_name="default_config",
    config_path="../../configs",
)
def main(cfg: DictConfig):
    # --- paths ---
    root_dir = Path(cfg.paths.root)
    results_root = root_dir / "results_concat"
    figures_dir = root_dir / "figures_paper"
    figures_dir.mkdir(exist_ok=True)

    # --- seaborn style (paper-like) ---
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.1,
    )

    records = []

    # --- loop over simulation groups ---
    for sim_name in cfg.config_results_concat.keys():
        csv_path = results_root / sim_name / "electricity_prices" / "scores" / "scores_mae.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")

        # rows = years, columns = countries
        df = pd.read_csv(csv_path, index_col=0)

        # long / tidy format
        df_long = df.reset_index(names="year").melt(
            id_vars="year",
            var_name="country",
            value_name="mae",
        )

        df_long["simulation"] = sim_name
        records.append(df_long)

    # concatenate all simulations
    long_df = pd.concat(records, ignore_index=True)

    # ensure year is treated as categorical and ordered
    long_df["year"] = long_df["year"].astype(str)
    year_order = sorted(long_df["year"].unique())

    # --- plot ---
    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=long_df,
        x="simulation",
        y="mae",
        hue="year",
        hue_order=year_order,
        palette="Set2",
        width=0.7,
        linewidth=1.1,
        showfliers=False,
    )

    plt.ylabel("MAE")
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.legend(title="Year", frameon=False)
    plt.tight_layout()

    plt.savefig(figures_dir / "mae_by_simulation_and_year.pdf")
    plt.close()


if __name__ == "__main__":
    main()
