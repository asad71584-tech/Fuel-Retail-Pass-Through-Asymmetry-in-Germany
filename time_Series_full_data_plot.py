import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# DATA PATH (FINAL – CONFIRMED)
# ============================================================
DATA_PATH = r"D:\python\tankerkoenig_panel 23-25(6)\panel_prices_brent_2023_2025H1.parquet"

if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Dataset not found at:\n{DATA_PATH}")

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_parquet(DATA_PATH)

print("Columns in parquet:")
print(df.columns.tolist())

# ============================================================
# COLUMN NAMES (FROM YOUR DATA)
# ============================================================
DATE_COL   = "date"
DIESEL_COL = "diesel"
E5_COL     = "e5"
BRENT_COL  = "Brent_EUR_per_Litre"

# ============================================================
# DAILY AVERAGES
# ============================================================
df[DATE_COL] = pd.to_datetime(df[DATE_COL])

daily = (
    df
    .groupby(DATE_COL)
    .agg({
        DIESEL_COL: "mean",
        E5_COL: "mean",
        BRENT_COL: "mean"
    })
    .reset_index()
    .dropna()
)

# ============================================================
# PLOT: DAILY TIME SERIES (DUAL AXIS)
# ============================================================
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left axis: retail prices
ax1.plot(
    daily[DATE_COL], daily[DIESEL_COL],
    color="tab:blue", linewidth=2.2, label="Diesel (daily avg)"
)

ax1.plot(
    daily[DATE_COL], daily[E5_COL],
    color="tab:orange", linewidth=2.2, label="E5 (daily avg)"
)

ax1.set_xlabel("Date")
ax1.set_ylabel("Retail fuel price (€/L)")

# Right axis: Brent (thin + transparent)
ax2 = ax1.twinx()
ax2.plot(
    daily[DATE_COL], daily[BRENT_COL],
    color="black", linewidth=1.2, alpha=0.5,
    label="Brent crude (daily avg, €/L)"
)

ax2.set_ylabel("Brent crude price (€/L)")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Daily retail fuel prices and Brent crude oil prices (dual axis)")
plt.tight_layout()
plt.show()