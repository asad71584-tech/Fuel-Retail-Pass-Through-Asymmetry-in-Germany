import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# DATA PATH (YOUR CONFIRMED PATH)
# ============================================================
DATA_PATH = r"D:\python\tankerkoenig_panel 23-25(6)\panel_prices_brent_2023_2025H1.parquet"

# ============================================================
# OUTPUT SETTINGS
# ============================================================
OUT_DIR = Path(r"D:\python\full data set plots\plots_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "02_long_run_pass_through_scatter_regression.png"

# ============================================================
# LOAD DATA
# ============================================================
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(f"Dataset not found at:\n{DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
print("Columns in parquet:", df.columns.tolist())

# ============================================================
# COLUMN NAMES (FROM YOUR DATA)
# ============================================================
DATE_COL   = "date"
DIESEL_COL = "diesel"
E5_COL     = "e5"
BRENT_COL  = "Brent_EUR_per_Litre"

# ============================================================
# DAILY AVERAGES (IMPORTANT: station-level -> daily mean)
# ============================================================
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

daily = (
    df
    .dropna(subset=[DATE_COL, DIESEL_COL, E5_COL, BRENT_COL])
    .groupby(DATE_COL)
    .agg({
        DIESEL_COL: "mean",
        E5_COL: "mean",
        BRENT_COL: "mean",
    })
    .reset_index()
    .dropna()
)

# Make sure regression inputs are non-empty
if daily.empty:
    raise ValueError("Daily dataframe is empty after cleaning. Check column names / missing values.")

x = daily[BRENT_COL].to_numpy()
y_diesel = daily[DIESEL_COL].to_numpy()
y_e5 = daily[E5_COL].to_numpy()

if len(x) == 0 or len(y_diesel) == 0 or len(y_e5) == 0:
    raise ValueError("Regression vectors are empty. Cannot fit regression line.")

# ============================================================
# OLS LINEAR FIT (levels): y = a + b*x
# ============================================================
b_diesel, a_diesel = np.polyfit(x, y_diesel, 1)  # returns slope, intercept
b_e5, a_e5 = np.polyfit(x, y_e5, 1)

# Create fitted lines
x_line = np.linspace(x.min(), x.max(), 200)
y_diesel_fit = a_diesel + b_diesel * x_line
y_e5_fit = a_e5 + b_e5 * x_line

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(12, 6))

# scatter points
plt.scatter(x, y_diesel, s=18, alpha=0.55, label="Diesel (daily avg)")
plt.scatter(x, y_e5, s=18, alpha=0.55, marker="x", label="E5 (daily avg)")

# regression lines
plt.plot(x_line, y_diesel_fit, linewidth=2.5,
         label=f"Diesel fit: y={a_diesel:.2f}+{b_diesel:.2f}×Brent")
plt.plot(x_line, y_e5_fit, linewidth=2.5, linestyle="--",
         label=f"E5 fit: y={a_e5:.2f}+{b_e5:.2f}×Brent")

plt.xlabel("Brent (€/L)")
plt.ylabel("Retail fuel price (€/L)")
plt.title("Long-run pass-through (levels regression): Brent → Diesel & E5")
plt.legend(loc="upper left")
plt.tight_layout()

# Save + show
plt.savefig(OUT_FIG, dpi=300)
print("\nSaved regression figure to:", OUT_FIG)

print("\nRegression equations (levels):")
print(f"Diesel: price_diesel = {a_diesel:.4f} + {b_diesel:.4f} · Brent")
print(f"E5:     price_e5     = {a_e5:.4f} + {b_e5:.4f} · Brent")

plt.show()