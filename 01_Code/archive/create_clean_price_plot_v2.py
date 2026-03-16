"""
Create clean price series plot (without crisis highlighting) for thesis chapters
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Paths
DATA_DIR = os.path.join("..", "00_Data", "raw")
OUTPUT_DIR = os.path.join("..", "..", "tex", "figures")

# Read CSV - skip first header row, use second row as column names
df = pd.read_csv(
    os.path.join(DATA_DIR, "Daily.csv"),
    sep=";",
    encoding="utf-8-sig",
    skiprows=1,  # Skip the first header row
    names=["Date", "Open", "High", "Low", "Close", "Volume"],  # Assign column names manually
)

# Parse date column (format: DD.MM.YYYY)
df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors='coerce')

# Remove any rows with invalid dates
df = df.dropna(subset=["Date"])

# Sort by date (oldest to newest)
df = df.sort_values("Date").reset_index(drop=True)

print(f"Loaded {len(df)} rows")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"First few rows:")
print(df.head())

# Create figure
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(df["Date"], df["Close"], linewidth=0.6, color="saddlebrown")
ax.set_title("ICE London Cocoa Futures — Close Price (GBP/tonne)", fontsize=14)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price (GBP)", fontsize=12)

# Format x-axis
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

fig.tight_layout()

# Save
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "price_series.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n✓ Saved clean price series to: {output_path}")
plt.close(fig)
