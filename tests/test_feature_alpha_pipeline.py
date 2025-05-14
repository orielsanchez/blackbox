from blackbox.data.duck import DuckDBDataLoader
from blackbox.models.alpha.mean_reversion import MeanReversionAlpha

# Parameters
DB_PATH = "db/ohlcv.duckdb"
UNIVERSE_CSV = "universe/top_500_by_volume.csv"
START_DATE = "2025-01-01"
END_DATE = "2025-03-01"

# Step 1: Load rolling snapshots
loader = DuckDBDataLoader(
    db_path=DB_PATH,
    universe_csv=UNIVERSE_CSV,
    rolling=True,
    window=21,  # 20-day feature + 1 target day
)

snapshots = loader.load(START_DATE, END_DATE)

# Step 2: Run alpha model on snapshots
alpha = MeanReversionAlpha()

for snapshot in snapshots:
    signals = alpha.generate(snapshot)
    print(f"\n📅 {snapshot['date'].date()} signals:")
    print(signals.sort_values(ascending=False).head())
