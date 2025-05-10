from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


class AlphaDiagnosticsLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.alpha_log = []

    def log_daily_alpha(self, alpha_df: pd.DataFrame, timestamp: pd.Timestamp):
        if not any(col.startswith("alpha_score") for col in alpha_df.columns):
            return

        alpha_df = alpha_df.copy()
        alpha_df["date"] = timestamp
        self.alpha_log.append(alpha_df)

    def save(self):
        if not self.alpha_log:
            return

        df = pd.concat(self.alpha_log)
        path = self.output_dir / "alpha_log.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        print(f"📊 Saved alpha log to {path}")
        return path

    def analyze(self):
        if not self.alpha_log:
            print("❌ No alpha logs to analyze.")
            return

        df = pd.concat(self.alpha_log)
        component_cols = [col for col in df.columns if col.startswith("alpha_score_")]
        if not component_cols:
            print("❌ No alpha component columns found.")
            return

        df = df[component_cols].dropna()

        # === Correlation Matrix ===
        plt.figure(figsize=(6, 5))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Alpha Component Correlation")
        plt.tight_layout()
        plt.show()

        # === PCA ===
        pca = PCA()
        components = pca.fit_transform(df)
        var_exp = pca.explained_variance_ratio_

        plt.figure()
        plt.bar(range(len(var_exp)), var_exp)
        plt.title("PCA: Variance Explained")
        plt.xlabel("Component")
        plt.ylabel("Variance Ratio")
        plt.tight_layout()
        plt.show()

        print("📈 Explained variance by PCA components:", var_exp.round(4))
