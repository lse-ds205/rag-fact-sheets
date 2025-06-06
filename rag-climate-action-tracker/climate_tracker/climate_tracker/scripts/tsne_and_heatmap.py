import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path

# === CONFIG ===
_project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels to reach project root
MD_DIR = _project_root / "policy_targets_pages"
OUTPUT_DIR = _project_root / "output"
CONF_THRESHOLD = 0.70

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_visualizations():
    # === STEP 1: Extract records from markdown files ===
    records = []

    for filename in os.listdir(MD_DIR):
        if not filename.endswith("_policy_targets.md"):
            continue

        country = filename.replace("_policy_targets.md", "").replace("_", " ").title()
        with open(os.path.join(MD_DIR, filename), "r", encoding="utf-8") as f:
            content = f.read()

        blocks = re.split(r"## ", content)[1:]

        for block in blocks:
            label = block.strip().splitlines()[0].strip().lower().replace(" ", "_")
            conf_match = re.search(r"^\s*[-*]\s+\*\*Confidence\*\*:\s+([0-9.]+)", block, re.MULTILINE)

            if conf_match:
                conf_str = conf_match.group(1)
                try:
                    conf = float(conf_str)
                    print(f"[DEBUG] Parsed: {conf} for {label} in {country}")
                except:
                    print(f"[WARNING] Could not parse confidence '{conf_str}' in {country} for {label}")
                    conf = 0.0
                low_conf = conf < CONF_THRESHOLD
                records.append({
                    "country": country,
                    "label": label,
                    "confidence": conf,
                    "low_confidence": low_conf
                })
            else:
                print(f"[WARNING] No confidence match found in {country} for {label}")

    # === STEP 2: Create DataFrame and Matrix ===
    df = pd.DataFrame(records)
    pivot_df = df.pivot(index="country", columns="label", values="confidence").fillna(0)

    # === STEP 3: Heatmap ===
    plt.figure(figsize=(12, max(6, len(pivot_df) * 0.4)))

    # Annotate with 🟥 if confidence < threshold
    annot_df = pivot_df.applymap(lambda v: f"{v:.2f} 🟥" if v < CONF_THRESHOLD else f"{v:.2f}")
    sns.heatmap(pivot_df, annot=annot_df, cmap="YlGnBu", fmt="", linewidths=0.5)

    plt.title("Confidence Scores per Country and Policy Area")
    plt.ylabel("Country")
    plt.xlabel("Policy Question")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_heatmap.png"))
    plt.close()

    # === STEP 4: t-SNE Projection (only if ≥2 countries) ===
    if len(pivot_df) >= 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        embedding = tsne.fit_transform(pivot_df)

        plt.figure(figsize=(10, 6))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=80)
        for i, country in enumerate(pivot_df.index):
            plt.annotate(country, (embedding[i, 0] + 0.5, embedding[i, 1]), fontsize=9)
        plt.title("t-SNE of Country Confidence Vectors")
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "tsne_plot.png"))
        plt.close()
    else:
        print("🔹 Skipped t-SNE: Need at least 2 countries for projection.")

    # === STEP 5: Summary of Low Confidence ===
    low_conf_summary = df[df["low_confidence"]].groupby("country")["label"].count().reset_index()
    low_conf_summary.columns = ["country", "low_confidence_count"]

    print("\n🔍 Countries with low-confidence extractions (< 0.70):")
    print(low_conf_summary.sort_values("low_confidence_count", ascending=False).to_string(index=False))

    # Save CSV summary
    if low_conf_summary.empty:
        print("\n✅ No countries flagged with low confidence (< 0.70) — double-check confidence parsing or threshold.")
    else:
        low_conf_summary.to_csv(os.path.join(OUTPUT_DIR, "low_confidence_summary.csv"), index=False)

if __name__ == "__main__":
    generate_visualizations()
