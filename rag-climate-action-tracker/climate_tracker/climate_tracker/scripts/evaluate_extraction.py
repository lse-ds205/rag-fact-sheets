# scripts/evaluate_extraction.py

import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

GROUND_TRUTH_PATH = "output/ground_truth_template.json"
PREDICTIONS_PATH = "output/policy_targets_output.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def binarize(dict_data, keys, accepted_values={"yes", "soft_yes"}):
    """
    Convert dict to 0/1 list based on 'yes_no' field for each key.
    Treat 'yes' and 'soft_yes' as 1; anything else as 0.
    """
    return [
        int(str(dict_data.get(k, {}).get("yes_no", "")).lower() in accepted_values)
        for k in keys
    ]

def evaluate():
    ground = load_json(GROUND_TRUTH_PATH)
    pred = load_json(PREDICTIONS_PATH)
    
    keys = ["net_zero_target", "sector_targets", "efficiency_target", "electricity_net_zero", "carbon_pricing"]
    
    y_true, y_pred = [], []
    skipped = []
    included_countries = []

    for country in ground:
        if country not in pred:
            skipped.append(country)
            continue
        y_true.append(binarize(ground[country], keys))
        y_pred.append(binarize(pred[country], keys))
        included_countries.append(country)

    if not y_true or not y_pred:
        print("❌ No data available for evaluation. Check your JSON files.")
        return

    if skipped:
        print(f"⚠️ Skipped {len(skipped)} countries with no predictions: {', '.join(skipped)}")

    extra_pred_countries = set(pred.keys()) - set(ground.keys())
    if extra_pred_countries:
        print(f"\nℹ️ {len(extra_pred_countries)} countries in predictions but not in ground truth (ignored during eval):")
        print(", ".join(sorted(extra_pred_countries)))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print(f"\n✅ Overall Evaluation Results:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-score:  {f1:.2f}")

    print("\n📊 Per-label Performance:")
    label_scores = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, key in enumerate(keys):
        print(f"{key:22s} | P: {label_scores[0][i]:.2f} R: {label_scores[1][i]:.2f} F1: {label_scores[2][i]:.2f}")

    print("\n🗺️ Per-country Performance:")
    for i, country in enumerate(included_countries):
        country_true = [y_true[i]]
        country_pred = [y_pred[i]]
        p, r, f1, _ = precision_recall_fscore_support(country_true, country_pred, average='micro', zero_division=0)
        print(f"{country:20s} | P: {p:.2f} R: {r:.2f} F1: {f1:.2f}")

    # --- Export to Markdown ---
    with open("output/evaluation_summary.md", "w", encoding="utf-8") as f:
        f.write("## ✅ Overall Evaluation Results\n")
        f.write(f"- **Precision:** {precision:.2f}\n")
        f.write(f"- **Recall:** {recall:.2f}\n")
        f.write(f"- **F1-score:** {f1:.2f}\n\n")

        f.write("## 📊 Per-label Performance\n")
        f.write("| Target Field           | Precision | Recall | F1-score |\n")
        f.write("|------------------------|-----------|--------|----------|\n")
        for i, key in enumerate(keys):
            f.write(f"| {key:22s} | {label_scores[0][i]:.2f}     | {label_scores[1][i]:.2f}  | {label_scores[2][i]:.2f}    |\n")

        f.write("\n## 🗺️ Per-country Performance\n")
        f.write("| Country          | Precision | Recall | F1-score |\n")
        f.write("|------------------|-----------|--------|----------|\n")
        for i, country in enumerate(included_countries):
            country_true = [y_true[i]]
            country_pred = [y_pred[i]]
            p, r, f1c, _ = precision_recall_fscore_support(country_true, country_pred, average='micro', zero_division=0)
            f.write(f"| {country:16s} | {p:.2f}     | {r:.2f}  | {f1c:.2f}    |\n")

    print("\n📝 Exported evaluation summary to `output/evaluation_summary.md`")

    # --- Plot Visualizations ---
    # Per-label F1
    label_f1_scores = [label_scores[2][i] for i in range(len(keys))]
    plt.figure(figsize=(8, 5))
    plt.bar(keys, label_f1_scores)
    plt.ylabel("F1-score")
    plt.title("📊 Per-label F1-scores")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("output/label_f1_scores.png")
    print("📈 Saved per-label F1-score chart to `output/label_f1_scores.png`")

    # Per-country F1
    country_f1 = []
    for i in range(len(included_countries)):
        country_true = [y_true[i]]
        country_pred = [y_pred[i]]
        _, _, f1c, _ = precision_recall_fscore_support(country_true, country_pred, average='micro', zero_division=0)
        country_f1.append(f1c)

    plt.figure(figsize=(10, 5))
    plt.bar(included_countries, country_f1)
    plt.ylabel("F1-score")
    plt.title("🗺️ Per-country F1-scores")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("output/country_f1_scores.png")
    print("📉 Saved per-country F1-score chart to `output/country_f1_scores.png`")

if __name__ == "__main__":
    evaluate()

