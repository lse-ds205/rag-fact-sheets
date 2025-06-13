import json
from pathlib import Path

# Load predictions to get list of countries
PREDICTIONS_PATH = "output/policy_targets_output.json"
GROUND_TRUTH_PATH = "ground_truth_template.json"

with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
    predictions = json.load(f)

# Remove accidental empty key if present
if "" in predictions:
    del predictions[""]

# List of policy labels
labels = [
    "net_zero_target",
    "sector_targets",
    "efficiency_target",
    "electricity_net_zero",
    "carbon_pricing"
]

# Create default template
ground_truth = {
    country: {
        label: {"yes_no": "no"} for label in labels
    }
    for country in sorted(predictions)
}

# Save to JSON
with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=2)

print(f"✅ Saved editable ground truth template to {GROUND_TRUTH_PATH}")
