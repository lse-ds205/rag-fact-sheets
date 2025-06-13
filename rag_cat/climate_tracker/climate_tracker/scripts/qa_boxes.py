import json
import re
from pathlib import Path

# Full country list
CAT_COUNTRIES = [
    "Argentina", "Australia", "Bhutan", "Brazil", "Canada", "Chile", "China", "Colombia",
    "Costa Rica", "EU", "Egypt", "Ethiopia", "Gabon", "Germany", "India", "Indonesia",
    "Iran", "Japan", "Kazakhstan", "Kenya", "Mexico", "Morocco",
    "Nepal", "New Zealand", "Nigeria", "Norway", "Peru", "Philippines", "Russia",
    "Saudi Arabia", "Singapore", "South Africa", "South Korea", "Switzerland", "Thailand",
    "The Gambia", "Turkey", "UAE", "USA", "Ukraine", "United Kingdom", "Vietnam"
]

def get_user_questions():
    print("🔧 Enter your questions (1 per tag, using '{country}' where needed):")
    tag_question_map = {}
    prediction_keys = [
        "net_zero_target",
        "sector_targets",
        "efficiency_target",
        "electricity_net_zero",
        "carbon_pricing"
    ]
    for tag in prediction_keys:
        question = input(f"Question for '{tag}' (press Enter to skip): ").strip()
        if question:
            tag_question_map[tag] = question
    return tag_question_map

def get_user_countries():
    user_input = input("Enter countries (comma-separated), or 'all' for all countries: ").strip()
    if user_input.lower() == "all":
        return CAT_COUNTRIES
    return [c.strip() for c in user_input.split(",") if c.strip() in CAT_COUNTRIES]

def format_answer_box(country, key, info, question_template):
    question = question_template.replace("{country}", country)
    yes_no = "✅ Yes" if info.get("yes_no") == "yes" else "❌ No"
    explanation = info.get("explanation", "None")
    years = ", ".join(info.get("year", []) or []) or "N/A"
    sectors = ", ".join(info.get("sectors", []) or []) or "N/A"

    quote_raw = info.get("quote", "").strip()
    if quote_raw.lower().startswith("answer/evidence"):
        quote = "No quote provided."
    else:
        quote = quote_raw.split("Similarity:")[0].strip() if "Similarity:" in quote_raw else quote_raw
    quote_display = f"> {quote}" if quote else "> No quote provided."

    try:
        confidence = round(float(info.get("confidence", 0)) * 100, 2)
    except (ValueError, TypeError):
        confidence = "N/A"

    source = info.get("source_url", "#")

    return f"""
### {country} – {key.replace('_', ' ').title()}

**Question:** {question}  
**Answer:** {yes_no}  
**Explanation:** {explanation}  
**Target Year(s):** {years}  
**Sectors:** {sectors}

**Quote:**  
{quote_display}

**Confidence:** {confidence}%  
**Source:** [Link]({source})
"""

def generate_qa_markdown(input_file, output_file="qa_boxes.md"):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tag_question_map = get_user_questions()
    selected_countries = get_user_countries()

    all_blocks = []

    for country in selected_countries:
        predictions = data.get(country.lower(), {})
        for tag, question in tag_question_map.items():
            info = predictions.get(tag)
            if info:
                box = format_answer_box(country, tag, info, question)
                all_blocks.append(box)

    markdown_output = "\n---\n".join(all_blocks)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_output)

    print(f"\n✅ Q&A boxes saved to '{output_file}'.")

if __name__ == "__main__":
    input_path = "output/policy_targets_output.json"
    output_path = "output/qa_boxes.md"

    if Path(input_path).exists():
        generate_qa_markdown(input_path, output_path)
    else:
        print(f"❌ File not found: {input_path}")
