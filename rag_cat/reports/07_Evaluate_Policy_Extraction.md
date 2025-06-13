# Evaluating Policy Extraction

From running the following scripts: 
`climate_tracker/climate_tracker/scripts/tsne_and_heatmap.py`
`climate_tracker/climate_tracker/scripts/evaluate_extraction.py`


## Evaluation Methodology

Our climate policy analysis pipeline evaluated country-level policy targets across five key areas:

- `carbon_pricing`
- `efficiency_target`
- `electricity_net_zero`
- `net_zero_target`
- `sector_targets`

Ground truth annotations were manually created based on textual evidence from authoritative sources (e.g., Climate Action Tracker).
Only dates **relevant to actual policy commitments** were included. References or citations to other sources were deliberately excluded, even if detected by the model.

---

## Confidence Heatmap Overview (`output/confidence_heatmap.png`)

Running `climate_tracker/climate_tracker/scripts/tsne_and_heatmap.py`

The heatmap visualizes average model confidence scores across countries and policy areas. Key highlights:

- **High Confidence:** Countries like Germany, Vietnam, and the Philippines consistently show high extraction confidence (≥0.85) across all categories.
- **Low Confidence:** Ukraine scored below 0.70 across all five policy categories (see `low_confidence_summary.csv`), with the lowest per-country total confidence.

- Why did **Ukraine** score so low? 
    - Ukraine received low (< 0.70) confidence scores across all five policy areas — the lowest in the entire dataset.
    - This result aligns with the lack of structured information available in its Climate Action Tracker (CAT) page. The CAT entry for Ukraine explicitly states that its climate policy rating has been suspended due to the war; most recent assessment dates back to March 2022 with only a brief summary available. 
    - No detailed sector breakdowns, quantified targets, or updated long-term strategies are provided.
    - Led to weak embeddings, poor chunk relevance scores
    - Hence a data limitation, not model flaw
    
- **Saudi Arabia** and **Switzerland** had 0.80 confidence scores (on the lower side). Possible reasons: 
    -  Saudi Arabia often presents offset-based or intensity-based targets, which are hard to distinguish from absolute reduction commitments.
    - Switzerland uses market-based instruments like international carbon credits, which adds complexity in deciding whether a sectoral or electricity target has actually been set.

---

## t-SNE Analysis of Country Vectors (`output/tsne_plot.png`)

Running `climate_tracker/climate_tracker/scripts/tsne_and_heatmap.py`

t-SNE helps reveal how consistently the model trusted its answers across countries. Instead of only looking at F1 scores (which compare prediction vs truth), this view tells us:

- Where the model was confident but possibly wrong (overconfident extractions)
- Where it was underconfident but potentially correct
- Whether certain countries share similar policy documentation structure, which can inform future pretraining or prompt tuning

**Key Takeaways:**
- High-performing clusters are valuable case studies for consistent document structure (Philippines, Vietnam, Kenya, Thailand, Morocco). 
- Middle-of-pack countries suggest room for improvement in policy-specific parsing.
    - Upper Middle (EU, Peru, Colombia, Bhutan, Egypt, Ethiopia)
    - Bottom Left (UK, Costa Rica, Australia, USA, India): Countries here had more variation and lower overall confidence, especially on electricity targets and sector specificity. Some may use vague or policy-heavy language (e.g., Australia's 82% renewables confusion).
    - Center Cluster (Moderate Confidence) (Germany, Argentina, Mexico, Turkey, Kazakhstan):  middling performers, generally achieving solid extraction on one or two policy areas but showing uneven confidence elsewhere.
- Ukraine's isolation in the far right corner reinforces what we saw: it had low confidence in all 5 areas, correlating with missing or outdated documentation.


- The USA and Australia are far from EU/Vietnam despite being high-income — indicating that **confidence values did not correlate with economic development**, but more with **clarity and availability of climate documentation**.

---

## Writing the `output/ground_truth_template.json` file 

Chose 9 countries to manually fill out the below fields: 

E.g. for Kenya: 

```json
},
  "kenya": {
    "net_zero_target": {
      "yes_no": "no",
      "target_year": null
    },
    "sector_targets": {
      "yes_no": "yes",
      "targets": [
        {
          "sector": ["transport", "agriculture"],
          "percentage_reduction": null,
          "target_year": "2037",
          "base_year": "2017"
        }
      ]
    },
    "efficiency_target": {
      "yes_no": "yes",
      "percentage_improvement": "2.8%",
      "target_year": null
    },
    "electricity_net_zero": {
      "yes_no": "no",
      "target_year": null
    },
    "carbon_pricing": {
      "yes_no": "no",
      "has_mechanism": false
    }
```

### Why I chose the above fields in my ground truths file:

This section explains why I added in some of the fields in my `output/ground_truth_template.json`, that were not included in running the model (and in the outputted file: `output/policy_targets_output.json`). 

In short, I added in any additional fields that I thought were more semantically related to answering the predefined questions. 

Both the ground truths file and model output file have: "yes", "no" and "soft_yes", for all 5 tags (each tag representing the answer to each of the 5 predefined questions).
- Note: `climate_tracker/climate_tracker/evaluate_extraction.py` has a binarization process that converts categorical responses into binary (0/1) values, where 'yes' and 'soft_yes' are treated as 1; anything else as 0.


#### net_zero_target
**In both files:** only explicitly stated, nationwide net zero commitments 

**What I excluded in ground_truth_template.json:**
- Sector references like “power”, “transport” (irrelevant to this question)
- Lists of sectors mentioned in passing under general goals


#### sector_targets

**In both files:** sectors listed. 

**What I added in ground_truth_template.json:**
- % improvement (makes sense as we're evaluating a multi-sector climate strategy)
- target year for the % improvement 
- base year
Ie. if the country's multi-sector climate strategy is aiming to reduce emissions by 'x'%, by 'y' year, compared to 'z' year's emissions: then, 'y' would be the target year, and 'z' would be the base year. 

**What I excluded in ground_truth_template.json:**
- A list of years  (the model picked up numerous years), but I felt that it was more meaningful to label them as having a "base" year and a "target" year, related to a set % improvement.


#### efficiency_target

**What I excluded in ground_truth_template.json:**
- List of sectors - not related to question 

**What I added in ground_truth_template.json:**
- % improvement 
- Target year for this improvement 

Again, this approach is more meaningful to answering the predifined question. 


#### electricity_net_zero

**What I excluded in ground_truth_template.json:**
- List of sectors - not related to question 

**What I added in ground_truth_template.json:**
- Target year the net zero target 


#### carbon_pricing

**What I excluded in ground_truth_template.json:** 
- List of sectors - not related to question 
- Years mentioned - either country has a carbon pricing mechanism or doesn't; a target / base year isn't required to answer the question.

**What I added in ground_truth_template.json:**
- Whether or not the country has the carbon pricing / tax / credits system in place or not. 



### Chose the following countries for the following reasons: 

#### 1. Argentina  
Argentina has submitted a long-term net zero strategy and includes LULUCF in its sector discussions. However, its targets have been rated as "Poor" by Climate Action Tracker, making it a useful case for analyzing vague or underdefined commitments in national strategies.

#### 2. Germany  
Germany has a legally binding 2045 net zero target and is considered one of the most transparent and advanced countries in terms of climate planning. It offers a strong reference point to evaluate system precision against a clearly articulated policy.

#### 3. India  
As a major developing economy with high emissions and differentiated responsibilities, India presents a critical test case to assess how well the model captures commitments with nuanced phrasing and conditional targets.

#### 4. Kenya  
Kenya's NDCs emphasize low-emission development pathways, but often lack detailed implementation plans. This makes it useful for testing the model's interpretation of aspirational versus firm targets.

#### 5. Australia  
Australia has committed to net zero by 2050, but its climate strategy has faced criticism for lacking enforceable sectoral targets and clarity. This makes it a relevant case to test how the system handles vague or partial commitments, especially when political commitment and legal enforceability diverge.

#### 6. USA  
As a global leader with a 2050 net zero pledge and sectoral plans like the Inflation Reduction Act, the USA offers a complex and multi-layered case to test the model's ability to extract multiple embedded targets.

#### 7. UAE  
As a fossil-fuel-rich country aiming for net zero, the UAE's documents often contain ambiguity and offset-based strategies. This challenges the system's ability to distinguish between absolute emissions reductions and offset reliance.

#### 8. Ethiopia  
Ethiopia expresses strong ambition to remain carbon neutral but lacks specifics on how this will be implemented. This tests the model's handling of vague targets with strong rhetorical language but low specificity.

#### 9. Brazil  
Brazil frequently references carbon neutrality and sectoral shifts, but its targets have been downgraded in ambition. Evaluating Brazil helps assess whether the model accurately flags rollbacks and inconsistencies in climate ambition.


----

## Evaluation Metrics Explained

We use **Precision**, **Recall**, and **F1-score** to assess model performance. We compared the ground truths from the 9 countries (`output/ground_truth_template.json`), with the outputs for the corresponding 9 countries (`output/policy_targets_outputs.json`). 

Running `climate_tracker/climate_tracker/scripts/evaluate_extraction.py`

Leading to the outputs: `output/country_f1_scores.png`, `output/evaluation_summary.md`, `output/label_f1_scores.png`


### Intro to the Metrics 

**Precision:** Of all the model's positive predictions, how many were correct? High precision ensures fewer false positives (e.g., avoids incorrectly saying a country has sector targets when it doesn't)
**Recall:** Of all the actual positives in the ground truth, how many did the model catch? High recall means the model doesn't miss true policy targets, even if they are phrased subtly.
**F1-Score:** Harmonic mean of precision and recall. Balances both metrics — essential for tasks like ours where both false positives and false negatives matter. 


### Overall Metrics

**Global Performance Summary:**
- **Precision:** 0.96
- **Recall:** 0.73
- **F1-score:** 0.80

These values indicate a reasonably strong system. High precision and low recall implies the **model is too strict**, leading to **false negatives** (missing valid cases).


### Per-label Performance

| Target Field           | Precision | Recall | F1-score |
|------------------------|-----------|--------|----------|
| net_zero_target        | 1.00     | 1.00  | 1.00    |
| sector_targets         | 0.89     | 1.00  | 0.94    |
| efficiency_target      | 1.00     | 0.88  | 0.93    |
| electricity_net_zero   | 1.00     | 0.14  | 0.25    |
| carbon_pricing         | 1.00     | 0.40  | 0.57    |

**Key Insights:**
- `net_zero_target`,`sector_targets` and `efficiency_target`, show strong F1-scores (`1.00`, `0.94` and `0.93`, respectively), indicating reliability in structured extraction.
- `electricity_net_zero` and `carbon-pricing` had perfect precision, but low recall and hence low F1 scores. This implies that the model is too strict for what it predicts to be a positive case   
  - The criteria for what defined a net zero country was too strict. 
    - Only return "yes" if both "net-zero" and a direct electricity keyword occur in the same passage.
    - Misses cases where:   
      - Net zero and electricity terms are in separate sentences or paragraphs.
      - It's phrased as a goal or strategy rather than a direct statement (e.g., "fully decarbonized power by 2045").
      - Terms like "carbon-free power", "clean electricity", or "zero-emission grid" are used instead.
  - The criteria for what defined a carbon pricing method was too strict.
    - Misses cases where: 
      - Synonyms or country-specific terms are used (e.g., "cap-and-trade system", "CO2 levy", "market-based emissions control").
      - Indirect mentions occur (e.g., "a mechanism to limit carbon through economic incentives").
      - The concept is implied (e.g., "carbon pricing is being explored", "pilot ETS in development").

- Model works well when the language is close to training examples or standard phrases.
- Model struggles with diverse, implicit, or nuanced phrasing — which is common for carbon_pricing and electricity_net_zero since countries often describe these using policy euphemisms or indirect plans.


### Per-country Performance

| Country          | Precision | Recall | F1-score |
|------------------|-----------|--------|----------|
| argentina        | 1.00     | 0.67  | 0.80    |
| australia        | 1.00     | 1.00  | 1.00    |
| germany          | 1.00     | 0.60  | 0.75    |
| india            | 1.00     | 0.80  | 0.89    |
| kenya            | 1.00     | 1.00  | 1.00    |
| usa              | 1.00     | 0.60  | 0.75    |
| uae              | 1.00     | 0.60  | 0.75    |
| ethiopia         | 0.75     | 0.75  | 0.75    |
| brazil           | 1.00     | 0.67  | 0.80    |


- 8 out of 9 countries have perfect precision (= 1.00), but recall and F1 score less than 1.00. This implies that the model is being too strict about when it predicts a positive case (as shown by the `electricity_net_zero` and `carbon-pricing` tags above).


### Low-Confidence Countries

The CSV `low_confidence_summary.csv` shows:

| Country  | Low Confidence Count (< 0.70) |
|----------|-------------------------------|
| Ukraine  | 5                             |

which proves our explanation above, relating to the lack of information on the Ukraine page on the CAT website. 

---

## Recommendations for Improvement

### Electricity Net Zero
- Add variants like: "clean energy", "carbon-free electricity", "decarbonised power sector", "zero-emissions grid", etc.
- Use spacy's dependency parsing to check co-occurrence of "net zero" and electricity concepts even if split across clauses.

### Carbon Pricing
- Expand regex with more synonyms: r"(carbon|emission[s]?) (tax|pricing|trading|scheme|ETS|market|levy|instrument|mechanism|cap-and-trade)".
- Add fuzzy logic for policy intent.
- Use NLP-based match on concepts like "market-based mechanism to reduce emissions".

### Apply regex filters to exclude citation years
Apply regex filters to exclude citation-based year patterns (`IPCC`, `UNFCCC`, `[12]`, etc.) from year extraction logic
- Citations and footnotes introduced irrelevant years, which misled the model during year extraction or quote selection

#### E.g. for Kenya: 

**Model output: `output/policy_targets_output.json`**

```json
  "kenya": {
    "net_zero_target": {
      "yes_no": "yes",
      "explanation": "Mentions net zero target. (Note: negation present in sentence, which may weaken the claim.)",
      "year": [
        "2020"
      ],
      "sectors": null,
      "quote": "Answer/Evidence (Similarity: 0.8123):\nKenya does not have a net zero target. It is preparing its long-term strategy. ## NDC Updates   KENYA:   - Submitted a stronger target on 28 December 2020.",
      "confidence": 0.8235,
      "source_url": "https://climateactiontracker.org/countries/kenya/targets/"
    },
    "sector_targets": {
      "yes_no": "yes",
      "explanation": "Mentions sector-specific targets. (Note: negation present in sentence, which may weaken the claim.)",
      "year": [
        "2015",
        "2020",
        "2030"
      ],
      "sectors": [
        "Lulucf"
      ],
      "quote": "Answer/Evidence (Similarity: 0.8269):\nAbsolute emissions from the LULUCF sector are projected to decrease from 26 MtCO2e in 2015 to 22 MtCO2e in 2030 and the contribution of this sector to total national emissions is expected to drop from 31% in 2015 to 14% in 2030 under BAU (Ministry of Environment",
      "confidence": 0.8276,
      "source_url": "https://climateactiontracker.org/countries/kenya/policies-action/"
    },
```

**Ground truth: `output/ground_truth_template.json`**

``` json
"kenya": {
    "net_zero_target": {
      "yes_no": "no",
      "target_year": null
    },
    "sector_targets": {
      "yes_no": "yes",
      "targets": [
        {
          "sector": ["transport", "agriculture"],
          "percentage_reduction": null,
          "target_year": "2030",
          "base_year": "2015"
        }
      ]
    },
```

As can be seen under the "sector_targets" tag, the "2020" date is picked up, when that in-fact is a citation. 

See the below the chunk of text from running `climate_tracker/climate_tracker/scripts/information_retrieval`, as can be found in `retrieved_country_reports_v2_chunked/kenya_md`:

```markdown
### Question 2: Does the country have a multi-sector climate strategy that sets quantified sector-specific emission targets or projections for key sectors like Electricity, Transport, Industry, LULUCF/Agriculture, and any other fifth sector with significant emissions?

**Answer/Evidence (Similarity: 0.8269):**
> Absolute emissions from the LULUCF sector are projected to decrease from 26 MtCO2e in 2015 to 22 MtCO2e in 2030 and the contribution of this sector to total national emissions is expected to drop from 31% in 2015 to 14% in 2030 under BAU (Ministry of Environment and Natural Resources, 2017a). In the NDC Sectoral Analysis, the forestry sector’s target is to reduce emissions by 20.1 MtCO2e below BAU by 2030, corresponding to 47% of the overall abatement task (Ministry of Environment and Natural Resources, 2017b). However, Kenya’s updated NDC does not specify a sectoral target for the LULUCF sector (Ministry of Environment and Forestry, 2020).

**Source URL:** [https://climateactiontracker.org/countries/kenya/policies-action/](https://climateactiontracker.org/countries/kenya/policies-action/)
```

