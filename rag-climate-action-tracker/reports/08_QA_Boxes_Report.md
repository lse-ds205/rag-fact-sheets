# Q&A Boxes Report

`climate_tracker/climate_tracker/scripts/qa_boxes.py`


## 1. Input Sources 

Each Q&A box is generated from the structured predictions stored in the `output/policy_targets_output.json` file, produced by the `climate_tracker/climate_tracker/scripts/policy_extractor.py` script. These predictions are organized by country and policy area (e.g., net_zero_target, sector_targets, etc.), and are derived `retrieved_country_reports_v2_chunked`.

For each policy area, the following fields are extracted:

* yes_no: Whether the policy commitment or target is mentioned (e.g., "yes" or "no")
* explanation: A brief justification or summary of why the prediction is “yes” or “no” (may be null)
* year: A list of years mentioned in the supporting text (e.g., ["2023", "2050"])
* sectors: If available, the sectors the policy applies to (e.g., ["Lulucf"] or ["Energy"])
* quote: A direct quote from the document excerpted during information retrieval, used as evidence
* confidence: A float between 0 and 1 representing the model’s confidence in its prediction
* source_url: The link to the external source where the information was retrieved (e.g., from Climate Action Tracker)

---

## 2. Question Mapping
Each prediction key (e.g., net_zero_target, sector_targets, carbon_pricing) is mapped to a natural-language question. Users may enter their own phrasing for each policy area, or a predefined question set can be used. The system then presents answers in response to these human-written questions.

For example:

Policy Key: net_zero_target
User Question: "Has the country committed to a net zero emissions target?"
Structured Output: "yes_no": "yes", "year": ["2050"], ...
Displayed Answer: Yes — with year, quote, and confidence.

----

## 3. Q&A Box Structure
Each Q&A box includes the following components:

Country and Policy Area
User-defined Question
Answer: Yes or No
Explanation (if present)
Target Year(s): Any identified target dates
Sectors: If sector-specific targets are mentioned
Quote: A supporting excerpt from the source text
Confidence: A percentage score indicating extraction confidence
Source: A URL to the source document

---

## 4. Example of Prompt Questions
Examples of questions that could be used. They can be either country specific or general. 

#### net_zero_target
Intent: Does the country commit to a net zero target? If so, when?

Suggested questions:

- Has {country} set a national net zero target by 2050?
- What is {country}’s long-term net zero emissions goal?
- Has {country} formally adopted a net zero policy?
- Is {country} committed to achieving net zero emissions by mid-century?
- When does {country} plan to reach net zero greenhouse gas emissions?


#### sector_targets

Intent: Are there targets for specific sectors like transport, electricity, etc.?

Suggested questions:

- What sector-specific climate targets has {country} implemented?
- Has {country} introduced emission targets for any key sectors?
- Are there reduction goals for transport, energy, or industry in {country}?
- Does {country} target specific sectors like LULUCF or electricity in its climate plan?
- Which sectors are included in {country}’s national mitigation strategies?


#### efficiency_target

Intent: Is there an energy efficiency strategy or target?

Suggested questions:

- Does {country} have a policy to improve energy efficiency in buildings or transport?
- Has {country} set energy efficiency targets for industry or households?
- Are there national standards or benchmarks for energy efficiency in {country}?
- What measures has {country} taken to improve energy use efficiency?
- Is improving energy efficiency a formal policy objective for {country}?


#### electricity_net_zero

Intent: Is the electricity sector on a net-zero or clean energy pathway?

Suggested questions:

- Is {country} aiming for net zero emissions in its electricity or power sector?
- What are {country}’s decarbonisation goals for electricity generation?
- Has {country} committed to clean or renewable electricity targets?
- Is {country} phasing out coal or fossil fuels from its power sector?
- How is {country} planning to transition to a net zero electricity system?


#### carbon_pricing

Intent: Is there a carbon tax or emissions trading system?

Suggested questions:

- Has {country} introduced carbon pricing or an emissions trading scheme?
- Is there a carbon tax or cap-and-trade system in place in {country}?
- How does {country} price carbon emissions, if at all?
- Has {country} implemented any form of carbon market mechanism?
- What role does carbon pricing play in {country}’s climate policy?

---

## 5. Example Output

Terminal Output: 

```
🔧 Enter your questions (1 per tag, using '{country}' where needed):
📝 Question for 'net_zero_target' (press Enter to skip): Has {country} set a national net zero target by 2050?
📝 Question for 'sector_targets' (press Enter to skip): What sector-specific climate targets has {country} implemented?
📝 Question for 'efficiency_target' (press Enter to skip): Does {country} have a policy to improve energy efficiency in buildings or transport?
📝 Question for 'electricity_net_zero' (press Enter to skip): Is {country} aiming for net zero emissions in its electricity or power sector?
📝 Question for 'carbon_pricing' (press Enter to skip): Has {country} introduced carbon pricing or an emissions trading scheme? 
🌍 Enter countries (comma-separated), or 'all' for all countries: Mexico, Kenya, Indonesia
```

Markdown Output (showing Mexico below, but the full output can be seen in `output/qa_boxes.md`): 

``` markdown

### Mexico – Net Zero Target

**Question:** Has Mexico set a national net zero target by 2050?  
**Answer:** ✅ Yes  
**Explanation:** Mentions net zero target.  
**Target Year(s):** N/A  
**Sectors:** N/A

**Quote:**  
> No quote provided.

**Confidence:** 83.1%  
**Source:** [Link](https://climateactiontracker.org/countries/mexico/net-zero-targets/)

---

### Mexico – Sector Targets

**Question:** What sector-specific climate targets has Mexico implemented?  
**Answer:** ✅ Yes  
**Explanation:** Mentions sector-specific targets.  
**Target Year(s):** 2020, 2021, 2022  
**Sectors:** Agriculture, Energy, Industry

**Quote:**  
> No quote provided.

**Confidence:** 84.35%  
**Source:** [Link](https://climateactiontracker.org/countries/mexico/assumptions/)

---

### Mexico – Efficiency Target

**Question:** Does Mexico have a policy to improve energy efficiency in buildings or transport?  
**Answer:** ❌ No  
**Explanation:** Mentions energy efficiency themes, but no clear policy or target.  
**Target Year(s):** N/A  
**Sectors:** Electricity

**Quote:**  
> No quote provided.

**Confidence:** 82.42%  
**Source:** [Link](https://climateactiontracker.org/countries/mexico/)

---

### Mexico – Electricity Net Zero

**Question:** Is Mexico aiming for net zero emissions in its electricity or power sector?  
**Answer:** ❌ No  
**Explanation:** None  
**Target Year(s):** 2015, 2018, 2020, 2021, 2024  
**Sectors:** Electricity, Energy, Power

**Quote:**  
> No quote provided.

**Confidence:** 83.12%  
**Source:** [Link](https://climateactiontracker.org/countries/mexico/policies-action/)

---

### Mexico – Carbon Pricing

**Question:** Has Mexico introduced carbon pricing or an emissions trading scheme?  
**Answer:** ✅ Yes  
**Explanation:** Mentions carbon pricing mechanism.  
**Target Year(s):** 2017, 2018  
**Sectors:** N/A

**Quote:**  
> No quote provided.

**Confidence:** 81.73%  
**Source:** [Link](https://climateactiontracker.org/countries/mexico/policies-action/)
```


## 6. Evaluation 

### Better quote extraction and validation 
1. Improve mechanism for quote extraction, as no quote is provided for any of the answers
2. Add a minimum confidence threshold for including quotes
3. Add fallback sources when primary quotes are unavailable
4. Only question 2 outputting the "sectors" section


### User Experience 
1. Adding a preview feature before generating the final markdown


### Performance and Scalability 
1. Batch processing for large numbers of countries 
