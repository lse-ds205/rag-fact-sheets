# Technical Report on key Design Choices made

This report details the main design choices made when building these tools.

## Table of Contents

- [Embeddings](#embeddings)
- [Retrival](#retrival)
- [Tools](#tools)
  - [General Design Choices](#general-design-choices)
  - [Pillar Tool (CP 1.a & CP 1.b)](#pillar-tool-cp-1a--cp-1b)
    - [Design Choices](#design-choices)
    - [Pillar Tool Evaluation](#pillar-tool-evaluation)
  - [Custom Report Generation Tool](#custom-report-generation-tool)
    - [Design Choices](#design-choices-1)
  - [Sectoral Analysis Tool](#sectoral-analysis-tool)
    - [Design Choices](#design-choices-2)
    - [Sector Report Tool Evaluation](#sector-report-tool-evaluation)

# Embeddings

**Embeddings model choices**

**ClimateBERT**

* For our analysis of the Climate Policy Radar dataset, we selected the **ClimateBERT** model due to its strong focus on climate-specific vocabulary and contexts, making it well-suited to capture the nuanced language prevalent in environmental policy documents. 

**Word2Vec**

* To provide a robust comparison and broaden our semantic representation, we also incorporated **Word2Vec** embeddings. Unlike the pretrained Word2Vec models, we trained our own Word2Vec model on the dataset texts because many domain-specific terms, such as country names, project titles, and specialized terminology, were absent from standard pretrained vocabularies. 

* This dual embedding approach allows us to evaluate how specialized contextual embeddings like ClimateBERT compare against custom-trained static embeddings within the information retrieval tasks. 

**Database Management**

To avoid overwhelming our PostgreSQL database with the abundance of original dataset details, many of which are unnecessary for retrieval, we created a dedicated table to store only the embeddings alongside essential metadata such as **page numbers, source url and document names etc**. This strategy ensures efficient retrieval and reduces data redundancy.

# Retrival

# Tools

### General Design Choices

**Model Temperature**
Given ASCOR analysts must perform reproducable assessments, a temperature of 0 was used for all models, ensuring given the same context, they will always produce the same assessment.

**General Model Choice**
Models are run using Nebius rather than locally for speed and ease of use. However, all models used are **open source** and cheap to run (building these tools used 3 cents of compute). If desired, the models used can easily be changed by changing the `model` parameter in `scripts\climate_policy_pipelines\shared\llm_models`.

**Use of Postgres Database**
The Huggingface Climate Policy Radar dataset is uploaded to a Postgres Database table to speed up country filtering during testing. 

**Markdown Output**
Tool output is in formatted as markdown to make assessments clearer and more readable for analysts. This is enforced using classes for some tools (Sectoral Analysis) or just in prompts (Pillar Pipeline).

**Multilingual**
To ensure that our tools perform well even when climate documents are in non-english languages, for the Pillar Tools, we detect chunk languages, and if they are non-english, a powerful, multilingual LLM is used.

## Pillar Tool (CP 1.a & CP 1.b)

This tool assesses climate legislation of different countires based on documents in the Climate Policy Radar Database, using the ASCOR methodology. 

The pipeline can asses two indicators:
1. CP 1.a: Does the country have a framework climate law or equivalent?
2. CP 1.b: Does the framework climate law specify key accountability elements?

It provides standardized Yes/No assessments with reasoning and mandatory citations for all claims (page number, document name). To make automated assessing many countries, it can be set to only output Yes/No with no justification.

### Design Choices

**Chain of Thought Interpretation of ASCOR Methodology** 
Based on (ASCOR Methodology)[https://www.transitionpathwayinitiative.org/publications/2024-ascor-framework-methodology-note-version-1-1], each pillar's criteria is broken into 3-4 specific criteria. Each criteria is evaluated separately by a different LLM to ensure it is evaluted rigourously. There is a final evalutor LLM which compares all evaluations together with the ASCOR methodology to determine the final evaluation.

**Language Detection**
- This tool uses the langdetect library to identify document language
- The model used depends on the language of the policy documents: if English content → standard LLM (Meta-Llama-3.1-70B-Instruct), if the content is non-English → use multilingual model (Qwen3-32B) which can use over 110 languages, but is more expensive and slower. This balances being multi-lingual and easy to use and scale. 

**Alternative Large-context Strategy**
A second large context window approach was also employed instead of the chain of thought apporach. A single comprehensive prompt using one copy of Llama-3.3-70B (a much larger but more expensive model) was used.

### Pillar Tool Evaluation

We tested the Pillar Tool using Albania as an example. Both the Chain of Thought and Large Context strategies generated an overall CP1a score of YES, and provide similar and reasonable justifications, for example correctly identifying the "Law for Climate Change" in the National Energy and Climate Plan 2019 Draft as evidence that climate change is enshrined in law. However, the Large Context strategy yields more comprehensive and convincing explanations for its assessments, perhaps due to the higher power of the model and less dilution of information through LangChain. Furthermore, the fourth criteria CP1a "exceptional case" was answered differently by the different approaches. The Large Context strategy answers NO to this criteria as there is "no sufficient evidence" that the criteria is met, even if the basic requirement of a broad environmental law and a climate strategy is met. This shows that criteria based on subjective evaluation may not be consistently handled well.

 ```NB07-CP1_Evaluation.ipynb``` compares the CP1 assessments obtained by the our pillar tool against the "grount truth" ASCOR assessments, to measure the accuracy of our RAG system. Unfortunately, since Albania is not included in the existing ASCOR assessments, we cannot check the results with our Evaluation Notebook.  

## Custom Report Generation Tool

This tool creates structured reports on any climate legislation topic using documents from the Climate Policy Radar Database.

The tool follows the below workflow:

1. Generating a template based on the topic using a Medium-power LLM
2. The user approves the template (Human in the Loop) with a fallback to higher-power model if the human rejects it
3. A Low-power model parses template into individual question components
4. Generating hypothetical subsections for retrival: Medium-power LLM generates expected responses to guide document retrieval
4. Generating each Subsection based on retrived chunks
5. Compiling the Document
6. Checking the report
High-power LLM evaluates report on three metrics: completeness, template adherence, and topic relevance. This means failed subsections are automatically rewritten by a higher powered LLM.
7. Having a human check the report
8. Rewriting parts of the report if the user doesn't like it

### Design Choices

**Human-in-the-Loop Workflow**
- Human approval checkpoints are built in at 1. template approval and  2. final report review
- If human rejects template or report, a higher powered model is used again balancing performance and efficency

**Hypothetical Document Retrieval**
- The function `generate_subsection_content()` generates hypothetical responses to subjections which are embedded to retrieve sub-section relevant chunks
- This compensates for the unbounded number of topics covered making knowing which propmt to use to retreive chunks difficult

**Multi-tier Model Strategy**
- Uses three different model tiers (low, medium, high power) assigned to different workflow steps based on how demanding the step is to balance performance and cost-effectiveness
- For example, simple tasks seperating the template into sections use low-power modeles

**Template-based Design**
- This ensures the report is well structured without trading off topics that could be covered
- This acts as way to loosly strucutre unpredictable LLm output, particuarily with human in the loop 

**Automatic Quality Control**
- A model automaticaly rewrites subsections that it does not approve of using a hgiher powered LLM model


## Sectoral Analysis Tool

This tool creates a stuctured report on the climate legislation relevant to a sector in a country based on documents in the Climate Policy Radar Database. This was a feature requested by the ASCOR client. This was integrated with the challenge question in the brief (how has legislation changed over 50 years).

The sector report covers two questions:
1. Has the sector set a net zero target?
2. How has the legislation of this country changed in the past 50 years?

It also provides confidence (`"High", "Medium", "Low"`) for each of these subcomponents of the report and citations for all claims (page number, document name). 

### Design Choices

**Structured Output Enforcement**

- Implements strict TypedDict classes for all output structures (eg. NetZeroAssessment, PeriodAnalysis1970s, LegislativeTimeline)
- Then Langchain's with_structured_output() is used to enforce schema compliance to the TypedDict classes 

**Seperating legislation into sub-periods**
- Divides legislative analysis into distinct time periods (1970s-1980s, 1990s-2000s, 2010s-Present) using dedicated LLMs for each era
- This allows this complex question to be better adressed and better ensures relevant context is included for each period
- Another LLM is used to summarise the overall change based on the sub-period analysis
- A larger models is used for 2010s-Present because of increased policy complexity

**Chain of Thought & Workflow Design**

- Tasks are split between different LLMs to ensure output is focused and follows the schema and then compiled together by another LLM. 
- Also allows for effiecny gains by giving simpler tasks to less powerful LLMs (eg. synthesising document)

**Confidence Rating**
- To make LLMs more honest about the credibility of their analysis, they must give a confidence of `"High", "Medium", "Low"` for each section
- This makes analysis more transparent


