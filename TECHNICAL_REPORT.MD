# Tools

#### General Design Choices

**Model Temperature**
Given ASCOR analysts must perform reproducable assessments, a temperature of 0 was used for all models, ensuring given the same context, they will always produce the same assessment.

**General Model Choice**
Models are run using Nebius rather than locally for speed and ease of use. However, all models used are **open source** and cheap to run (building these tools used 3 cents of compute). If desired, the models used can easily be changed by changing the `model` parameter in `scripts\climate_policy_pipelines\shared\llm_models`.

**Markdown Output**
Tool output is in formatted as markdown to make assessments clearer and more readable for analysts. This is enforced using classes for some tools (Sectoral Analysis) or just in prompts (Pillar Pipeline).

**Multilingual**
To ensure that our tools perform well even when climate documents are in non-english languages, for the Pillar Tools, we detect chunk languages, and if they are non-english, a powerful, multilingual LLM is used.

### Pillar Tool (CP 1.a & CP 1.b)

This tool assesses climate legislation of different countires based on documents in the Climate Policy Radar Database, using the ASCOR methodology. 

The pipeline can asses two indicators:
1. CP 1.a: Does the country have a framework climate law or equivalent?
2. CP 1.b: Does the framework climate law specify key accountability elements?

It provides standardized Yes/No assessments with reasoning and mandatory citations for all claims (page number, document name). To make automated assessing many countries, it can be set to only output Yes/No with no justification.

#### Design Choices

**Chain of Thought Interpretation of ASCOR Methodology** 
Based on (ASCOR Methodology)[https://www.transitionpathwayinitiative.org/publications/2024-ascor-framework-methodology-note-version-1-1], each pillar's criteria is broken into 3-4 specific criteria. Each criteria is evaluated separately by a different LLM to ensure it is evaluted rigourously. There is a final evalutor LLM which compares all evaluations together with the ASCOR methodology to determine the final evaluation.

**Language Detection**
- This tool uses the langdetect library to identify document language
- The model used depends on the language of the policy documents: if English content → standard LLM (Meta-Llama-3.1-70B-Instruct), if the content is non-English → use multilingual model (Qwen3-32B) which can use over 110 languages, but is more expensive and slower. This balances being multi-lingual and easy to use and scale. 

**Alternative Large-context Strategy**
A second large context window approach was also employed instead of the chain of thought apporach. A single comprehensive prompt using one copy of Llama-3.3-70B (a much larger but more expensive model) was used

#### Pillar Tool Evaluation


### Custom Report Generation Tool

This tool creates structured reports on any climate legislation topic using documents from the Climate Policy Radar Database.

#### Design Choices

The tool follows the below workflow:

1. Generating a template 
Medium-power LLM creates report structure based on user topic
2. Having the user approve the template (Human in the Loop)
Template validation with fallback to higher-power model if rejected. This balances performance and computational efficency.
3. Splitting the report template into subsections
Low-power model parses template into individual question components
4. Generating hypothetical subsections for retrival
Medium-power LLM generates expected responses to guide document retrieval
4. Generating each Subsection based on retrived chunks
5. Compiling the Document
6. Checking the report
High-power LLM evaluates report on three metrics: completeness, template adherence, and topic relevance. This means failed subsections are automatically rewritten by a higher powered LLM.
7. Having a human check the report
8. Rewriting parts of the report if the user doesn't like it

#### Custom Report Tool Evaluation


### Sectoral Analysis Tool

This tool creates a stuctured report on the climate legislation relevant to a sector in a country based on documents in the Climate Policy Radar Database. This was a feature requested by the ASCOR client. This was integrated with the challenge question in the brief (how has legislation changed over 50 years).

The sector report covers two questions:
1. Has the sector set a net zero target?
2. How has the legislation of this country changed in the past 50 years?

It also provides confidence (`"High", "Medium", "Low"`) for each of these subcomponents of the report and citations for all claims (page number, document name). 

#### Design Choices

**Output Format Standardization**

A typedDict class was created for all structures with strict fields to ensure output is standardized. Some are literals, such as confidence. Langchain `with_structured_output()` is then used to ensure LLMs adhere to classes.

**Chain of Thought & Workflow Design**

- Tasks are split between different LLMs to ensure output is focused and follows the schema and then compiled together by another LLM. 
- Also allows for effiecny gains by giving simpler tasks to less powerful LLMs (eg. synthesising document)

#### Sector Report Tool Evaluation
