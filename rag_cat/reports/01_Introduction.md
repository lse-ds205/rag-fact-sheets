# Introduction and Team Organisation 

## Team Organisation & Workflow

We began by meeting together, and reviewing the ASCOR Framework and identifying a wide range of potential policy-relevant questions. We confirmed the relevant questions through a first meeting with Sylvan.

As a group, we narrowed these down to a final set of questions that:
* Were most relevant to the ClimateActionTracker.org data source,
* Would return consistent and non-null results,
* Included a balance of qualitative and quantitative elements, as per guidance.

We then met with Dr. Cardoso Silva during the DS205 office hour, to confirm the project direction and discussed:
* How best to integrate knowledge from previous problem sets,
* What sorts of questions to include: qualitative versus quantitative questions. Dr Cardoso-Silva said that qualitative were the best, but with elements of binary yes/no and percentages/years for more quantitative output. So went with qualitative questions, with the refined NLP policy extraction method allowing us to output the binary yes/no output, and percentages/years/sectors for a more quantitative output. 
* Task automation via tasks.py and best practices for modular project structure.
* Using GitHub. Discussed working in separate branches versus altogether on a single main branch. And we decided as a group, to go with the latter approach. 


## Git & Collaboration Strategy

We decided to work sequentially on a single main branch, coordinating tasks via internal group scheduling and updates.
This linear workflow allowed all four of us to "work in chain" with minimal merge conflicts while maintaining code cohesion.


## Client Interaction & Question Design

* We conducted a group meeting with Sylvan (our client/analyst) where we presented our preliminary research and refined our scope.
* Based on Sylvan’s feedback, we agreed to focus on five key policy questions that an analyst might want answered for each country:
    * Net Zero Target: Does the country have a net zero target, and if so, what year is it set for?
    * Sectoral Strategy: Does the country have a multi-sector climate strategy with quantified sector-specific targets for sectors like Electricity, Transport, Industry, Agriculture/LULUCF, etc.?
    * Energy Efficiency: Does the country have an energy efficiency law or framework, and has it set a target?
    * Net Zero Electricity: Has the country set a 1.5°C-aligned net zero electricity target (e.g., by 2035 for high-income countries)?
    * Carbon Pricing: Is there a carbon pricing mechanism in place (e.g., carbon tax or emissions trading system)?

Sylvan explicitly mentioned that **cross-country analysis was not necessary for this project**. We have intentionally excluded this component—not due to oversight, but to stay aligned with the brief.


## Meeting Cadence & Updates

We held group meetings every 2–3 days to:
* Share progress,
* Troubleshoot challenges collaboratively,
* Reassign tasks as needed based on availability and specialisation.
* Technical Implementation


## Division of Workload

Since we were working sequentially, we divided up the workload fairly, based on everyone's exam schedules. 

**First: Toscane**
* Setting up format, README.md, items.py and main pipeline for the project 
* Scraping CAT website and storing it 
* Quantitativa analysis

**Second: Max**
* Finalising and polishing scraping + cleaning.
* Chunking text and generating embeddings.
* And documenting the above

**Third: Adrien**
* Create retrieval system (embedding search or RAG pipeline)
* Fact-sheets
* Frequent meetings and interactions with Sylvan, addressing what questions we should address, and what to improve on at every stage of the process
* And documenting the above

**Fourth: Sumayya**
* Breaking down complex policy questions into components, and using advanced NLP techniques to extract more structured, explainable answers 
* Running confidence heatmap and tSNE on the above ouput
* Manually writing the ground truths, and computing precision/recall/F-1 metrics
* Question and answer boxes
* And documenting the above

Tasks.py: all four of us 


## Decision Making 


### Choosing the BAAI Model 

A key technical decision in our project was to use the BAAI bge-m3 model for embedding generation, diverging from the SentenceTransformer-based approach used in previous problem set by Toscane:
- The BAAI bge-m3 model is specifically optimized for dense retrieval tasks, making it ideal for our RAG-style question-answering system. Unlike standard SentenceTransformers, it has been instruction-tuned and trained on broad, multilingual corpora, improving its ability to handle varied climate policy language. 
- The model produces high-quality, unit-normalized embeddings that work well with vector search tools like pgvector, ensuring accurate and efficient semantic matching. 
- Its strong zero-shot performance and efficiency in batch embedding also made it a technically robust and practical choice for our team.
-  The main reason we switched to this model is to accomodate for those using Nuvolos.


### Data Handling, Structure, and Embedding Strategy

We ran a scrapy crawl and generated both structured and unstructured output for each country, found in `climate_tracker/data/full_text`. 

During ingestion, we preserved the structured metadata (e.g., `section_title`, `section_url`, `country_doc_id`) in our `country_page_sections_v2` table to ensure traceability and citation. However, for semantic embedding and information retrieval, we deliberately focused only on the unstructured text_content, ignoring tables or numeric scorecards, as unstructured narrative text provides **richer context for qualitative question answering**. We decided to focus on answering qualitative questions, as per Dr Cardoso-Silva's and Sylvan's recommendations. 

In our retrieval script (`climate_tracker/climate_tracker/scripts/information_retrieval.py`), these text blocks were chunked using sentence-based logic and passed through the BAAI bge-m3 model, which was chosen for reasons discussed above. Each chunk was then compared against a set of predefined analytical questions using cosine similarity between embeddings, and the most relevant chunk was surfaced along with its source and similarity score.

This separation of concerns — structured metadata for referencing and unstructured content for embedding — allowed us to maintain a cohesive, modular architecture that is both analyst-friendly and performant in a RAG-style setting.


## Cohesive Structure 

Our project has a cohesive structure, where each script leads neatly on from one another: 

`climate_tracker/climate_tracker/spiders/climate_action_tracker.py`
Scraping the data from all the websites
Chose to go with the unstructured data 

`climate_tracker/climate_tracker/scripts/init_db.py`
Initializes the database with vector support by creating necessary extensions and tables, setting up the foundation for storing and querying vector embeddings.

`climate_tracker/climate_tracker/scripts/create_table.py`
Creates the countries table in the database to store country-specific climate policy data, including document IDs, country names, languages, and text content.

`climate_tracker/climate_tracker/scripts/store.py`
Stores extracted text data from climate policy documents into the database, organizing it by country and maintaining proper metadata.

`climate_tracker/climate_tracker/scripts/generate_embeddings.py`
Generates vector embeddings for document sections using the BAAI/bge-m3 model, enabling semantic search capabilities for policy information retrieval.

`climate_tracker/climate_tracker/scripts/information_retrieval.py`
Processes country data to retrieve and format answers to predefined climate policy questions, using semantic similarity to find relevant information from the database.

`climate_tracker/climate_tracker/scripts/policy_extraction.py`
Extracts structured policy information from retrieved answers, including confidence scores, target years, and sector-specific details, producing both JSON outputs and human-readable Markdown summaries. 

`climate_tracker/climate_tracker/scripts/tsne_and_heatmap.py`
Generates visualizations of policy extraction results, creating heatmaps of confidence scores and t-SNE plots to show relationships between countries' climate policies.

`climate_tracker/climate_tracker/scripts/evaluate_extraction.py`
Evaluates the accuracy of policy extraction by comparing predictions against ground truth data, calculating precision, recall, and F1 scores.

`climate_tracker/climate_tracker/scripts/qa_boxes.py`
Generates formatted Q&A boxes in Markdown format, presenting policy information in a clear, structured way with confidence scores and source citations.




