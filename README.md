# Climate Documents RAG system

This project uses the [Climate Policy Radar](https://huggingface.co/ClimatePolicyRadar) documents database to create a RAG system to help automate climate policy analysis. Using, relevant information from climate policy documents, 6 tools are avialible:

1. **Climate Pillar Evaluators** which use chain of thought and multilingual LLMs to evaluate the following climate pillars for a given country:
    a) CP1a (Does the country have a framework climate law or equivalent?) evaluator
    b) CP1b (Does the country’s framework climate law specify key accountability elements?) Evaluator

2. **Climate Policy Report Maker** which creates a structured report on any input climate policy topic

3. **Sectoral Transition Report Maker** creates a report on a sector in a country answering two key questions requested by ASCOR analysts

## Use Case

These tools are aimed at ASCOR analysts at the [Transitions Pathways Initative](https://www.transitionpathwayinitiative.org/) to help automate their country assessment workflow. The report generating tools will allow an analyst to retrieve relevant information more efficently from a large corpus of climate documents rather than manually searching for information. This could also be of interest to climate researchers outside of ASCOR, in accademia or ESG.

The Pillar evaluating tools can be used to help automate the annual assessment process. An example of how this could be done for all ASCOR countries is in `NB07-CP1_Evaluation`.

Unfortunatley, to use these tools, you must first follow this setup guide:

# Setup

### 1. Clone the repository

To get started, clone this repository to your local machine:

```
git clone https://github.com/Jessie-Fung/group-6-final-project.git
cd group-6-final-project
```

### 2. Create the Virtual Environemnt

It's strongly recommended to use a dedicated virtual environment for this project to avoid conflicts with other Python packages.

```
# Create a virtual environment
python -m venv scraping-env

# Activate the virtual environment
# On Mac/Linux:
source scraping-env/bin/activate
# On Windows:
scraping-env\Scripts\activate
```
### 3. Install Dependencies

With your virtual environment activated, install the required dependencies:
```
pip install -r requirements.txt
```

### 4. Create `.env`

To have your database and API keys work, you must copy the `dotenv_setup` file and rename it `.env`. This is where you will store your API keys and database url.

**WARNING**: make sure your `.env` file is included in the `.gitignore`, otherwise people can steal your API keys. 

### 5. Database setup:
This RAG system uses a postgres database operated through docker, which should be installed. To set it up, in terminal, run the following commands:

1. docker run -p 5432:5432 --name pgai -e POSTGRES_PASSWORD=password timescale/timescaledb-ha:pg17
2. docker exec -it pgai psql -c "CREATE EXTENSION ai CASCADE;"

### 6. Dotenv setup:

1. Create a .env file in the root directory
2. Copy and paste the contents of .env.sample to the .env file.
3. Replace "your-api-key" with your Nebius API key and "your-home-dir" with your home directory.

### 7. Generate Embeddings

#### Loading the Huggingface Dataset and embeddings models

To load the Hugging Face dataset (Climate Policy Radar) and models, you should:
1. Create an account and get an access token
2. Initiate it in the terminal with the following code:

```
huggingface-cli login
```
3. After logging in, you can go to the Climate Policy Radar Page and gain access. The same thing applies to the embedding models.

#### Creating a new table in Postgres SQL

Before generating the embeddings, a new table should be created so the embeddings can be stored into the table. 

To create a new table for embeddings:

1. Go to `create_table.sql`
2. Select the Postgres Server (it's at the bottom for mac, probably the same for windows)
3. Highlight the code and right click to run query

### 8. Set up an LLM API Key

In order to use these tools, you need to have an API key that lets you run inference on LLM models. This tool is built around Nebius, which gives you $1 free inference computer, enough to run our tools 1000s of times. To get a key and setup the system:
1. Go to [https://studio.nebius.com/](https://studio.nebius.com/)
2. Make an account
3. Go to "get an API key"
4. Generate the key
5. Add the key to `.env` under `NEBIUS_API_KEY=<your API key>`
**WARNING**: Make sure your `.env` is in your `.gitignore` so you don't loose lots of money!

You can use a different inference providor because Nebius uses the OpenAI package. You will have to 1. adapt the model names in `scripts/shared/llm_models.py`, 2. make sure the name of the API key matches the key name in `.env` and 3. changes the `base_url` paramter. 

# Features Guide

### Climate Pillar Evaluator

This RAG system evaluates the CP1a (Does the country have a framework climate law or equivalent?) and CP1b (Does the country’s framework climate law specify key accountability elements?) climate pillars for ASCOR. It uses the documents used by ASCOR climate policy analysts to evaluate these pillars. The tool outputs Yes/No with a justification.

To run it, see the demo in `NB04-Pillar_assessment_demo`, under `Notebooks/Tools` or run `run_cp1a_assessment()`.


## Climate Policy Report Maker

This tool creates a report on a climate policy topic which the usesr inputs. The report follows an enforced pre-determined markdown strucutre using typing classes.

To run it, see the demo in `NB05-Report_Genorator_demo` or run `report_workflow.run_workflow("<topic>")`.


## Sectoral Transition Report Maker

This tool creates short report on sectoral transition in given country and sector, answering key questions relevant to ASCOR analysts. It looks at (1) Have they set a net zero target? and (2) How has the legislation of this country changed in the past 50 years?.

To run it, see the demo in `NB06-Sector_transition_demo` or run `generate_ascor_report(country=country, sector=sector)`.

