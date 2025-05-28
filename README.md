# Climate Documents RAG system

This project uses the (Climate Policy Radar)[https://huggingface.co/ClimatePolicyRadar] documents database to create a RAG system to help automate climate policy analysis. In particular, the tools are aimed at ASCOR analysts at the (Transitions Pathways Initative)[https://www.transitionpathwayinitiative.org/] to help automate their workflow. Using, relevant information from climate policy documents, 6 tools are avialible:

1. Climate Pillar Evaluators which use chain of thought and multilingual LLMs to evaluate the following climate pillars for a given country:
    a) CP1a (Does the country have a framework climate law or equivalent?) evaluator
    b) CP1b (Does the country’s framework climate law specify key accountability elements?) Evaluator

2. Climate Policy Report Maker which creates a structured report on any input climate policy topic

3. Sectoral Transition Report Maker creates a report on a sector in a country answering two key questions requested by ASCOR analysts

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

### 4. Create .env

To have your database and API keys work, you must copy the `dotenv_setup` file and rename it `.env`. This is where you will store your API keys and database url.

**IMPORTANT**: make sure your `.env` file is included in the `.gitignore`, otherwise people can steal your API keys. 

### 5. Database setup:
This RAG system uses a postgres database operated through docker, which should be installed. To set it up, in terminal, run the following commands:

1. docker run -p 5432:5432 --name pgai -e POSTGRES_PASSWORD=password timescale/timescaledb-ha:pg17
2. docker exec -it pgai psql -c "CREATE EXTENSION ai CASCADE;"

Now create a new table for embeddings. 

1. Go to create_table.sql
2. Select the Postgres Server (it's at the bottom for mac, probably the same for windows)
3. Highlight the code and right click to run query

### 6. Dotenv setup:

1. Create a .env file in the root directory
2. Copy and paste the contents of ./.env.sample to the .env file.
3. Replace "your-api-key" with your Nebius API key and "your-home-dir" with your home directory.

### 7. Generate Embeddings

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

