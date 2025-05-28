# Database setup:
In terminal, run the following commands:

1. docker run -p 5432:5432 --name pgai -e POSTGRES_PASSWORD=password timescale/timescaledb-ha:pg17
2. docker exec -it pgai psql -c "CREATE EXTENSION ai CASCADE;"
3. docker exec -it pgai psql

# Create a new table in the database
Created a new table for embeddings. 

1. Go to create_table.sql
2. Select the Postgres Server (it's at the bottom for mac, probably the same for windows)
3. Highlight the code and right click to run query


# Features

### Climate Pillar Evaluator

This RAG system evaluates the CP1a (Does the country have a framework climate law or equivalent?) and CP1b (Does the country’s framework climate law specify key accountability elements?) climate pillars for ASCOR. It uses the documents used by ASCOR climate policy analysts to evaluate these pillars. 

The tool outputs Yes/No with a justification and asks for human feedback which it can act on.

## Climate Policy Report Maker

This tool creates a report on a climate policy topic which the usesr inputs. The report follows an enforced pre-determined markdown strucutre.

## Sectoral Transition Report Maker

This tool creates short report on sectoral transition in given country and sector, answering key questions relevant to ASCOR analysts. It looks at (1) Have they set a net zero target? and (2) How has the legislation of this country changed in the past 50 years?.