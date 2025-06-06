# Report for Scripts (01-03):

`climate_tracker/climate_tracker/scripts/init-db.py`

This script is designed to initialize the PostgreSQL database schema and ensure support for vector-based search.

Rationales behind the code :

- Ensure the vector extension is enabled in PostgreSQL.
-  Use a raw connection instead of ORM session for early DB setup.Avoids using higher-level ORM abstractions too early (clean separation of setup and ORM logic).


`climate_tracker/climate_tracker/scripts/create-table.py`

Your script is a management utility that sets up the countries table in your PostgreSQL database for the Climate Action Tracker project

`climate_tracker/climate_tracker/scripts/store.py`

This script is production-grade: robust, modular, and informative. It enables seamless ingestion of structured policy data scraped via Scrapy into a normalized relational schema (countries_v2 + country_page_sections_v2) for use in downstream tasks like information retrieval, summarization, and analysis.