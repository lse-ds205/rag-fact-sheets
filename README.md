# Database setup:
In terminal:

docker run -p 5432:5432 --name pgai -e POSTGRES_PASSWORD=postgres timescale/timescaledb-ha:pg17
docker exec -it pgai psql -c "CREATE EXTENSION ai CASCADE;"
docker exec -it pgai psql

# Create a new table in the database
Created a new table for embeddings. 

1. Go to create_table.sql
2. Select the Postgres Server (it's at the bottom for mac, probably the same for windows)
3. Highlight the code and right click to run query

