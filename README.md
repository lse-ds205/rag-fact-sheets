# Database setup:
In terminal:

docker run -p 5432:5432 --name pgai -e POSTGRES_PASSWORD=postgres timescale/timescaledb-ha:pg17
docker exec -it pgai psql -c "CREATE EXTENSION ai CASCADE;"
docker exec -it pgai psql
