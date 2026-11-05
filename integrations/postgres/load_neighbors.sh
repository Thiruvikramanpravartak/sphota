#!/bin/sh
set -e
psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "\copy sphota_l1_neighbors FROM '/docker-entrypoint-initdb.d/neighbors.csv' CSV"
