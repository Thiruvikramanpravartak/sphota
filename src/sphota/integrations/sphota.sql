-- Sphota s1 helpers for PostgreSQL (pure SQL, no extensions). Apache-2.0.
--
-- Convention: store codes as TEXT in the canonical display form ('JYZA-7M-X0-A8').
-- A B-tree index with text_pattern_ops makes prefix queries (LIKE 'JYZA-7M%') index scans.
--
--   CREATE TABLE docs (id bigserial PRIMARY KEY, body text, code sphota_code);
--   CREATE INDEX docs_code_prefix ON docs (code text_pattern_ops);
--   SELECT * FROM docs WHERE code LIKE sphota_like('JYZA-7M');           -- a region
--   SELECT sphota_prefix(code, 1) AS topic, count(*) FROM docs GROUP BY 1; -- analytics

CREATE OR REPLACE FUNCTION sphota_is_code(c text) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT c ~ '^[0-9A-HJKMNP-TV-Z]{4}(-[0-9A-HJKMNP-TV-Z]{2}){0,3}$'
$$;

DO $$ BEGIN
  CREATE DOMAIN sphota_code AS text CHECK (sphota_is_code(VALUE));
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- number of levels (1..4)
CREATE OR REPLACE FUNCTION sphota_depth(c text) RETURNS int
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT CASE WHEN c IS NULL OR c = '' THEN 0
              ELSE array_length(string_to_array(c, '-'), 1) END
$$;

-- prefix of depth d ('JYZA-7M-X0-A8', 2) -> 'JYZA-7M'
CREATE OR REPLACE FUNCTION sphota_prefix(c text, d int) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT array_to_string((string_to_array(c, '-'))[1:greatest(1, least(d, sphota_depth(c)))], '-')
$$;

-- level symbol (1-based): ('JYZA-7M-X0-A8', 2) -> '7M'
CREATE OR REPLACE FUNCTION sphota_level(c text, l int) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT (string_to_array(c, '-'))[l]
$$;

-- LIKE pattern matching a prefix and all its extensions (level-aligned: 'JYZA' never matches 'JYZB')
CREATE OR REPLACE FUNCTION sphota_like(p text) RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT p || '%'
$$;

-- true if p is a level-aligned prefix of c
CREATE OR REPLACE FUNCTION sphota_is_prefix(p text, c text) RETURNS boolean
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT c = p OR left(c, length(p) + 1) = p || '-'
$$;

-- number of shared leading levels (0..4): the symbolic distance
CREATE OR REPLACE FUNCTION sphota_common_prefix(a text, b text) RETURNS int
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
  SELECT coalesce(max(i), 0) FROM generate_series(1, least(sphota_depth(a), sphota_depth(b))) AS i
  WHERE sphota_prefix(a, i) = sphota_prefix(b, i)
$$;

-- L1 neighbour table (load with sphota.integrations.postgres.neighbors_copy())
CREATE TABLE IF NOT EXISTS sphota_l1_neighbors (
  l1 char(4) NOT NULL,
  neighbor char(4) NOT NULL,
  rank smallint NOT NULL,
  distance real NOT NULL,
  similarity real NOT NULL,
  PRIMARY KEY (l1, rank)
);

-- hierarchy-aware similarity approximation in SQL:
-- 1.0 for identical codes; shared levels weighted (0.55, 0.25, 0.12, 0.08);
-- different L1: the L1-centroid cosine from the neighbour table (0 if not neighbours).
CREATE OR REPLACE FUNCTION sphota_similarity(a text, b text) RETURNS real
LANGUAGE sql STABLE PARALLEL SAFE AS $$
  WITH s AS (SELECT sphota_common_prefix(a, b) AS k, least(sphota_depth(a), sphota_depth(b)) AS n)
  SELECT CASE
    WHEN s.k = 0 THEN coalesce((SELECT similarity FROM sphota_l1_neighbors
                                WHERE l1 = sphota_prefix(a, 1) AND neighbor = sphota_prefix(b, 1)), 0)
                      * (0.55 / (SELECT sum(w) FROM unnest(ARRAY[0.55, 0.25, 0.12, 0.08]) WITH ORDINALITY AS t(w, i) WHERE i <= s.n))
    ELSE (SELECT sum(w) FROM unnest(ARRAY[0.55, 0.25, 0.12, 0.08]) WITH ORDINALITY AS t(w, i) WHERE i <= s.k)
       / (SELECT sum(w) FROM unnest(ARRAY[0.55, 0.25, 0.12, 0.08]) WITH ORDINALITY AS t(w, i) WHERE i <= s.n)
  END::real
  FROM s
$$;

-- rows in the same L1 region as c or in its k nearest L1 regions
CREATE OR REPLACE FUNCTION sphota_nearby_l1(c text, k int DEFAULT 8) RETURNS SETOF text
LANGUAGE sql STABLE AS $$
  SELECT sphota_prefix(c, 1)
  UNION ALL
  SELECT neighbor::text FROM sphota_l1_neighbors WHERE l1 = sphota_prefix(c, 1) AND rank <= k
$$;
