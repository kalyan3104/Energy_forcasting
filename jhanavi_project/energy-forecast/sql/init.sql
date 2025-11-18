CREATE TABLE IF NOT EXISTS forecasts (
  id serial PRIMARY KEY,
  ts timestamptz NOT NULL,
  actual numeric,
  predicted numeric,
  model text
);
