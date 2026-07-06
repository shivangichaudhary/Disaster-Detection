-- scripts/init_db.sql
-- Run automatically by Docker on first startup

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS disaster_alerts (
    id             SERIAL PRIMARY KEY,
    uuid           UUID DEFAULT uuid_generate_v4(),
    timestamp      TIMESTAMPTZ NOT NULL,
    lat            FLOAT NOT NULL,
    lon            FLOAT NOT NULL,
    geom           GEOMETRY(Point, 4326),
    disaster_type  VARCHAR(32)  NOT NULL DEFAULT 'unknown',
    severity       VARCHAR(16)  NOT NULL DEFAULT 'low',
    confidence     FLOAT        NOT NULL DEFAULT 0.0,
    is_disaster    BOOLEAN      NOT NULL DEFAULT FALSE,
    tweet_count    INT          DEFAULT 0,
    geohash        VARCHAR(12),
    raw_json       JSONB,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_geom      ON disaster_alerts USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON disaster_alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_type      ON disaster_alerts(disaster_type);
CREATE INDEX IF NOT EXISTS idx_alerts_severity  ON disaster_alerts(severity);

-- Auto-populate geom from lat/lon
CREATE OR REPLACE FUNCTION set_geom()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geom = ST_SetSRID(ST_MakePoint(NEW.lon, NEW.lat), 4326);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_set_geom
    BEFORE INSERT OR UPDATE ON disaster_alerts
    FOR EACH ROW EXECUTE FUNCTION set_geom();

-- Seed with a few sample alerts for dashboard demo
INSERT INTO disaster_alerts (timestamp, lat, lon, disaster_type, severity, confidence, is_disaster)
VALUES
    (NOW() - INTERVAL '2 hours',  19.076, 72.877, 'flood',      'high',   0.92, TRUE),
    (NOW() - INTERVAL '5 hours',  28.613, 77.209, 'earthquake', 'medium', 0.78, TRUE),
    (NOW() - INTERVAL '8 hours',  13.082, 80.270, 'cyclone',    'high',   0.88, TRUE),
    (NOW() - INTERVAL '12 hours', 22.572, 88.363, 'flood',      'low',    0.61, TRUE),
    (NOW() - INTERVAL '24 hours', 17.385, 78.486, 'wildfire',   'medium', 0.74, TRUE);
