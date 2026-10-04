-- =====================================================================
-- Virtual Try-On Platform — Relational schema (PostgreSQL)
-- Images themselves live in object storage; this DB stores only
-- identifiers, metadata, and object-storage KEYS (not the bytes).
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------------------------------------------------------------------
-- Brands integrating the widget (the paying customer of the platform)
-- ---------------------------------------------------------------------
CREATE TABLE brands (
    brand_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,
    api_key_hash    TEXT NOT NULL UNIQUE,
    plan_tier       TEXT NOT NULL DEFAULT 'trial',   -- trial | standard | premium
    storage_bucket  TEXT NOT NULL,                    -- per-brand bucket or shared bucket+prefix
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- Shoppers (the brand's end customers)
-- ---------------------------------------------------------------------
CREATE TABLE customers (
    customer_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand_id        UUID NOT NULL REFERENCES brands(brand_id) ON DELETE CASCADE,
    external_ref    TEXT,                              -- brand's own customer id, if any
    email           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (brand_id, external_ref)
);

-- ---------------------------------------------------------------------
-- Products (garments) catalogued by the brand
-- ---------------------------------------------------------------------
CREATE TABLE products (
    product_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand_id            UUID NOT NULL REFERENCES brands(brand_id) ON DELETE CASCADE,
    sku                 TEXT NOT NULL,
    name                TEXT NOT NULL,
    category            TEXT NOT NULL,                 -- top | bottom | dress | outerwear | full_body
    garment_image_key   TEXT NOT NULL,                 -- object storage key: flat-lay / ghost-mannequin shot
    garment_mask_key    TEXT,                           -- pre-computed segmentation mask, cached
    size_chart          JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (brand_id, sku)
);

CREATE INDEX idx_products_brand ON products(brand_id);

-- ---------------------------------------------------------------------
-- Try-on sessions — one row per try-on request/result
-- ---------------------------------------------------------------------
CREATE TYPE session_status AS ENUM ('pending', 'processing', 'done', 'failed');
CREATE TYPE model_variant  AS ENUM ('pretrained', 'scratch');

CREATE TABLE try_on_sessions (
    session_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    brand_id         UUID NOT NULL REFERENCES brands(brand_id) ON DELETE CASCADE,
    customer_id      UUID REFERENCES customers(customer_id) ON DELETE SET NULL,
    product_id       UUID NOT NULL REFERENCES products(product_id),

    person_image_key TEXT NOT NULL,          -- shopper's uploaded photo, object storage key
    result_image_key TEXT,                    -- generated try-on result, filled in when done

    model_used        model_variant NOT NULL DEFAULT 'pretrained',
    status             session_status NOT NULL DEFAULT 'pending',
    error_message      TEXT,

    latency_ms         INTEGER,               -- inference wall-clock time, for monitoring
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at         TIMESTAMPTZ
);

CREATE INDEX idx_sessions_customer ON try_on_sessions(customer_id);
CREATE INDEX idx_sessions_product  ON try_on_sessions(product_id);
CREATE INDEX idx_sessions_status   ON try_on_sessions(status);

-- ---------------------------------------------------------------------
-- Optional: shopper's saved/favorited try-on results (drives retention
-- policy — anything favorited is exempt from the storage lifecycle rule)
-- ---------------------------------------------------------------------
CREATE TABLE saved_results (
    saved_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id  UUID NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    session_id   UUID NOT NULL REFERENCES try_on_sessions(session_id) ON DELETE CASCADE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (customer_id, session_id)
);
