
def create_tables(conn):
    """Create the necessary tables if they don't exist"""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id INTEGER PRIMARY KEY,
            entity_name VARCHAR NOT NULL,
            entity_type VARCHAR,
            metadata VARCHAR
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            fact_id INTEGER PRIMARY KEY,
            entity_id INTEGER NOT NULL,
            fact_text VARCHAR NOT NULL,
            fact_vector VARCHAR,
            source_url VARCHAR,
            source_name VARCHAR,
            timestamp_captured TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            timestamp_published TIMESTAMP,
            confidence_score FLOAT DEFAULT 1.0,
            hash_signature VARCHAR
        )
    """)
    
    # Create indexes
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_hash ON facts(hash_signature)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_entity_id ON facts(entity_id)
    """)