
import duckdb
from .schema import create_tables

def get_connection(db_path="facts.duckdb"):
    """Get a database connection and ensure tables exist"""
    conn = duckdb.connect(db_path)
    create_tables(conn)
    return conn