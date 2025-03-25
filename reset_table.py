from app.db.database import engine
from sqlalchemy import text
from app.model.request import Base

# Drop the existing table
with engine.connect() as conn:
    conn.execute(text('DROP TABLE IF EXISTS processing_requests CASCADE'))
    conn.commit()

# Recreate the table with the new schema
Base.metadata.create_all(bind=engine)

print("Successfully reset the processing_requests table with the updated schema.")
