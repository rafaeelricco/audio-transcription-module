from sqlalchemy import text, inspect
from app.db.database import engine
from app.model.request import Base as RequestBase
from app.model.user import Base as UserBase
import time


def table_exists(table_name):
    """Check if a table exists in the database."""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


def reset_processing_requests_table():
    """Reset the processing_requests table by dropping and recreating it."""
    try:
        # Drop the table if it exists
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS processing_requests CASCADE"))
        
        # Verify the table was dropped
        if table_exists('processing_requests'):
            print("Warning: processing_requests table still exists after drop attempt")
        
        # Small delay to ensure DB operations complete
        time.sleep(0.5)
        
        # Recreate the table with the new schema
        RequestBase.metadata.create_all(bind=engine)
        print("Successfully reset the processing_requests table with the updated schema.")
    except Exception as e:
        print(f"Error resetting processing_requests table: {e}")
        raise


def reset_users_table():
    """Reset the users table by dropping and recreating it."""
    try:
        # Need to drop processing_requests first as it depends on users
        if table_exists('processing_requests'):
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE IF EXISTS processing_requests CASCADE"))
                print("Dropped dependent table processing_requests first")
        
        # Drop the users table
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))
        
        # Verify the table was dropped
        if table_exists('users'):
            print("Warning: users table still exists after drop attempt")
        
        # Small delay to ensure DB operations complete
        time.sleep(0.5)
        
        # Recreate the table with the new schema
        UserBase.metadata.create_all(bind=engine)
        print("Successfully reset the users table with the updated schema including access_token field.")
    except Exception as e:
        print(f"Error resetting users table: {e}")
        raise


def reset_all_tables():
    """Reset all tables in the database in the correct order."""
    try:
        # Reset in correct order - dependent tables first
        reset_processing_requests_table()
        reset_users_table()
        print("All tables have been reset successfully.")
    except Exception as e:
        print(f"Error during table reset: {e}")


if __name__ == "__main__":
    reset_all_tables()
