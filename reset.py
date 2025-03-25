from sqlalchemy import text
from app.db.database import engine
from app.model.request import Base as RequestBase
from app.model.user import Base as UserBase


def reset_processing_requests_table():
    """Reset the processing_requests table by dropping and recreating it."""
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS processing_requests CASCADE"))
        conn.commit()

    RequestBase.metadata.create_all(bind=engine)
    print("Successfully reset the processing_requests table with the updated schema.")


def reset_users_table():
    """Reset the users table by dropping and recreating it."""
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS users CASCADE"))
        conn.commit()

    UserBase.metadata.create_all(bind=engine)
    print(
        "Successfully reset the users table with the updated schema including access_token field."
    )


def reset_all_tables():
    """Reset all tables in the database."""
    reset_processing_requests_table()
    reset_users_table()
    print("All tables have been reset successfully.")


if __name__ == "__main__":
    reset_all_tables()
