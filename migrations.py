"""Database migration script for audio-to-text application using Alembic."""

import os
import sys
import click
import alembic.config
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from config import settings

# SQLAlchemy setup
engine = create_engine(settings.DATABASE_URL)
Base = declarative_base()


def run_alembic_command(args):
    """Run an Alembic command programmatically."""
    # Create the Alembic config
    alembic_args = [
        '--raiseerr',
        *args
    ]
    alembic.config.main(argv=alembic_args)


@click.group()
def cli():
    """FastAPI audio-to-text application database management CLI."""
    pass


@cli.command("init")
def init_alembic():
    """Initialize Alembic migrations."""
    run_alembic_command(['init', 'alembic'])
    click.echo("Alembic migrations directory has been created.")


@cli.command("migrate")
@click.option("--message", "-m", help="Migration message")
def create_migration(message):
    """Create a new migration."""
    if message:
        run_alembic_command(['revision', '--autogenerate', '-m', message])
    else:
        run_alembic_command(['revision', '--autogenerate', '-m', 'auto-generated'])
    click.echo("New migration created.")


@cli.command("upgrade")
@click.option("--revision", "-r", default="head", help="Revision to upgrade to")
def upgrade_db(revision):
    """Upgrade database to a specific revision."""
    run_alembic_command(['upgrade', revision])
    click.echo(f"Database upgraded to {revision}.")


@cli.command("downgrade")
@click.option("--revision", "-r", default="-1", help="Revision to downgrade to")
def downgrade_db(revision):
    """Downgrade database to a specific revision."""
    run_alembic_command(['downgrade', revision])
    click.echo(f"Database downgraded to {revision}.")


@cli.command("init-db")
def init_db():
    """Initialize the database and create all tables directly (without migrations)."""
    from app.main import Base, engine
    Base.metadata.create_all(bind=engine)
    click.echo("Database tables created successfully.")


@cli.command("reset-db")
def reset_db():
    """Reset the database (drop all tables and recreate)."""
    from app.main import Base, engine
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    click.echo("Database has been reset successfully.")


if __name__ == "__main__":
    cli()
