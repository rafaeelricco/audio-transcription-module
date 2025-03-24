"""Database migration script for audio-to-text application."""
import os
import click
from flask.cli import FlaskGroup
from flask_migrate import Migrate

from app.factory import create_app, db
from app.model import User, ProcessingRequest

# Create the Flask application instance
app = create_app(os.getenv('FLASK_CONFIG') or 'development')

# Initialize the migration manager
migrate = Migrate(app, db)

# Create CLI group
cli = FlaskGroup(create_app=lambda: app)

@cli.command('init-db')
def init_db():
    """Initialize the database and create all tables."""
    # Import all models here to ensure they're tracked by migrations
    with app.app_context():
        db.create_all()
        click.echo('Database tables created successfully.')

@cli.command('reset-db')
def reset_db():
    """Reset the database (drop all tables and recreate)."""
    with app.app_context():
        db.drop_all()
        db.create_all()
        click.echo('Database has been reset successfully.')

if __name__ == '__main__':
    cli()
