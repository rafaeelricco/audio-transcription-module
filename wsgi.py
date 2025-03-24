import os
from dotenv import load_dotenv
from app.factory import create_app

env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env):
    load_dotenv(env)

app = create_app("production")

if __name__ == "__main__":
    app.run(debug=True)
