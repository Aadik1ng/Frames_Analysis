import os

# Set environment variables for the PostgreSQL database
os.environ["DB_NAME"] = "face_rec_1"
os.environ["DB_USER"] = "aaditya"
os.environ["DB_PASSWORD"] = "Hxjrhao22abjbkal1Qjz"
os.environ["DB_HOST"] = "194.195.112.175"
os.environ["DB_PORT"] = "5432"  # Default PostgreSQL port
os.environ["SECRET_KEY"] = "aZ2@xPvX7b@1UwsYy$R!qzL!kW8oRjZ#f3pL0X9#4xTvm6Mk7L7TwY"  # Replace with an actual secret key

# Now check for missing environment variables
REQUIRED_ENV_VARS = [
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
    "DB_HOST",
    "DB_PORT",
    "SECRET_KEY"
]

class Env:
    @staticmethod
    def check_missing_vars():
        missing_vars = [var for var in REQUIRED_ENV_VARS if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @staticmethod
    def get_env(var_name, default=None):
        return os.getenv(var_name, default)

Env.check_missing_vars()

DB_NAME = Env.get_env("DB_NAME")
DB_USER = Env.get_env("DB_USER")
DB_PASSWORD = Env.get_env("DB_PASSWORD")
DB_HOST = Env.get_env("DB_HOST")
DB_PORT = Env.get_env("DB_PORT")
SECRET_KEY = Env.get_env("SECRET_KEY")

# Now you can use these variables for connecting to your PostgreSQL database
print("Database Name:", DB_NAME)
print("Database User:", DB_USER)
