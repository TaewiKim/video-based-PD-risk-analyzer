import os

from .base import *  # noqa: F401,F403

DEBUG = True
ALLOWED_HOSTS = ["*"]
DATABASES = {
    "default": postgres_database_config(db_name=os.getenv("POSTGRES_DB", "mydb"))  # noqa: F405
}
