"""
Configurations from .env file. 
"""
from pathlib import Path

from loguru import logger as Logger
from pydantic import BaseSettings


class Configs(BaseSettings):
    BASE_PATH: Path = Path("./")
    DS_ECMWF_WEB: str = "vector_db/ecmwf_web"
    DS_CONFLUENCE_GITHUB: str = "vector_db/deeplake_ds"
    DS_OPENAPI_REF: str = "vector_db/openapi_ref"
    REPLICATE_API_TOKEN: str = ""
    REPLICATE_MODEL: str = "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf"
    HUGGINGFACEHUB_API_TOKEN: str = ""
    LOGGING_FILE: str = "logging.log"
    ECMWF_CHARTS_SERVER: str = "https://charts.ecmwf.int/opencharts-api/v1/"
    PORT: int = 4444
    MAX_TOKENS: int = 512
    DEBUG: bool = False
    BIND_IP: str = "0.0.0.0"

    class Config:
        env_file = "/opt/run/.env"


configs = Configs(_env_file='/opt/run/.env')
Logger.add(configs.BASE_PATH / configs.LOGGING_FILE)
