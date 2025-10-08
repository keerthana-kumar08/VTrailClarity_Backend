from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "HireSense API"
    
    DATABASE_URI: str

    TOKEN_EXPIRE_DAYS: int = 15

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

settings = Settings()