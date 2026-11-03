import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'ergasia.db'}")
    JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-change-me")
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", str(BASE_DIR.parent / "frontend")))
    CORS_ORIGINS = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:8000").split(",") if o]

    PIPELINE = os.getenv("PIPELINE", "mock")
    MASTERY_BACKEND = os.getenv("MASTERY_BACKEND", "fallback")
    MOCK_STEP_DELAY = float(os.getenv("MOCK_STEP_DELAY", "1.2"))
    BKT_CORRECT_THRESHOLD = float(os.getenv("BKT_CORRECT_THRESHOLD", "0.6"))

    def impl(self, component: str) -> str:
        """mock | real for one pipeline component, e.g. impl('GENERATOR')."""
        return os.getenv(f"{component}_IMPL", self.PIPELINE)


settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
