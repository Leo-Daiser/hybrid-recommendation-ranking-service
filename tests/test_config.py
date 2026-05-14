from src.core.config import settings

def test_config_defaults():
    assert settings.app_name == "Hybrid Recommendation API"
    assert "postgresql" in settings.database_url
