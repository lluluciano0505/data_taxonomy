"""Provider-aware OpenAI-compatible connections; credentials stay in .env."""
import os
from urllib.parse import urlsplit, urlunsplit

PROVIDERS = {
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY", "model": "google/gemini-2.5-flash"},
    "vectorengine": {"label": "Vector Engine", "base_url": "https://api.vectorengine.ai/v1", "key_env": "VECTOR_ENGINE_API_KEY", "model": ""},
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
    "custom": {"label": "Custom (OpenAI compatible)", "base_url": "", "key_env": "CUSTOM_API_KEY", "model": ""},
}


def provider_for(processing):
    provider = processing.get("provider")
    if provider:
        if provider not in PROVIDERS:
            raise ValueError("Unknown API provider")
        return provider
    host = urlsplit(processing.get("base_url") or PROVIDERS["openrouter"]["base_url"]).hostname or ""
    if host in {"api.vectorengine.ai", "api.vectorengine.cn", "api.zhongzhuan.vip"}:
        return "vectorengine"
    if host == "api.deepseek.com":
        return "deepseek"
    return "openrouter" if host == "openrouter.ai" else "custom"


def normalize_base_url(value):
    parts = urlsplit(str(value).strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Base URL must be an HTTP(S) address without credentials, query or fragment")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[:-len("/chat/completions")]
    if not path:
        path = "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def connection_config(processing):
    provider = provider_for(processing)
    defaults = PROVIDERS[provider]
    return {"provider": provider, "base_url": normalize_base_url(processing.get("base_url") or defaults["base_url"]),
            "model": str(processing.get("model", defaults["model"])).strip()}


def get_api_key(provider):
    names = [PROVIDERS[provider]["key_env"]]
    if provider == "openrouter":
        names.append("OPENROUTER_API_KEY_BACKUP")
    for name in names:
        key = os.getenv(name, "").strip()
        if key:
            return key, name
    return "", names[0]


def require_api_key(provider):
    key, name = get_api_key(provider)
    if not key:
        raise ValueError(f"Missing API key: set {name} in .env")
    return key
