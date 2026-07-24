"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# override=True so editing .env + Flask reload actually picks up new flags
# (default dotenv leaves stale os.environ values from the first load).
load_dotenv(override=True)


def reload_env() -> None:
    """Re-read .env into os.environ (wins over prior values)."""
    load_dotenv(override=True)


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    ib_host: str
    ib_port: int
    ib_client_id: int
    ib_readonly: bool
    ib_allow_orders: bool
    ib_max_order_notional: float
    agent_max_iterations: int


def get_ib_settings() -> tuple[str, int, int, bool]:
    """IB connection settings only (no OpenAI key required)."""
    reload_env()
    allow = _env_bool("IB_ALLOW_ORDERS", "false")
    # If orders are allowed, force a non-readonly session.
    readonly = _env_bool("IB_READONLY", "true") and not allow
    base_client_id = int(os.getenv("IB_CLIENT_ID", "1"))
    # Flask debug reloader runs parent+child with the same .env clientId. Two
    # sockets on one clientId steal messages and leave later orders PendingSubmit.
    # Offset by PID so each process gets a unique id (still deterministic).
    client_id = base_client_id + (os.getpid() % 200)
    return (
        os.getenv("IB_HOST", "127.0.0.1").strip(),
        int(os.getenv("IB_PORT", "7497")),
        client_id,
        readonly,
    )


def order_mode_status() -> dict[str, object]:
    """Current order-permission flags for diagnostics / tool errors."""
    reload_env()
    allow_raw = _env_bool("IB_ALLOW_ORDERS", "false")
    readonly_raw = _env_bool("IB_READONLY", "true")
    host, port, client_id, readonly_effective = get_ib_settings()
    allowed = allow_raw and not readonly_effective
    return {
        "IB_ALLOW_ORDERS": allow_raw,
        "IB_READONLY": readonly_raw,
        "ib_connect_readonly": readonly_effective,
        "orders_allowed": allowed,
        "host": host,
        "port": port,
        "client_id": client_id,
    }


def orders_allowed() -> bool:
    """True only when explicitly enabled (and connection is not readonly)."""
    return bool(order_mode_status()["orders_allowed"])


def get_max_order_notional() -> float:
    reload_env()
    return float(os.getenv("IB_MAX_ORDER_NOTIONAL", "50000"))


def get_ib_request_timeout() -> float:
    """Seconds before IB request waits raise (qualify, hist, etc.). 0 = wait forever."""
    reload_env()
    return float(os.getenv("IB_REQUEST_TIMEOUT", "10"))


def get_settings() -> Settings:
    reload_env()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    host, port, client_id, readonly = get_ib_settings()
    return Settings(
        openai_api_key=key,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o").strip(),
        ib_host=host,
        ib_port=port,
        ib_client_id=client_id,
        ib_readonly=readonly,
        ib_allow_orders=orders_allowed(),
        ib_max_order_notional=get_max_order_notional(),
        agent_max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "18")),
    )
