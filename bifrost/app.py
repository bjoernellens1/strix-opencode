from __future__ import annotations

# pyright: reportMissingImports=false

import asyncio
import logging
import os
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logger = logging.getLogger("bifrost")
logging.basicConfig(
    level=os.getenv("BIFROST_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


COMPOSE_PROJECT_DIR = "/app"
COMPOSE_FILE = "/app/compose/hybrid.yml"
COMPOSE_FILE_MOUNT_PATH = "/app/compose/hybrid.yml"
ENV_FILE = "/app/.env"

PROFILE_STANDARD = "standard"
PROFILE_ON_DEMAND: set[str] = {"heimdall", "loki", "frigga", "odin"}
MANAGED_AGENTS: set[str] = PROFILE_ON_DEMAND | {"valkyrie"}
ALL_KNOWN_AGENTS: set[str] = MANAGED_AGENTS | {"thor"}
ProfileName = Literal["standard", "heimdall", "loki", "frigga", "odin"]

LLAMA_AGENTS: set[str] = set()
VLLM_AGENTS: set[str] = {"thor", "valkyrie", "odin", "heimdall", "loki", "frigga"}


def container_name(agent: str) -> str:
    if agent in LLAMA_AGENTS:
        return f"llama_{agent}"
    return f"vllm_{agent}"


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using default=%d", name, value, default)
        return default


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RuntimeState:
    session_start: datetime = field(default_factory=now_utc)
    active_profile: ProfileName = PROFILE_STANDARD
    last_summon_at: datetime | None = None
    last_release_at: datetime | None = None
    odin_last_release_at: datetime | None = None
    summon_counts: dict[str, int] = field(default_factory=lambda: {"odin": 0, "utility": 0})
    health: dict[str, bool] = field(default_factory=dict)
    health_checked_at: datetime | None = None
    agent_started_at: dict[str, datetime] = field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    port: int
    timestamp: datetime


class StatusResponse(BaseModel):
    active_profile: ProfileName
    running_agents: list[str]
    uptime_seconds: int
    session_start: datetime
    last_summon_at: datetime | None
    last_release_at: datetime | None
    summon_counts: dict[str, int]
    health: dict[str, bool]
    health_checked_at: datetime | None
    gpu_utilization: dict[str, Any]


class ActionResponse(BaseModel):
    ok: bool
    action: str
    agent: str | None = None
    profile: str | None = None
    duration_seconds: float = Field(ge=0)
    detail: str
    timestamp: datetime


class BifrostManager:
    def __init__(self) -> None:
        self.port = env_int("BIFROST_PORT", 8899)
        self.session_budget_odin = env_int("BIFROST_SESSION_BUDGET_ODIN", 3)
        self.session_budget_utility = env_int("BIFROST_SESSION_BUDGET_UTILITY", 10)
        self.odin_cooldown_seconds = env_int("BIFROST_ODIN_COOLDOWN", 600)
        self.odin_ttl_seconds = env_int("BIFROST_ODIN_TTL", 1200)
        self.health_timeout_seconds = env_int("BIFROST_HEALTH_TIMEOUT", 300)
        self.health_poll_interval = env_int("BIFROST_HEALTH_POLL_INTERVAL", 5)

        self.agent_ports = {
            "thor": env_int("THOR_PORT", 8001),
            "valkyrie": env_int("VALKYRIE_PORT", 8002),
            "odin": env_int("ODIN_PORT", 8011),
            "heimdall": env_int("HEIMDALL_PORT", 8012),
            "loki": env_int("LOKI_PORT", 8013),
            "frigga": env_int("FRIGGA_PORT", 8014),
        }

        self.gpu_profile_utilization = {
            "standard": {"thor": "~8GB", "valkyrie": "~17GB", "total": "~25GB"},
            "heimdall": {"thor": "~8GB", "valkyrie": "~17GB", "heimdall": "~6GB", "total": "~31GB"},
            "loki": {"thor": "~8GB", "valkyrie": "~17GB", "loki": "~14GB", "total": "~39GB"},
            "frigga": {"thor": "~8GB", "valkyrie": "~17GB", "frigga": "~28GB", "total": "~53GB"},
            "odin": {"thor": "~8GB", "odin": "~40GB", "total": "~48GB"},
        }

        self.state = RuntimeState(active_profile=PROFILE_STANDARD)
        for agent in ALL_KNOWN_AGENTS:
            self.state.health[agent] = False

        self._lock = asyncio.Lock()

    async def run_compose(self, args: list[str]) -> None:
        cmd = [
            "docker",
            "compose",
            "--project-directory",
            COMPOSE_PROJECT_DIR,
            "--env-file",
            ENV_FILE,
            "-f",
            COMPOSE_FILE,
            *args,
        ]

        logger.info("Running compose command: %s", " ".join(cmd))

        def _run() -> None:
            subprocess.run(cmd, check=True, capture_output=True, text=True)

        try:
            await asyncio.to_thread(_run)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            logger.error("Compose command failed. stdout=%r stderr=%r", stdout, stderr)
            raise HTTPException(status_code=500, detail=f"docker compose failed: {stderr or stdout or exc}") from exc

    async def stop_agent(self, agent: str) -> None:
        if agent not in MANAGED_AGENTS:
            raise HTTPException(status_code=400, detail=f"agent {agent} is not managed by Bifrost")
        await self.run_compose(["stop", container_name(agent)])

    async def start_profile_agent(self, profile: str, agent: str) -> None:
        await self.run_compose(["--profile", profile, "up", "-d", container_name(agent)])

    async def wait_for_health(self, port: int, timeout: int | None = None) -> bool:
        effective_timeout = timeout or self.health_timeout_seconds
        deadline = time.monotonic() + effective_timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                try:
                    resp = await client.get(f"http://127.0.0.1:{port}/v1/models")
                    if resp.status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(self.health_poll_interval)
        return False

    async def poll_health(self) -> None:
        updates: dict[str, bool] = {}
        for agent, port in self.agent_ports.items():
            healthy = await self.wait_for_health(port=port, timeout=2)
            updates[agent] = healthy
        self.state.health.update(updates)
        self.state.health_checked_at = now_utc()

    def running_agents(self) -> list[str]:
        active = ["thor"]
        if self.state.active_profile == PROFILE_STANDARD:
            active.append("valkyrie")
        else:
            active.append(self.state.active_profile)
        return active

    async def stop_non_target_agents(self, target: str | None) -> None:
        if target == "odin":
            to_stop = ["valkyrie", "heimdall", "loki", "frigga"]
        else:
            to_stop = [agent for agent in MANAGED_AGENTS if agent != target and agent != "valkyrie"]
        for agent in to_stop:
            try:
                await self.stop_agent(agent)
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("Non-fatal stop failure for %s: %s", agent, exc)

    def check_policy(self, agent: str) -> None:
        if agent == "odin":
            if self.state.summon_counts["odin"] >= self.session_budget_odin:
                raise HTTPException(status_code=429, detail="Odin session summon budget exceeded")
            if self.state.odin_last_release_at is not None:
                elapsed = (now_utc() - self.state.odin_last_release_at).total_seconds()
                if elapsed < self.odin_cooldown_seconds:
                    remaining = int(self.odin_cooldown_seconds - elapsed)
                    raise HTTPException(status_code=429, detail=f"Odin cooldown active ({remaining}s remaining)")
        else:
            if self.state.summon_counts["utility"] >= self.session_budget_utility:
                raise HTTPException(status_code=429, detail="Utility session summon budget exceeded")

    async def summon(self, agent: str) -> ActionResponse:
        if agent not in PROFILE_ON_DEMAND:
            raise HTTPException(status_code=400, detail=f"Invalid summon agent: {agent}")

        started = time.monotonic()
        async with self._lock:
            self.check_policy(agent)
            await self.stop_non_target_agents(target=agent)
            await self.start_profile_agent(profile=agent, agent=agent)

            healthy = await self.wait_for_health(self.agent_ports[agent], timeout=self.health_timeout_seconds)
            if not healthy:
                raise HTTPException(status_code=504, detail=f"Timed out waiting for {agent} health")

            self.state.active_profile = cast(ProfileName, agent)
            self.state.last_summon_at = now_utc()
            self.state.agent_started_at[agent] = self.state.last_summon_at
            if agent == "odin":
                self.state.summon_counts["odin"] += 1
            else:
                self.state.summon_counts["utility"] += 1

            await self.poll_health()

        duration = time.monotonic() - started
        return ActionResponse(
            ok=True,
            action="summon",
            agent=agent,
            profile=agent,
            duration_seconds=duration,
            detail=f"Agent {agent} is healthy and active",
            timestamp=now_utc(),
        )

    async def release(self, agent: str) -> ActionResponse:
        if agent not in PROFILE_ON_DEMAND:
            raise HTTPException(status_code=400, detail=f"Invalid release agent: {agent}")

        started = time.monotonic()
        async with self._lock:
            await self.stop_agent(agent)
            self.state.last_release_at = now_utc()
            self.state.agent_started_at.pop(agent, None)
            if agent == "odin":
                self.state.odin_last_release_at = self.state.last_release_at
            if self.state.active_profile == agent:
                self.state.active_profile = PROFILE_STANDARD

            await self.poll_health()

        duration = time.monotonic() - started
        return ActionResponse(
            ok=True,
            action="release",
            agent=agent,
            profile=self.state.active_profile,
            duration_seconds=duration,
            detail=f"Agent {agent} stopped",
            timestamp=now_utc(),
        )

    async def restore(self) -> ActionResponse:
        started = time.monotonic()
        async with self._lock:
            await self.stop_non_target_agents(target="valkyrie")
            await self.start_profile_agent(profile=PROFILE_STANDARD, agent="valkyrie")
            healthy = await self.wait_for_health(self.agent_ports["valkyrie"], timeout=self.health_timeout_seconds)
            if not healthy:
                raise HTTPException(status_code=504, detail="Timed out waiting for valkyrie health")

            self.state.active_profile = PROFILE_STANDARD
            self.state.last_release_at = now_utc()
            self.state.agent_started_at.pop("odin", None)
            self.state.agent_started_at.pop("heimdall", None)
            self.state.agent_started_at.pop("loki", None)
            self.state.agent_started_at.pop("frigga", None)

            await self.poll_health()

        duration = time.monotonic() - started
        return ActionResponse(
            ok=True,
            action="restore",
            agent="valkyrie",
            profile=PROFILE_STANDARD,
            duration_seconds=duration,
            detail="Standard profile restored and valkyrie is healthy",
            timestamp=now_utc(),
        )

    async def ttl_watchdog_once(self) -> None:
        async with self._lock:
            odin_started = self.state.agent_started_at.get("odin")
            if odin_started is None:
                return
            elapsed = (now_utc() - odin_started).total_seconds()
            if elapsed < self.odin_ttl_seconds:
                return
            logger.info("Odin TTL exceeded (%ds >= %ds), auto-releasing", int(elapsed), self.odin_ttl_seconds)
            await self.run_compose(["stop", container_name("odin")])
            released_at = now_utc()
            self.state.last_release_at = released_at
            self.state.odin_last_release_at = released_at
            self.state.agent_started_at.pop("odin", None)
            if self.state.active_profile == "odin":
                self.state.active_profile = PROFILE_STANDARD

    async def background_loop(self) -> None:
        while True:
            try:
                await self.ttl_watchdog_once()
                await self.poll_health()
            except Exception:
                logger.exception("background loop iteration failed")
            await asyncio.sleep(30)

    def status(self) -> StatusResponse:
        uptime = int((now_utc() - self.state.session_start).total_seconds())
        profile_gpu = self.gpu_profile_utilization.get(self.state.active_profile, {})
        return StatusResponse(
            active_profile=self.state.active_profile,
            running_agents=self.running_agents(),
            uptime_seconds=uptime,
            session_start=self.state.session_start,
            last_summon_at=self.state.last_summon_at,
            last_release_at=self.state.last_release_at,
            summon_counts=dict(self.state.summon_counts),
            health=dict(self.state.health),
            health_checked_at=self.state.health_checked_at,
            gpu_utilization={
                "estimated_profile_utilization": profile_gpu,
                "compose_file": COMPOSE_FILE_MOUNT_PATH,
            },
        )


manager = BifrostManager()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting Bifrost scheduler on port %d", manager.port)
    task = asyncio.create_task(manager.background_loop())
    await manager.poll_health()
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Bifrost", version="1.0.0", lifespan=lifespan)


@app.get("/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", port=manager.port, timestamp=now_utc())


@app.get("/v1/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return manager.status()


@app.post("/v1/summon/{agent}", response_model=ActionResponse)
async def summon(agent: str) -> ActionResponse:
    return await manager.summon(agent)


@app.post("/v1/restore", response_model=ActionResponse)
async def restore() -> ActionResponse:
    return await manager.restore()


@app.post("/v1/release/{agent}", response_model=ActionResponse)
async def release(agent: str) -> ActionResponse:
    return await manager.release(agent)
