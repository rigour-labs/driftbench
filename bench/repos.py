"""The repository list: load `repos.yaml` and reject entries a run can't reproduce."""
from __future__ import annotations

import dataclasses
import re
from datetime import datetime
from pathlib import Path

import yaml

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REQUIRED = ("name", "licence", "default_branch", "pin", "pinned_at")


class RepoListError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class PinnedRepo:
    name: str
    licence: str
    default_branch: str
    pin: str
    pinned_at: datetime
    enabled: bool = True
    licence_note: str = ""

    @property
    def slug(self) -> str:
        """Filesystem-safe name, e.g. `zulip__zulip`."""
        return self.name.replace("/", "__")


def load_repos(path: Path, enabled_only: bool = True) -> list[PinnedRepo]:
    """Parse and validate the repo list; with `enabled_only`, drop disabled repos."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RepoListError(f"cannot read {path}: {exc}") from exc
    entries = data.get("repos")
    if not isinstance(entries, list) or not entries:
        raise RepoListError(f"{path}: expected a non-empty `repos` list")
    repos = [parse_entry(entry, index) for index, entry in enumerate(entries)]
    reject_duplicates(repos)
    return [repo for repo in repos if repo.enabled or not enabled_only]


def parse_entry(entry: object, index: int) -> PinnedRepo:
    if not isinstance(entry, dict):
        raise RepoListError(f"repos[{index}]: expected a mapping")
    missing = [key for key in REQUIRED if not entry.get(key)]
    if missing:
        raise RepoListError(f"repos[{index}]: missing {', '.join(missing)}")
    name, pin = str(entry["name"]), str(entry["pin"])
    if not NAME_RE.match(name):
        raise RepoListError(f"repos[{index}]: name {name!r} is not owner/repo")
    if not SHA_RE.match(pin):
        raise RepoListError(f"{name}: pin must be a full 40-character commit SHA")
    return PinnedRepo(
        name=name,
        licence=str(entry["licence"]),
        default_branch=str(entry["default_branch"]),
        pin=pin,
        pinned_at=parse_time(name, entry["pinned_at"]),
        enabled=bool(entry.get("enabled", True)),
        licence_note=str(entry.get("licence_note", "")),
    )


def parse_time(name: str, value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RepoListError(f"{name}: pinned_at {value!r} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise RepoListError(f"{name}: pinned_at must carry a timezone")
    return parsed


def reject_duplicates(repos: list[PinnedRepo]) -> None:
    seen: set[str] = set()
    for repo in repos:
        if repo.name in seen:
            raise RepoListError(f"{repo.name}: listed twice")
        seen.add(repo.name)
