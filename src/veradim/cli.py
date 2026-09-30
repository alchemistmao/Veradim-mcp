"""Human-only governance CLI.

    veradim list                 approved and proposed metrics
    veradim diff <metric>        proposal vs. currently approved version
    veradim approve <metric>     validate, move to approved/, record who/when, git commit
    veradim reject <metric>      move to rejected/ with a reason, git commit
    veradim validate             validate every approved and proposed file

There is intentionally no MCP tool for any of this: an agent can propose a
metric but can never approve its own proposal.
"""

from __future__ import annotations

import difflib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import typer
import yaml

from veradim.config import Settings
from veradim.engine.connection import ReadOnlyDatabase
from veradim.errors import ApprovalError, DatabaseUnavailable, VeradimError
from veradim.semantic.registry import LoadedMetric, Registry, git_blob_hash, load_yaml, parse_metric, validate_all

cli = typer.Typer(add_completion=False, no_args_is_help=True, help="Veradim metric governance (humans only).")


# ---------------------------------------------------------------------------
# git helpers
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise ApprovalError(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


def _repo_root(path: Path) -> Path:
    root = _git(path, "rev-parse", "--show-toplevel", check=False)
    if not root:
        raise ApprovalError(f"{path} is not inside a git repository; approvals must be recorded in git")
    return Path(root)


def _identity(repo: Path) -> str:
    name = _git(repo, "config", "user.name", check=False)
    email = _git(repo, "config", "user.email", check=False)
    if not name:
        raise ApprovalError("git user.name is not configured; approvals must be attributable to a person")
    return f"{name} <{email}>" if email else name


def _commit(repo: Path, paths: list[Path], message: str) -> str:
    rel = [str(p.resolve().relative_to(repo.resolve())) for p in paths]
    _git(repo, "add", "-A", "--", *rel)
    staged = _git(repo, "diff", "--cached", "--name-only", "--no-renames", "--", *rel).splitlines()
    if not staged:
        raise ApprovalError("nothing to commit")
    _git(repo, "commit", "-q", "-m", message, "--", *staged)
    return _git(repo, "rev-parse", "--short", "HEAD")


# ---------------------------------------------------------------------------
# core operations (importable for tests)
# ---------------------------------------------------------------------------


def _schema(settings: Settings) -> dict[str, set[str]] | None:
    try:
        db = ReadOnlyDatabase(settings.db_path)
        try:
            return db.column_sets()
        finally:
            db.close()
    except DatabaseUnavailable:
        return None


def validate_proposal(settings: Settings, name: str) -> LoadedMetric:
    """Validate a proposal as if it were approved alongside the current catalog."""
    path = settings.proposed_dir / f"{name}.yaml"
    if not path.exists():
        raise ApprovalError(f"no proposal named {name!r} in {settings.proposed_dir}")
    data = load_yaml(path)
    if "approval" in data:
        raise ApprovalError(f"{path.name}: proposals must not contain an approval block")
    metric = parse_metric(data, source=path.name)
    if metric.name != name:
        raise ApprovalError(f"{path.name}: metric name {metric.name!r} must match the file name")

    schema = _schema(settings)
    registry = Registry.load(settings.metrics_dir, schema=schema)
    candidate = LoadedMetric(metric, path, git_blob_hash(path.read_bytes()))
    others = {n: m for n, m in registry.metrics.items() if n != name}
    errors: dict[str, str] = {}
    valid = validate_all({**others, name: candidate}, registry.catalog, schema, errors)
    if name not in valid:
        raise ApprovalError(f"{name} does not validate: {errors.get(name)}")
    # Approving a new version of a simple metric must not break composites built on it.
    broken = [n for n in registry.metrics if n != name and n not in valid]
    if broken:
        raise ApprovalError(
            f"approving {name} would invalidate approved metric(s) {', '.join(broken)}: "
            + "; ".join(errors[b] for b in broken)
        )
    return candidate


def approve(settings: Settings, name: str, *, commit: bool = True) -> tuple[Path, str]:
    candidate = validate_proposal(settings, name)
    proposal_path = settings.proposed_dir / f"{name}.yaml"
    approved_path = settings.approved_dir / f"{name}.yaml"
    repo = _repo_root(settings.metrics_dir)

    previous = None
    if approved_path.exists():
        old = parse_metric(load_yaml(approved_path), source=approved_path.name)
        if candidate.definition.version <= old.version:
            raise ApprovalError(
                f"{name}: proposal version {candidate.definition.version} must be greater than the "
                f"approved version {old.version}"
            )
        previous = git_blob_hash(approved_path.read_bytes())

    approval = {
        "approved_by": _identity(repo),
        "approved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "proposal_hash": candidate.version_hash,
    }
    if previous:
        approval["previous_version"] = previous

    text = proposal_path.read_text(encoding="utf-8").rstrip("\n")
    block = yaml.safe_dump({"approval": approval}, sort_keys=False, default_flow_style=False)
    settings.approved_dir.mkdir(parents=True, exist_ok=True)
    approved_path.write_text(
        f"{text}\n\n# Recorded by `veradim approve`. Do not edit by hand.\n{block}", encoding="utf-8"
    )
    proposal_path.unlink()

    # Make sure what we wrote still parses and validates before committing.
    reg = Registry.load(settings.metrics_dir, schema=_schema(settings))
    if name not in reg.metrics:
        raise ApprovalError(f"internal error: approved file for {name} does not load: {reg.errors.get(name)}")

    if commit:
        _commit(
            repo,
            [proposal_path, approved_path],
            f"metrics: approve {name} v{candidate.definition.version}\n\n"
            f"Approved-by: {approval['approved_by']}\nProposal-hash: {candidate.version_hash}",
        )
    return approved_path, git_blob_hash(approved_path.read_bytes())


def reject(settings: Settings, name: str, reason: str, *, commit: bool = True) -> Path:
    proposal_path = settings.proposed_dir / f"{name}.yaml"
    if not proposal_path.exists():
        raise ApprovalError(f"no proposal named {name!r}")
    repo = _repo_root(settings.metrics_dir)
    record = {
        "rejection": {
            "rejected_by": _identity(repo),
            "rejected_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "reason": reason,
            "proposal_hash": git_blob_hash(proposal_path.read_bytes()),
        }
    }
    settings.rejected_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    target = settings.rejected_dir / f"{name}.{stamp}.yaml"
    text = proposal_path.read_text(encoding="utf-8").rstrip("\n")
    target.write_text(f"{text}\n\n{yaml.safe_dump(record, sort_keys=False)}", encoding="utf-8")
    proposal_path.unlink()
    if commit:
        _commit(repo, [proposal_path, target], f"metrics: reject {name}\n\nReason: {reason}")
    return target


def diff(settings: Settings, name: str) -> str:
    proposal = settings.proposed_dir / f"{name}.yaml"
    approved = settings.approved_dir / f"{name}.yaml"
    if not proposal.exists():
        raise ApprovalError(f"no proposal named {name!r}")
    new = proposal.read_text(encoding="utf-8").splitlines(keepends=True)
    if not approved.exists():
        return f"# {name} is a new metric (no approved version)\n" + "".join(new)
    old = approved.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(difflib.unified_diff(old, new, fromfile=f"approved/{name}.yaml", tofile=f"proposed/{name}.yaml"))


# ---------------------------------------------------------------------------
# typer commands
# ---------------------------------------------------------------------------


def _fail(exc: VeradimError) -> None:
    typer.secho(f"error: {exc.message}", fg=typer.colors.RED, err=True)
    if exc.hint:
        typer.echo(f"hint: {exc.hint}", err=True)
    raise typer.Exit(1)


@cli.command("list")
def list_cmd() -> None:
    """Show approved and proposed metrics."""
    settings = Settings()
    try:
        reg = Registry.load(settings.metrics_dir, schema=_schema(settings))
    except VeradimError as exc:
        _fail(exc)
    typer.secho("Approved", bold=True)
    for m in reg.metrics.values():
        approved_by = m.definition.approval.approved_by if m.definition.approval else "?"
        typer.echo(
            f"  {m.name:<20} {m.definition.kind:<8} v{m.definition.version}  {m.version_hash[:12]}  {approved_by}"
        )
    for name, err in reg.errors.items():
        typer.secho(f"  {name:<20} INVALID: {err}", fg=typer.colors.RED)
    typer.secho("Proposed", bold=True)
    for path in sorted(settings.proposed_dir.glob("*.yaml")):
        typer.echo(f"  {path.stem:<20} {git_blob_hash(path.read_bytes())[:12]}")


@cli.command("diff")
def diff_cmd(metric: str) -> None:
    """Show the proposal against the currently approved version."""
    try:
        typer.echo(diff(Settings(), metric))
        validate_proposal(Settings(), metric)
        typer.secho("\nvalidation: OK", fg=typer.colors.GREEN)
    except VeradimError as exc:
        _fail(exc)


@cli.command("approve")
def approve_cmd(
    metric: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
) -> None:
    """Approve a proposed metric (validates, moves it to approved/, records who/when, commits)."""
    settings = Settings()
    try:
        typer.echo(diff(settings, metric))
        validate_proposal(settings, metric)
        if not yes:
            typer.confirm(f"Approve {metric}?", abort=True)
        path, version = approve(settings, metric)
    except VeradimError as exc:
        _fail(exc)
    typer.secho(f"approved {metric} -> {path} (version {version})", fg=typer.colors.GREEN)


@cli.command("reject")
def reject_cmd(metric: str, reason: str = typer.Option(..., "--reason", "-r")) -> None:
    """Reject a proposed metric with a reason."""
    try:
        path = reject(Settings(), metric, reason)
    except VeradimError as exc:
        _fail(exc)
    typer.secho(f"rejected {metric} -> {path}", fg=typer.colors.YELLOW)


@cli.command("validate")
def validate_cmd() -> None:
    """Validate the catalog, every approved metric and every proposal."""
    settings = Settings()
    ok = True
    try:
        reg = Registry.load(settings.metrics_dir, schema=_schema(settings))
    except VeradimError as exc:
        _fail(exc)
    for name in reg.metrics:
        typer.secho(f"approved  {name}: OK", fg=typer.colors.GREEN)
    for name, err in reg.errors.items():
        ok = False
        typer.secho(f"approved  {name}: {err}", fg=typer.colors.RED)
    for path in sorted(settings.proposed_dir.glob("*.yaml")):
        try:
            validate_proposal(settings, path.stem)
            typer.secho(f"proposed  {path.stem}: OK", fg=typer.colors.GREEN)
        except VeradimError as exc:
            typer.secho(f"proposed  {path.stem}: {exc.message}", fg=typer.colors.YELLOW)
    raise typer.Exit(0 if ok else 1)


def main() -> None:
    cli()
