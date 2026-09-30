"""Approval is human-only: CLI behaviour and absence of any approving MCP tool."""

from __future__ import annotations

import asyncio
import subprocess

import pytest
import yaml
from mcp.server.mcpserver.exceptions import ToolError

from veradim.cli import approve, diff, reject
from veradim.errors import ApprovalError
from veradim.tools.core import VeradimApp


def _git_log(settings) -> str:
    return subprocess.run(
        ["git", "-C", str(settings.metrics_dir), "log", "--format=%s"], capture_output=True, text=True, check=True
    ).stdout


def test_approve_moves_file_records_approval_and_commits(unapproved_workspace):
    s = unapproved_workspace
    path, _ = approve(s, "revenue")
    assert path == s.approved_dir / "revenue.yaml"
    assert not (s.proposed_dir / "revenue.yaml").exists()
    record = yaml.safe_load(path.read_text())["approval"]
    assert record["approved_by"] == "Test Approver <approver@example.com>"
    assert "approved_at" in record and len(record["proposal_hash"]) == 40
    assert "metrics: approve revenue v1" in _git_log(s)
    status = subprocess.run(["git", "-C", str(s.metrics_dir), "status", "--porcelain"], capture_output=True, text=True)
    assert status.stdout.strip() == ""
    assert VeradimApp(s).query_metric("revenue").rows[0]["value"] == pytest.approx(490.0)


def test_composite_cannot_be_approved_before_components(unapproved_workspace):
    with pytest.raises(ApprovalError, match="not approved"):
        approve(unapproved_workspace, "avg_ticket")


def test_proposal_with_prefilled_approval_is_refused(unapproved_workspace):
    s = unapproved_workspace
    path = s.proposed_dir / "revenue.yaml"
    path.write_text(
        path.read_text() + "\napproval:\n  approved_by: agent\n  approved_at: 2020-01-01T00:00:00\n  proposal_hash: x\n"
    )
    with pytest.raises(ApprovalError, match="must not contain an approval"):
        approve(s, "revenue")


def test_new_version_must_bump_version(workspace):
    s = workspace
    (s.proposed_dir / "revenue.yaml").write_text(
        (s.approved_dir / "revenue.yaml").read_text().split("\n# Recorded by")[0]
    )
    with pytest.raises(ApprovalError, match="must be greater"):
        approve(s, "revenue")


def test_reject_moves_to_rejected_with_reason(unapproved_workspace):
    s = unapproved_workspace
    target = reject(s, "margin", "not a real margin")
    assert target.parent == s.rejected_dir
    assert yaml.safe_load(target.read_text())["rejection"]["reason"] == "not a real margin"
    assert "metrics: reject margin" in _git_log(s)


def test_diff_shows_new_metric(unapproved_workspace):
    assert "new metric" in diff(unapproved_workspace, "revenue")


def test_no_mcp_tool_can_approve():
    from veradim.server import mcp

    tools = {t.name for t in asyncio.run(mcp.list_tools())}
    assert tools >= {"list_metrics", "describe_metric", "query_metric"}
    forbidden = ("approve", "reject", "promote", "publish", "write", "edit", "delete")
    assert not [t for t in tools if any(word in t for word in forbidden)]


def test_errors_reach_the_agent_as_structured_json(monkeypatch, unapproved_workspace):
    import json

    from veradim import server

    monkeypatch.setattr(server, "app", VeradimApp(unapproved_workspace))
    # call_tool raises a ToolError; the protocol layer turns it into an is_error
    # result whose text is this message.
    with pytest.raises(ToolError) as err:
        asyncio.run(server.mcp.call_tool("query_metric", {"metric": "revenue"}))
    payload = json.loads(str(err.value).split(": ", 1)[1])
    assert payload["error"]["code"] == "METRIC_NOT_APPROVED"
    assert "veradim approve revenue" in payload["error"]["hint"]
