"""
Phase 1 - Step 2: Per-sub-query access check against a flat role-permission map.

No graph, no cross-query memory -- each sub-query resource is checked in
isolation against the requesting user's role. That combinatorial-disclosure
gap is intentional; it's exactly what later phases add on top of this.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml

from src.decompose import SubQuery, DecompositionResult

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


class Verdict(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass
class SubQueryResult:
    sub_query: SubQuery
    decision: Decision
    reason: str


@dataclass
class QueryResult:
    user: str
    role: str
    original_query: str
    sub_results: List[SubQueryResult]
    verdict: Verdict


class AccessCheckError(RuntimeError):
    pass


class PermissionStore:
    """Loads permissions.yaml + users.yaml once and answers resource checks."""

    def __init__(self, permissions_path: Optional[str] = None, users_path: Optional[str] = None):
        permissions_path = permissions_path or str(CONFIG_DIR / "permissions.yaml")
        users_path = users_path or str(CONFIG_DIR / "users.yaml")

        with open(permissions_path) as f:
            self.roles = yaml.safe_load(f)["roles"]
        with open(users_path) as f:
            self.users = yaml.safe_load(f)["users"]

    def role_for_user(self, user: str) -> str:
        if user not in self.users:
            raise AccessCheckError(f"Unknown user: {user!r}")
        return self.users[user]["role"]

    @staticmethod
    def _matches(resource: str, pattern: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            return resource.split(".")[0] == pattern[:-2]
        return resource == pattern

    def check_resource(self, role: str, resource: str) -> SubQueryResult.__class__:
        if role not in self.roles:
            raise AccessCheckError(f"Unknown role: {role!r}")
        role_def = self.roles[role]

        if resource == "UNKNOWN":
            return Decision.DENY, "Resource could not be mapped to the known vocabulary; default-deny."

        for pattern in role_def.get("denied_resources", []) or []:
            if self._matches(resource, pattern):
                return Decision.DENY, f"Explicitly denied for role '{role}' (matched rule '{pattern}')."

        for pattern in role_def.get("allowed_resources", []) or []:
            if self._matches(resource, pattern):
                return Decision.ALLOW, f"Allowed for role '{role}' (matched rule '{pattern}')."

        return Decision.DENY, f"No matching allow-rule for role '{role}'; default-deny (least privilege)."


def _aggregate(decisions: List[Decision]) -> Verdict:
    allowed = sum(1 for d in decisions if d == Decision.ALLOW)
    if allowed == len(decisions):
        return Verdict.FULL
    if allowed == 0:
        return Verdict.REJECTED
    return Verdict.PARTIAL


def check_query(
    user: str,
    original_query: str,
    decomposition: DecompositionResult,
    store: Optional[PermissionStore] = None,
) -> QueryResult:
    store = store or PermissionStore()
    role = store.role_for_user(user)

    sub_results: List[SubQueryResult] = []
    for sq in decomposition.sub_queries:
        decision, reason = store.check_resource(role, sq.resource)
        sub_results.append(SubQueryResult(sub_query=sq, decision=decision, reason=reason))

    verdict = _aggregate([r.decision for r in sub_results])
    return QueryResult(
        user=user,
        role=role,
        original_query=original_query,
        sub_results=sub_results,
        verdict=verdict,
    )
