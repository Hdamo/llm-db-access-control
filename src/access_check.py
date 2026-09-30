"""
Phase 1 - Step 2: Per-sub-query access check against a flat role-permission map.

Phase 3a (see docs/phase3/role-hierarchy.md) adds one thing on top of the
original flat model: when a role's own allow/deny rules say nothing about a
resource, PermissionStore.check_resource additionally checks whether ANY
descendant role in config/role_hierarchy.yaml explicitly allows it -- a
senior role inherits from its junior/descendant roles, never the reverse
(see the design doc for why that direction, not the other one, is correct
and safe). A role's own rules are still always checked first and always win;
inheritance only ever fills in silence. This is additive: every role with
no hierarchy file, or with no matching descendants, behaves exactly as
before.

Cross-query memory (session-level combinatorial disclosure) is still not
implemented -- see docs/phase3/relationship-graph.md (Phase 3b, deferred).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml

from src.decompose import SubQuery, DecompositionResult
from src.role_hierarchy import RoleHierarchy

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
    """Loads permissions.yaml + users.yaml once and answers resource checks.

    Also loads config/role_hierarchy.yaml (Phase 3a) by default -- pass
    hierarchy_path="" explicitly to disable hierarchy lookups entirely and
    reproduce pre-Phase-3a flat-only behavior (used by tests that want to
    isolate the flat resolution logic from inheritance).
    """

    def __init__(
        self,
        permissions_path: Optional[str] = None,
        users_path: Optional[str] = None,
        hierarchy_path: Optional[str] = None,
    ):
        permissions_path = permissions_path or str(CONFIG_DIR / "permissions.yaml")
        users_path = users_path or str(CONFIG_DIR / "users.yaml")

        with open(permissions_path) as f:
            self.roles = yaml.safe_load(f)["roles"]
        with open(users_path) as f:
            self.users = yaml.safe_load(f)["users"]

        if hierarchy_path == "":
            self.hierarchy: Optional[RoleHierarchy] = None
        else:
            hierarchy_path = hierarchy_path or str(CONFIG_DIR / "role_hierarchy.yaml")
            self.hierarchy = RoleHierarchy(
                known_roles=set(self.roles.keys()), hierarchy_path=hierarchy_path
            )

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

    def check_resource(
        self, role: str, resource: str, visited: Optional[frozenset] = None
    ) -> SubQueryResult.__class__:
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

        # Own rules are silent -- Phase 3a: check whether any descendant role
        # explicitly allows this resource. Senior roles inherit from junior
        # roles, never the reverse (see docs/phase3/role-hierarchy.md) --
        # this only ever ADDS an allow on top of default-deny, so a role
        # with no hierarchy loaded, or with no children, is unaffected.
        if self.hierarchy is not None:
            visited = (visited or frozenset()) | {role}
            for child in self.hierarchy.children_of(role):
                if child in visited:
                    continue  # defensive cycle guard; unreachable given load-time validation
                child_decision, child_reason = self.check_resource(child, resource, visited)
                if child_decision == Decision.ALLOW:
                    return (
                        Decision.ALLOW,
                        f"Allowed for role '{role}' via inherited rule from descendant "
                        f"role '{child}' ({child_reason})",
                    )

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
