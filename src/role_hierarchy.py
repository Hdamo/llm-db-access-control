"""
Phase 3a: hierarchical role/permission graph.

Loads config/role_hierarchy.yaml once and answers parent/children queries
for src.access_check.PermissionStore's inheritance walk. See
docs/phase3/role-hierarchy.md for the full design, in particular the
corrected inheritance direction: a SENIOR role inherits from its
JUNIOR/descendant roles when its own rules are silent on a resource --
never the reverse. This module only stores the tree and answers structural
queries (parent_of / children_of); the actual inheritance walk and its
own-rules-first precedence live in access_check.py, unchanged in spirit
from Phase 1's flat resolution -- this file adds no new policy of its own.

Deliberately a single-parent TREE, not a general DAG (see the design doc for
why), so `parent_of` returns at most one role and validation only needs to
rule out cycles, not resolve multi-parent conflicts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_HIERARCHY_PATH = CONFIG_DIR / "role_hierarchy.yaml"


class RoleHierarchyError(RuntimeError):
    pass


class RoleHierarchy:
    """Loads config/role_hierarchy.yaml once; validates at load time (not
    query time) so a bad hierarchy fails fast at startup:
      - every role passed in `known_roles` (normally permissions.yaml's
        role set) appears here exactly once, and vice versa -- no role is
        silently missing from the hierarchy, and no hierarchy entry refers
        to a role that doesn't exist in permissions.yaml.
      - every `parent` value (when not null) is itself a defined role.
      - the graph is acyclic.
    """

    def __init__(
        self,
        known_roles: Optional[Set[str]] = None,
        hierarchy_path: Optional[str] = None,
    ):
        hierarchy_path = hierarchy_path or str(DEFAULT_HIERARCHY_PATH)
        with open(hierarchy_path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}

        roles_doc = doc.get("roles", {}) or {}
        self._parent: Dict[str, Optional[str]] = {}
        for role, entry in roles_doc.items():
            self._parent[role] = (entry or {}).get("parent")

        if known_roles is not None:
            missing = known_roles - self._parent.keys()
            extra = self._parent.keys() - known_roles
            if missing:
                raise RoleHierarchyError(
                    f"{hierarchy_path} is missing {len(missing)} role(s) defined in "
                    f"permissions.yaml: {sorted(missing)}"
                )
            if extra:
                raise RoleHierarchyError(
                    f"{hierarchy_path} defines {len(extra)} role(s) not present in "
                    f"permissions.yaml: {sorted(extra)}"
                )

        for role, parent in self._parent.items():
            if parent is not None and parent not in self._parent:
                raise RoleHierarchyError(
                    f"Role {role!r} in {hierarchy_path} has parent {parent!r}, which "
                    "is not itself a role defined in this file."
                )

        self._children: Dict[str, List[str]] = {role: [] for role in self._parent}
        for role, parent in self._parent.items():
            if parent is not None:
                self._children[parent].append(role)

        self._check_acyclic(hierarchy_path)

    def _check_acyclic(self, hierarchy_path: str) -> None:
        for role in self._parent:
            seen = {role}
            current = self._parent[role]
            while current is not None:
                if current in seen:
                    raise RoleHierarchyError(
                        f"Cycle detected in {hierarchy_path} involving role {role!r}."
                    )
                seen.add(current)
                current = self._parent.get(current)

    def parent_of(self, role: str) -> Optional[str]:
        return self._parent.get(role)

    def children_of(self, role: str) -> List[str]:
        return self._children.get(role, [])
