"""
Maps OrgAccess dataset ``user_role`` strings (280 distinct variants across
the 70,503-row dataset) to a specific user in config/users.yaml, so each row
can be evaluated under the persona its own role actually implies instead of
one fixed test user for the whole run.

Resolution order:
  1. Exact match against a users.yaml alias (the part of a user id after the
     first '.', e.g. "diana.Chief_Of_Staff" -> alias "Chief_Of_Staff").
  2. Alias match against the parenthetical name lists in each role's
     `description` field in permissions.yaml (e.g. "(IT_Manager,
     Director_IT, Network_Engineer, ...)").
  3. Keyword heuristic fallback -- substring match against the normalized
     role name using an ordered list of domain keywords.
  4. Default to 'intern' (least privilege) if nothing matches.

Every resolution reports which tier it was resolved at (`match_method`) so
downstream reports stay transparent about how confident the role mapping is.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# Ordered fallback keyword buckets. Checked top-to-bottom; first match wins.
# Domain-specific buckets come before generic ones (e.g. "hr" before
# "manager") so a role like "HR_Manager" doesn't fall into the generic
# senior_manager bucket instead of hr_generalist.
_HEURISTICS: list[tuple[list[str], str]] = [
    (["ceo", "cfo", "cto", "coo", "chief", "vp_", "vice_president", "president"], "executive"),
    (["director"], "director"),
    (["intern", "temp_", "new_hire", "junior", "trainee"], "intern"),
    (["vendor", "contractor", "external", "partner"], "vendor"),
    (["audit"], "auditor"),
    (["compliance"], "compliance_officer"),
    (["risk", "privacy", "ethic"], "risk_manager"),
    (["security"], "security_analyst"),
    (["legal", "regulatory", "contract", "policy", "governance"], "legal_counsel"),
    (["hr_", "human_resource", "talent", "recruit", "onboard", "offboard", "employee_relation"], "hr_generalist"),
    (["finance", "financial", "payroll", "budget"], "finance_analyst"),
    (["market"], "marketing_specialist"),
    (["sales", "account_manager", "support"], "sales_representative"),
    (["engineer", "developer", "devops", "ml_", "ai_", "data_scien", "software"], "engineer"),
    (["admin", "it_", "network", "database", "cloud", "system"], "it_manager"),
    (["data", "analyst", "research"], "data_analyst"),
    (["operations", "logistics"], "operations_manager"),
    (["product"], "engineer"),
    (["manager", "lead", "supervisor"], "senior_manager"),
]


class RoleResolver:
    """Loads users.yaml + permissions.yaml once and resolves dataset
    user_role strings to (user_key, role, match_method) triples."""

    def __init__(self, users_path: Optional[str] = None, permissions_path: Optional[str] = None):
        users_path = users_path or str(CONFIG_DIR / "users.yaml")
        permissions_path = permissions_path or str(CONFIG_DIR / "permissions.yaml")

        with open(users_path) as f:
            self.users: dict = yaml.safe_load(f)["users"]
        with open(permissions_path) as f:
            self.roles_def: dict = yaml.safe_load(f)["roles"]

        # normalized alias -> specific user_key, from users.yaml keys
        self._alias_to_user: dict[str, str] = {}
        # role -> representative user_key (first one defined for that role)
        self._role_to_user: dict[str, str] = {}
        for user_key, info in self.users.items():
            role = info["role"]
            self._role_to_user.setdefault(role, user_key)
            alias = user_key.split(".", 1)[1] if "." in user_key else user_key
            norm = _normalize(alias)
            if norm:
                self._alias_to_user.setdefault(norm, user_key)

        # normalized alias -> role, from permissions.yaml description text
        self._alias_to_role: dict[str, str] = {}
        for role, role_def in self.roles_def.items():
            desc = " ".join((role_def.get("description") or "").split())
            m = re.search(r"\(([^)]*)\)", desc)
            if not m:
                continue
            for name in m.group(1).split(","):
                name = name.strip().rstrip(".")
                if not name:
                    continue
                norm = _normalize(name)
                if norm:
                    self._alias_to_role.setdefault(norm, role)

    def resolve(self, dataset_user_role: str) -> tuple[str, str, str]:
        """Return (user_key, role, match_method) for a dataset user_role
        string. match_method is one of: 'exact', 'alias',
        'heuristic:<role>', 'default'."""
        norm = _normalize(dataset_user_role or "")

        if norm in self._alias_to_user:
            user_key = self._alias_to_user[norm]
            return user_key, self.users[user_key]["role"], "exact"

        if norm in self._alias_to_role:
            role = self._alias_to_role[norm]
            user_key = self._role_to_user.get(role)
            if user_key:
                return user_key, role, "alias"

        for keywords, role in _HEURISTICS:
            if any(kw in norm for kw in keywords):
                user_key = self._role_to_user.get(role)
                if user_key:
                    return user_key, role, f"heuristic:{role}"

        role = "intern"
        user_key = self._role_to_user.get(role, next(iter(self.users)))
        return user_key, role, "default"


if __name__ == "__main__":
    # Quick coverage check: how many of the dataset's observed user_role
    # values resolve at each tier?
    import json
    import sys
    from collections import Counter

    schema_path = Path(__file__).resolve().parent.parent / "data" / "orgaccess_schema.json"
    with open(schema_path, encoding="utf-8") as f:
        schema = json.load(f)
    value_counts = schema["fields"]["user_role"]["value_counts"]

    resolver = RoleResolver()
    tier_counts: Counter = Counter()
    examples_by_tier: dict[str, list[str]] = {}
    for role_name, count in value_counts.items():
        _, resolved_role, method = resolver.resolve(role_name)
        tier = method.split(":")[0]
        tier_counts[tier] += count
        examples_by_tier.setdefault(tier, []).append(f"{role_name} -> {resolved_role} ({method})")

    total = sum(value_counts.values())
    print(f"Total rows: {total:,}  |  Unique user_role values: {len(value_counts)}")
    for tier, count in tier_counts.most_common():
        print(f"  {tier:10s}: {count:6,} rows ({count/total*100:.1f}%)")
        for ex in examples_by_tier[tier][:5]:
            print(f"      e.g. {ex}")
    if "--full" in sys.argv:
        for role_name in sorted(value_counts):
            _, resolved_role, method = resolver.resolve(role_name)
            print(f"{role_name!r:40s} -> {resolved_role:22s} ({method})")
