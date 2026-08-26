"""
Builds the closed vocabulary of resource IDs from permissions.yaml.

Giving the decomposition LLM a fixed vocabulary to choose from (rather than
letting it invent free-text resource names) is what makes the access-check
step programmatically consumable -- mirrors the "guided generation" strategy
OrgAccess used to keep synthetic data consistent (Sanyal et al., 2025, Sec 2.2).
"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import List


def load_resource_vocab(permissions_path: str | Path) -> List[str]:
    """Return a sorted, deduplicated list of every concrete (non-wildcard)
    resource ID mentioned anywhere in permissions.yaml, across all roles."""
    with open(permissions_path, "r") as f:
        data = yaml.safe_load(f)

    resources = set()
    for role_def in data.get("roles", {}).values():
        for bucket in ("allowed_resources", "denied_resources"):
            for res in role_def.get(bucket, []) or []:
                if "*" not in res:
                    resources.add(res)

    return sorted(resources)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "config/permissions.yaml"
    for r in load_resource_vocab(path):
        print(r)
