"""
Unit tests for the access-check logic (Stage 2).

These tests do NOT touch decompose.py or Ollama — they build DecompositionResult
objects directly and feed them into check_query, isolating the permission logic.

Updated to use the OrgAccess-aligned resource vocabulary (department.action)
and the new users / roles defined in config/permissions.yaml and config/users.yaml.

Run:
    python tests/test_access_check.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.access_check import PermissionStore, check_query, Verdict, Decision
from src.decompose import DecompositionResult
from src.role_hierarchy import RoleHierarchy, RoleHierarchyError


def make_decomp(*resources):
    return DecompositionResult.model_validate({
        "sub_queries": [
            {"text": f"sub-query for {r}", "resource": r, "rationale": "test"}
            for r in resources
        ]
    })


# ── Full allow ────────────────────────────────────────────────────────────────

def test_full_allow_executive():
    """CEO (executive role) gets full access to finance.approve + executive.confidential."""
    store = PermissionStore()
    result = check_query(
        "alex.CEO", "q",
        make_decomp("finance.approve", "executive.confidential"),
        store,
    )
    assert result.verdict == Verdict.FULL, f"Expected FULL, got {result.verdict}"
    assert all(sr.decision == Decision.ALLOW for sr in result.sub_results)


def test_full_allow_finance_manager():
    """Finance manager gets full access to finance.view + finance.reports."""
    store = PermissionStore()
    result = check_query(
        "uma.Finance_Manager", "q",
        make_decomp("finance.view", "finance.reports", "finance.export"),
        store,
    )
    assert result.verdict == Verdict.FULL, f"Expected FULL, got {result.verdict}"
    assert all(sr.decision == Decision.ALLOW for sr in result.sub_results)


def test_full_allow_security_analyst():
    """Security analyst gets full access to security.view + security.audit."""
    store = PermissionStore()
    result = check_query(
        "vera.Security_Analyst", "q",
        make_decomp("security.view", "security.audit"),
        store,
    )
    assert result.verdict == Verdict.FULL, f"Expected FULL, got {result.verdict}"


# ── Full reject ───────────────────────────────────────────────────────────────

def test_full_reject_intern_denied():
    """Intern is denied finance.edit + it.deploy (both explicitly denied)."""
    store = PermissionStore()
    result = check_query(
        "tim.IT_Intern", "q",
        make_decomp("finance.edit", "it.deploy"),
        store,
    )
    assert result.verdict == Verdict.REJECTED, f"Expected REJECTED, got {result.verdict}"
    assert all(sr.decision == Decision.DENY for sr in result.sub_results)


def test_full_reject_vendor():
    """Vendor is denied finance.view + hr.edit (both blocked by deny rules)."""
    store = PermissionStore()
    result = check_query(
        "liam.Vendor_Support", "q",
        make_decomp("finance.view", "hr.edit"),
        store,
    )
    assert result.verdict == Verdict.REJECTED, f"Expected REJECTED, got {result.verdict}"
    assert all(sr.decision == Decision.DENY for sr in result.sub_results)


# ── Partial ───────────────────────────────────────────────────────────────────

def test_partial_hr_generalist():
    """HR generalist can view HR data but cannot approve finance actions."""
    store = PermissionStore()
    result = check_query(
        "paula.HR_Generalist", "q",
        make_decomp("hr.view", "finance.approve"),
        store,
    )
    assert result.verdict == Verdict.PARTIAL, f"Expected PARTIAL, got {result.verdict}"
    decisions = {sr.sub_query.resource: sr.decision for sr in result.sub_results}
    assert decisions["hr.view"] == Decision.ALLOW
    assert decisions["finance.approve"] == Decision.DENY


def test_partial_sales_rep():
    """Sales rep can view sales reports but cannot deploy engineering code."""
    store = PermissionStore()
    result = check_query(
        "max.Sales_Representative", "q",
        make_decomp("sales.reports", "engineering.deploy"),
        store,
    )
    assert result.verdict == Verdict.PARTIAL, f"Expected PARTIAL, got {result.verdict}"
    decisions = {sr.sub_query.resource: sr.decision for sr in result.sub_results}
    assert decisions["sales.reports"] == Decision.ALLOW
    assert decisions["engineering.deploy"] == Decision.DENY


# ── Explicit deny overrides wildcard allow ────────────────────────────────────

def test_explicit_deny_overrides_wildcard_allow():
    """
    director has broad allowed_resources but explicitly denied executive.confidential
    and executive.approve — explicit deny must win even though no wildcard covers it.
    """
    store = PermissionStore()
    result = check_query(
        "sarah.Director_Finance", "q",
        make_decomp("executive.confidential"),
        store,
    )
    assert result.verdict == Verdict.REJECTED, f"Expected REJECTED, got {result.verdict}"
    assert result.sub_results[0].decision == Decision.DENY


def test_explicit_deny_overrides_wildcard_allow_intern():
    """
    Intern has intern.finance.view allowed but intern.finance.edit is explicitly denied.
    """
    store = PermissionStore()
    # intern has finance.view allowed, finance.edit denied
    result_allow = check_query(
        "rob.Finance_Intern", "q", make_decomp("finance.view"), store
    )
    result_deny = check_query(
        "rob.Finance_Intern", "q", make_decomp("finance.edit"), store
    )
    assert result_allow.verdict == Verdict.FULL
    assert result_deny.verdict == Verdict.REJECTED


# ── Default-deny (unknown / unlisted resource) ────────────────────────────────

def test_unknown_resource_defaults_deny():
    """UNKNOWN tag (decomposer couldn't classify) is always denied."""
    store = PermissionStore()
    result = check_query("alex.CEO", "q", make_decomp("UNKNOWN"), store)
    assert result.verdict == Verdict.REJECTED
    assert result.sub_results[0].decision == Decision.DENY


def test_unlisted_resource_defaults_deny():
    """A valid-looking but unrecognised resource ID is default-denied."""
    store = PermissionStore()
    result = check_query(
        "uma.Finance_Manager", "q",
        make_decomp("some_table.some_field"),
        store,
    )
    assert result.verdict == Verdict.REJECTED


# ── Unknown user raises ───────────────────────────────────────────────────────

def test_unknown_user_raises():
    """Requesting access for a user not in users.yaml must raise AccessCheckError."""
    store = PermissionStore()
    try:
        check_query("nonexistent.user", "q", make_decomp("finance.view"), store)
        assert False, "should have raised AccessCheckError"
    except Exception:
        pass  # expected


# ── Wildcard allow (executive *) ──────────────────────────────────────────────

def test_wildcard_allow_executive():
    """Executive role allows every resource including edge cases."""
    store = PermissionStore()
    resources = [
        "hr.delete", "finance.admin", "it.deploy",
        "executive.confidential", "security.admin",
    ]
    result = check_query("alex.CEO", "q", make_decomp(*resources), store)
    assert result.verdict == Verdict.FULL, f"Expected FULL, got {result.verdict}"
    assert all(sr.decision == Decision.ALLOW for sr in result.sub_results)


# ── Auditor read-only pattern ─────────────────────────────────────────────────

def test_auditor_read_only():
    """Auditor can audit finance and HR but cannot edit them."""
    store = PermissionStore()
    result_allow = check_query(
        "anna.Internal_Auditor", "q",
        make_decomp("finance.audit", "hr.audit", "audit.reports"),
        store,
    )
    result_deny = check_query(
        "anna.Internal_Auditor", "q",
        make_decomp("finance.edit", "hr.delete"),
        store,
    )
    assert result_allow.verdict == Verdict.FULL
    assert result_deny.verdict == Verdict.REJECTED


# ── Phase 3a: role hierarchy inheritance ──────────────────────────────────────

def test_hierarchy_own_deny_overrides_inherited_allow():
    """senior_manager explicitly denies 'finance.approve', even though its
    descendant finance_manager explicitly ALLOWS it -- a role's own rules
    are checked in full before any descendant is ever consulted, so the
    own-level deny wins regardless of what's inherited."""
    store = PermissionStore()
    result = check_query(
        "amy.Senior_Manager_HR", "q",
        make_decomp("finance.approve"),
        store,
    )
    assert result.verdict == Verdict.REJECTED, f"Expected REJECTED, got {result.verdict}"
    assert result.sub_results[0].decision == Decision.DENY
    assert "descendant" not in result.sub_results[0].reason, (
        "own-rule deny should resolve before any descendant is consulted"
    )


def test_hierarchy_inherits_allow_from_descendant_when_silent():
    """senior_manager's own rules say nothing about 'hr.audit', but its
    descendant hr_manager explicitly allows it -- senior_manager should
    inherit that ALLOW (senior inherits from junior, per
    docs/phase3/role-hierarchy.md's corrected direction)."""
    store = PermissionStore()
    result = check_query(
        "amy.Senior_Manager_HR", "q",
        make_decomp("hr.audit"),
        store,
    )
    assert result.verdict == Verdict.FULL, f"Expected FULL, got {result.verdict}"
    assert result.sub_results[0].decision == Decision.ALLOW
    assert "descendant" in result.sub_results[0].reason, (
        "reason should be attributed to inheritance, not an own-level match"
    )


def test_hierarchy_cycle_raises_at_load_time():
    """A cycle in role_hierarchy.yaml must raise RoleHierarchyError when the
    file is loaded, not silently succeed or fail later at query time."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "roles:\n"
            "  a: { parent: b }\n"
            "  b: { parent: a }\n"
        )
        temp_path = f.name
    try:
        RoleHierarchy(known_roles={"a", "b"}, hierarchy_path=temp_path)
        assert False, "should have raised RoleHierarchyError for a cycle"
    except RoleHierarchyError:
        pass  # expected
    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__}  —  {e}")
        except Exception as e:
            print(f"ERROR: {t.__name__}  —  {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
