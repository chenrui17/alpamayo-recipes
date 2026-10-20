"""Verify public hashes and timing summaries; optional private strings stay local."""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secret-file", type=Path, help="Optional local-only exact credential check")
    parser.add_argument("--environment-file", type=Path, help="Optional local-only registry-host check")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root.parents[3]))
    from alpamayo1_5_sft.benchmark_h20.analyze import summarize

    spec = importlib.util.spec_from_file_location("archive_runs", root / "archive_runs.py")
    archive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive)
    secret_values = re.findall(r"密码\s+(\S+)", args.secret_file.read_text()) if args.secret_file else []
    if args.secret_file and not secret_values:
        parser.error("No credential found in --secret-file; exact-value check cannot be verified")
    environment = args.environment_file.read_text() if args.environment_file else ""
    private_hosts = archive.Redactor(environment=environment).private_hosts
    indexed = json.loads((root / "runs.json").read_text())["runs"]
    verified_hashes = 0
    for path in sorted(root.rglob("*provenance.json")):
        source = json.loads(path.read_text())
        entries = source.get("files", {source.get("file"): source})
        for relative, record in entries.items():
            assert relative, str(path)
            target = path.parent / relative
            assert hashlib.sha256(target.read_bytes()).hexdigest() == record["public_sha256"], str(target)
            verified_hashes += 1
    for row in indexed:
        folder = root / "runs" / row["run_id"]
        manifest = json.loads((folder / "manifest.json").read_text())
        source = json.loads((folder / "provenance.json").read_text())
        raw = json.loads((folder / "steps.json").read_text())
        assert manifest["status"] == "complete" and manifest["mode"] == "timing", row["run_id"]
        assert source["source_commit"] == manifest["code"]["head"] == row["source_commit"], row["run_id"]
        assert source["source_patch_sha256"] == manifest["code"]["tracked_diff_sha256"] == row["source_patch_sha256"], row["run_id"]
        computed = summarize(raw)["rank_max"]
        assert abs(computed["mean_seconds"] - row["mean_seconds"]) <= 1e-12, row["run_id"]
    profiles = list((root / "profile").glob("*/analysis.json"))
    for path in profiles:
        value = json.loads(path.read_text())
        window = value["capture_validation"]
        assert window["valid"] and window["observed_rank_processes"] == 8, str(path)
        assert len(value["devices"]) == 8 and len(value["optimizer_steps"]) == 32, str(path)
        assert not window["steps_without_gpu_kernels"], str(path)
    real_checks = list((root / "checks").glob("correctness_r[0-9]*.json"))
    real_checks = [path for path in real_checks if not path.name.endswith((".sources.json", ".provenance.json"))]
    dataset_checks = list((root / "checks").glob("data_equivalence_r[0-9]*.json"))
    dataset_checks = [path for path in dataset_checks if not path.name.endswith(".provenance.json")]
    for path in real_checks + dataset_checks:
        assert json.loads(path.read_text())["status"] == "passed", str(path)
    figure_path = root / "figures.json"
    if figure_path.is_file():
        figures = json.loads(figure_path.read_text())
        assert set(figures["timing_means"]) == {row["run_id"] for row in indexed if row.get("comparison_eligible", True)}
    ci_path = root / "checks/ci/final_ci_status.json"
    ci = json.loads(ci_path.read_text()) if ci_path.is_file() else []
    assert all(item["status"] == "complete" and item["returncode"] == 0 for item in ci)
    checked_files = 0
    checked_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "archive_audit.json":
            continue
        data = path.read_bytes()
        checked_files += 1
        checked_bytes += len(data)
        if path.suffix in {".png", ".pyc"}:
            continue
        text = data.decode("utf-8")
        assert not any(secret in text for secret in secret_values), str(path)
        assert not any(host in text for host in private_hosts), str(path)
        for match in archive.IP_RE.finditer(text):
            # The redactor/auditor contain this public address-range constant.
            if path.suffix == ".py" and match.group(0) == "100.64.0.0":
                continue
            import ipaddress
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            assert not (address.is_private or address in ipaddress.ip_network("100.64.0.0/10")), str(path)
    report = {
        "status": "passed", "date": "2026-10-20", "complete_timing_runs": len(indexed),
        "comparison_eligible_runs": sum(row.get("comparison_eligible", True) for row in indexed),
        "profile_runs": len(profiles), "strict_profile_windows_verified": len(profiles),
        "profile_clipped_kernel_counts": {path.parent.name: json.loads(path.read_text())["capture_validation"]["clipped_kernel_count"] for path in profiles},
        "passed_real_model_reports": len(real_checks), "passed_dataset_reports": len(dataset_checks),
        "completed_clean_checkout_checks_including_compile": len(ci),
        "verified_public_file_hashes": verified_hashes, "verified_raw_timing_means": len(indexed),
        "scanned_files": checked_files, "scanned_bytes": checked_bytes,
        "exact_credential_check_requested": bool(args.secret_file),
        "exact_credential_values_checked": len(secret_values),
        "actual_registry_host_check_requested": bool(args.environment_file),
        "private_ipv4_check": True, "sensitive_matches": 0,
        "privacy_scan_exception": "The CGNAT network constant in the public redactor/auditor source is not a host address.",
        "note": "Source patch hashes were independently verified during import; patch bodies remain outside this public archive."
    }
    (root / "archive_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
