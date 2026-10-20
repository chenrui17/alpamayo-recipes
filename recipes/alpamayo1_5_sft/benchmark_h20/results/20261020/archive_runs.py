"""Archive complete runs as sanitized, verifiable public timing evidence.

Only result files are read remotely. No training command is launched. Source
archives/logs/patch bodies and credentials are deliberately never copied here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import ipaddress
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path


FILES = ("manifest.json", "config.resolved.yaml", "steps.json", "summary.json", "training/config.yaml")
DIGEST_RE = re.compile(r"sha256:[a-fA-F0-9]{64}")
IP_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Redactor:
    def __init__(self, secrets=(), private_roots=(), environment=""):
        self.secrets = [value for value in secrets if value]
        self.private_roots = sorted(private_roots, key=len, reverse=True)
        self.private_hosts = set()
        for match in re.finditer(r"([A-Za-z0-9.-]+(?::\d+)?)/[^\s\"'\[\],]+@sha256:[a-fA-F0-9]{64}", environment):
            self.private_hosts.add(match.group(1))
        self.replacements = 0

    def text(self, value):
        before = value
        for secret in self.secrets:
            value = value.replace(secret, "<redacted-secret>")
        for root in self.private_roots:
            value = value.replace(root, "/work")
        # Preserve only the immutable image digest, never a registry/name/tag.
        value = re.sub(r"[A-Za-z0-9.-]+(?::\d+)?/[^\s\"'\[\],]+@(sha256:[a-fA-F0-9]{64})", r"image@\1", value)
        for host in self.private_hosts:
            value = value.replace(host, "<redacted-registry>")
        value = re.sub(r"(?i)(?:https?://)?(?:[\w.-]+\.)?nvcr\.io(?:/[^\s\"']*)?", "<redacted-registry>", value)
        value = re.sub(r"H20-GPU-(\d+)", r"H20-node-\1", value)

        def private_ip(match):
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                return match.group(0)
            if address.is_private or address in ipaddress.ip_network("100.64.0.0/10"):
                return "<redacted-private-ip>"
            return match.group(0)

        value = IP_RE.sub(private_ip, value)
        self.replacements += int(before != value)
        return value

    def tree(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.tree(item) for item in value]
        if isinstance(value, dict):
            return {key: "<redacted-secret>" if key.lower() in {
                "password", "passwd", "api_key", "access_token", "refresh_token", "authorization", "secret"
            } and item is not None else self.tree(item) for key, item in value.items()}
        return value

    def file(self, name, data):
        text = data.decode("utf-8")
        if name.endswith(".json"):
            return (json.dumps(self.tree(json.loads(text)), indent=2, allow_nan=False) + "\n").encode()
        return self.text(text).encode()


def local_run(root, name):
    folder = root / name
    if not (folder / "manifest.json").exists():
        return None
    names = [*FILES, *(str(path.relative_to(folder)) for path in (folder / "training").glob("stage2*.json"))]
    files = {item: (folder / item).read_bytes() for item in names if (folder / item).is_file()}
    patch = folder / "code.patch"
    if patch.is_file():
        files["__patch_sha256"] = sha(patch.read_bytes()).encode()
    return files


def tar_runs(data, wanted):
    runs = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if not parts or member.name.startswith("/") or ".." in parts or parts[0] not in wanted:
                continue
            relative = str(Path(*parts[1:]))
            allowed = relative in FILES or (relative.startswith("training/stage2") and relative.endswith(".json"))
            if allowed:
                runs.setdefault(parts[0], {})[relative] = archive.extractfile(member).read()
            elif relative == "code.patch":
                runs.setdefault(parts[0], {})["__patch_sha256"] = sha(archive.extractfile(member).read()).encode()
            elif relative == "__patch_sha256":
                runs.setdefault(parts[0], {})[relative] = archive.extractfile(member).read()
    return runs


def fetch_runs(helper, remote_root, names):
    # Run a read-only standard-library exporter through the existing SSH helper.
    # Values are passed as literals inside Python; no shell interpolation occurs.
    program = """import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path(ROOT)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in NAMES:
  folder=root/name
  manifest=json.loads((folder/'manifest.json').read_text())
  if manifest.get('status')!='complete': raise RuntimeError(name+' is not complete')
  names=list(FILES)+[str(p.relative_to(folder)) for p in (folder/'training').glob('stage2*.json')]
  for item in names:
   path=folder/item
   if path.is_file(): archive.add(path,arcname=name+'/'+item,recursive=False)
  patch=folder/'code.patch'
  if patch.is_file():
   digest=hashlib.sha256(patch.read_bytes()).hexdigest().encode()
   header=tarfile.TarInfo(name+'/__patch_sha256');header.size=len(digest)
   archive.addfile(header,io.BytesIO(digest))
""".replace("ROOT", repr(remote_root)).replace("NAMES", repr(names)).replace("FILES", repr(FILES))
    command = "python3 - <<'ARCHIVE_READ_ONLY_PY'\n" + program + "\nARCHIVE_READ_ONLY_PY"
    result = subprocess.run([sys.executable, str(helper), "--timeout", "60", command], stdout=subprocess.PIPE, check=True)
    return tar_runs(result.stdout, set(names))


def archive_run(output, name, files, redactor):
    from alpamayo1_5_sft.benchmark_h20.analyze import summarize

    manifest = json.loads(files["manifest.json"])
    steps = json.loads(files["steps.json"])
    saved = json.loads(files["summary.json"])
    if manifest.get("status") != "complete" or manifest.get("mode") != "timing":
        raise ValueError(name + " is not a completed timing run")
    if steps.get("run_id") != name or not steps.get("complete"):
        raise ValueError(name + " has an incomplete or mismatched steps.json")
    recalculated = summarize(steps)
    if abs(recalculated["rank_max"]["mean_seconds"] - saved["rank_max"]["mean_seconds"]) > 1e-12:
        raise ValueError(name + " saved mean does not match raw step records")
    code = manifest["code"]
    actual_patch = files.get("__patch_sha256", b"").decode().strip()
    if not actual_patch or actual_patch != code["tracked_diff_sha256"]:
        raise ValueError(name + " source patch SHA256 is missing or mismatched")
    folder = output / "runs" / name
    folder.mkdir(parents=True, exist_ok=True)
    provenance = {"run_id": name, "source_commit": code["head"], "source_patch_sha256": actual_patch,
                  "status": "complete", "mode": "timing", "world_size": steps["world_size"],
                  "steady_window": [steps["steady_start_step"], steps["steady_end_step"]], "files": {}}
    for relative, raw in sorted(files.items()):
        if relative.startswith("__"):
            continue
        public = redactor.file(relative, raw)
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(public)
        provenance["files"][relative] = {"source_sha256": sha(raw), "public_sha256": sha(public), "sanitized": raw != public}
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def build_index(output):
    runs = []
    annotation_path = output / "annotations.json"
    annotations = json.loads(annotation_path.read_text()) if annotation_path.is_file() else {}
    for folder in sorted((output / "runs").iterdir()):
        provenance = folder / "provenance.json"
        if not provenance.is_file():
            continue
        source = json.loads(provenance.read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        environment = manifest.get("benchmark_environment", {})
        overrides = dict(arg.lstrip("+").split("=", 1) for arg in manifest.get("command", []) if "=" in arg)
        summary = json.loads((folder / "summary.json").read_text())
        metrics = summary["rank_max"]
        data_path = folder / "training/stage2_data.json"
        data = json.loads(data_path.read_text()) if data_path.exists() else {}
        runs.append({"run_id": folder.name, "source_commit": source["source_commit"],
                     "source_patch_sha256": source["source_patch_sha256"], "world_size": source["world_size"],
                     "steady_start_step": source["steady_window"][0], "steady_end_step": source["steady_window"][1],
                     **metrics, "max_memory_allocated_bytes": summary.get("max_memory_allocated_bytes"),
                     "preload_seconds_rank0": data.get("preprocessing_seconds"),
                     "data_report_rank": 0 if data else None,
                     "cache_tensor_bytes_per_rank": data.get("cache_tensor_bytes"),
                     "requested_skip_head": environment.get("APPLY_VLM_SKIP_HEAD"),
                     "requested_liger": environment.get("APPLY_LIGER_KERNEL"),
                     "requested_prefix": environment.get("APPLY_VLM_PREFIX_ONLY"),
                     "requested_factor_cache": environment.get("APPLY_TRAJECTORY_FACTOR_CACHE"),
                     "requested_preload": environment.get("APPLY_PRELOAD"),
                     "requested_sample_cache": environment.get("APPLY_SAMPLE_CACHE"),
                     "ddp_bucket_cap_mb_override": overrides.get("trainer.ddp_bucket_cap_mb"),
                     "ddp_find_unused_override": overrides.get("trainer.ddp_find_unused_parameters"),
                     "dataloader_workers_override": overrides.get("trainer.dataloader_num_workers"),
                     "provenance": str(provenance.relative_to(output)),
                     "candidate_activation": annotations.get(folder.name, {}).get("candidate_activation"),
                     "comparison_eligible": annotations.get(folder.name, {}).get("comparison_eligible", True),
                     "annotation": annotations.get(folder.name, {}).get("note")})
    (output / "runs.json").write_text(json.dumps({"time_unit": "seconds", "runs": runs}, indent=2) + "\n")
    if runs:
        with (output / "runs.csv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(dict.fromkeys(key for row in runs for key in row)))
            writer.writeheader()
            writer.writerows(runs)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--local-source", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--remote-helper", type=Path)
    parser.add_argument("--remote-results-root")
    parser.add_argument("--secret-file", type=Path, help="Read credential only for exact-value redaction; never copied")
    parser.add_argument("--environment-file", type=Path)
    parser.add_argument("--private-root", action="append", default=[])
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    for name in args.runs:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            parser.error("Run names must be simple directory names")
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    args.output.mkdir(parents=True, exist_ok=True)
    secrets = re.findall(r"密码\s+(\S+)", args.secret_file.read_text()) if args.secret_file else []
    if args.secret_file and not secrets:
        parser.error("No credential found in --secret-file; exact-value redaction cannot be verified")
    environment = args.environment_file.read_text() if args.environment_file else ""
    redactor = Redactor(secrets, args.private_root, environment)
    sources = tar_runs(args.source_archive.read_bytes(), set(args.runs)) if args.source_archive else {}
    if args.local_source:
        for name in args.runs:
            source = local_run(args.local_source, name)
            if source is not None:
                sources[name] = source
    missing = [name for name in args.runs if name not in sources]
    if missing:
        if not args.remote_helper or not args.remote_results_root:
            parser.error("Missing local runs need --remote-helper and --remote-results-root")
        sources.update(fetch_runs(args.remote_helper, args.remote_results_root, missing))
    for name in args.runs:
        provenance = archive_run(args.output, name, sources[name], redactor)
        print(name, provenance["source_commit"][:12], provenance["source_patch_sha256"][:12], "verified")
    runs = build_index(args.output)
    print(json.dumps({"archived_complete_runs": len(runs), "redacted_string_count": redactor.replacements}))


if __name__ == "__main__":
    main()
