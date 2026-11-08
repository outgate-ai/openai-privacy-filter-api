#!/usr/bin/env python3
"""Resolve and fetch a released OPF model for opf-api (stdlib only).

Contract: /workspace/opf-release/CONTRACT.md (opf-model-manifest/1, opf-channel/1).

    eval "$(python3 opf_models.py fetch --exports)"   # before `exec opf-api`

Reads OPF_API_MODEL_ID or OPF_API_MODEL_CHANNEL. Unset, or a model that resolves
to the upstream baseline, prints nothing: opf-api then behaves exactly as before
(upstream checkpoint in ~/.opf/privacy_filter). Any failure falls back to a
verified cache of the requested model, then to the baseline, unless
OPF_MODELS_REQUIRED=true.

Also the S3 client the publisher (tools/publish_model.py) uses.
"""

from __future__ import annotations

import argparse
import configparser
import contextlib
import datetime as dt
import fcntl
import hashlib
import hmac
import http.client
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_BUCKET = "outgate-prod-opf-models"
DEFAULT_PREFIX = "opf"
DEFAULT_REGION = "eu-central-1"
BASELINE_IDS = {"opf-v0-baseline", "baseline", "upstream"}
MODEL_ID_RE = re.compile(r"^opf-v[0-9]+[a-z0-9-]*$")
CHANNEL_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
MANIFEST_SCHEMA = "opf-model-manifest/1"
CHANNEL_SCHEMA = "opf-channel/1"
CHUNK = 1 << 20
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class S3Error(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"S3 {status}: {message}")
        self.status = status


# ---------------------------------------------------------------------------
# credentials


def _credentials_from_web_identity(region: str) -> dict | None:
    role = os.environ.get("AWS_ROLE_ARN")
    token_file = os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
    if not role or not token_file:
        return None
    token = Path(token_file).read_text().strip()
    query = urllib.parse.urlencode({
        "Action": "AssumeRoleWithWebIdentity",
        "Version": "2011-06-15",
        "RoleArn": role,
        "RoleSessionName": os.environ.get("AWS_ROLE_SESSION_NAME", "opf-models"),
        "WebIdentityToken": token,
        "DurationSeconds": "3600",
    })
    conn = http.client.HTTPSConnection(f"sts.{region}.amazonaws.com", timeout=30)
    conn.request("GET", "/?" + query, headers={"Accept": "application/xml"})
    resp = conn.getresponse()
    body = resp.read()
    if resp.status != 200:
        raise S3Error(resp.status, "AssumeRoleWithWebIdentity failed")
    ns = {"s": "https://sts.amazonaws.com/doc/2011-06-15/"}
    creds = ET.fromstring(body).find(".//s:Credentials", ns)
    return {
        "key": creds.findtext("s:AccessKeyId", namespaces=ns),
        "secret": creds.findtext("s:SecretAccessKey", namespaces=ns),
        "token": creds.findtext("s:SessionToken", namespaces=ns),
    }


def _credentials_from_process() -> dict | None:
    path = Path(os.environ.get("AWS_CONFIG_FILE", Path.home() / ".aws" / "config"))
    if not path.exists():
        return None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    profile = os.environ.get("AWS_PROFILE", "default")
    section = profile if profile == "default" else f"profile {profile}"
    command = cfg.get(section, "credential_process", fallback=None)
    if not command:
        return None
    out = json.loads(subprocess.run(shlex.split(command), check=True, capture_output=True, text=True).stdout)
    return {"key": out["AccessKeyId"], "secret": out["SecretAccessKey"], "token": out.get("SessionToken")}


def load_credentials(region: str) -> dict:
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return {
            "key": os.environ["AWS_ACCESS_KEY_ID"],
            "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
            "token": os.environ.get("AWS_SESSION_TOKEN"),
        }
    creds = _credentials_from_web_identity(region) or _credentials_from_process()
    if not creds:
        raise RuntimeError("no AWS credentials (env keys, web identity or credential_process)")
    return creds


# ---------------------------------------------------------------------------
# S3 (SigV4, virtual-hosted style, one bucket)


def _sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


class S3:
    def __init__(self, bucket: str, region: str, creds: dict | None = None, endpoint: str | None = None):
        self.bucket = bucket
        self.region = region
        self.creds = creds or load_credentials(region)
        # endpoint override is for tests only (http://127.0.0.1:port, path style)
        self.endpoint = endpoint
        self.host = f"{bucket}.s3.{region}.amazonaws.com"

    def _conn(self):
        if self.endpoint:
            u = urllib.parse.urlparse(self.endpoint)
            return http.client.HTTPConnection(u.hostname, u.port, timeout=300), u.netloc
        return http.client.HTTPSConnection(self.host, timeout=300), self.host

    def _path(self, key: str) -> str:
        path = "/" + urllib.parse.quote(key, safe="/-_.~")
        return f"/{self.bucket}{path}" if self.endpoint else path

    def request(self, method: str, key: str, *, query: dict | None = None, body=None,
                payload_sha256: str = EMPTY_SHA256, headers: dict | None = None):
        conn, host = self._conn()
        path = self._path(key)
        now = dt.datetime.now(dt.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        datestamp = now.strftime("%Y%m%d")
        hdrs = {"host": host, "x-amz-date": amz_date, "x-amz-content-sha256": payload_sha256}
        if self.creds.get("token"):
            hdrs["x-amz-security-token"] = self.creds["token"]
        for k, v in (headers or {}).items():
            hdrs[k.lower()] = str(v)
        q = sorted((query or {}).items())
        canonical_query = "&".join(
            f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(str(v), safe='-_.~')}" for k, v in q
        )
        signed = sorted(h for h in hdrs if h != "content-length")
        canonical = "\n".join([
            method, path, canonical_query,
            "".join(f"{h}:{hdrs[h].strip()}\n" for h in signed),
            ";".join(signed), payload_sha256,
        ])
        scope = f"{datestamp}/{self.region}/s3/aws4_request"
        to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
        k = _sign(("AWS4" + self.creds["secret"]).encode(), datestamp)
        for part in (self.region, "s3", "aws4_request"):
            k = _sign(k, part)
        signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
        hdrs["authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self.creds['key']}/{scope}, "
            f"SignedHeaders={';'.join(signed)}, Signature={signature}"
        )
        url = path + (("?" + canonical_query) if canonical_query else "")
        conn.request(method, url, body=body, headers=hdrs)
        return conn.getresponse()

    @staticmethod
    def _fail(resp, key: str):
        body = resp.read()[:500].decode(errors="replace")
        code = re.search(r"<Code>([^<]+)</Code>", body)
        raise S3Error(resp.status, f"{key}: {code.group(1) if code else resp.reason}")

    def head(self, key: str) -> dict | None:
        resp = self.request("HEAD", key)
        resp.read()
        if resp.status == 404:
            return None
        if resp.status != 200:
            raise S3Error(resp.status, f"HEAD {key}")
        return {k.lower(): v for k, v in resp.getheaders()}

    def get_bytes(self, key: str) -> bytes:
        resp = self.request("GET", key)
        if resp.status != 200:
            self._fail(resp, key)
        return resp.read()

    def get_to_file(self, key: str, dest: Path) -> tuple[str, int]:
        """Stream an object to dest; return (sha256, size)."""
        resp = self.request("GET", key)
        if resp.status != 200:
            self._fail(resp, key)
        h = hashlib.sha256()
        size = 0
        with open(dest, "wb") as f:
            while chunk := resp.read(CHUNK):
                h.update(chunk)
                size += len(chunk)
                f.write(chunk)
        return h.hexdigest(), size

    def digest(self, key: str) -> tuple[str, int]:
        """sha256 and size of an object, streamed without touching disk."""
        resp = self.request("GET", key)
        if resp.status != 200:
            self._fail(resp, key)
        h = hashlib.sha256()
        size = 0
        while chunk := resp.read(CHUNK):
            h.update(chunk)
            size += len(chunk)
        return h.hexdigest(), size

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/json") -> None:
        sha = hashlib.sha256(data).hexdigest()
        resp = self.request("PUT", key, body=data, payload_sha256=sha, headers={
            "content-length": len(data), "content-type": content_type, "x-amz-meta-sha256": sha,
        })
        if resp.status != 200:
            self._fail(resp, key)
        resp.read()

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        # S3 rejects the body if it does not hash to x-amz-content-sha256.
        size = path.stat().st_size
        with open(path, "rb") as f:
            resp = self.request("PUT", key, body=f, payload_sha256=sha256, headers={
                "content-length": size, "content-type": "application/octet-stream", "x-amz-meta-sha256": sha256,
            })
        if resp.status != 200:
            self._fail(resp, key)
        resp.read()

    def list(self, prefix: str, delimiter: str | None = None) -> tuple[list[str], list[str]]:
        keys, prefixes, token = [], [], None
        while True:
            query = {"list-type": "2", "prefix": prefix}
            if delimiter:
                query["delimiter"] = delimiter
            if token:
                query["continuation-token"] = token
            resp = self.request("GET", "", query=query)
            if resp.status != 200:
                self._fail(resp, prefix)
            root = ET.fromstring(resp.read())
            for el in root.iter():
                el.tag = el.tag.rpartition("}")[2]
            keys += [c.findtext("Key") for c in root.findall("Contents")]
            prefixes += [c.findtext("Prefix") for c in root.findall("CommonPrefixes")]
            if root.findtext("IsTruncated") != "true":
                return keys, prefixes
            token = root.findtext("NextContinuationToken")


# ---------------------------------------------------------------------------
# manifest helpers (shared with the publisher)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def safe_relpath(p: str) -> str:
    parts = p.split("/")
    if not p or p.startswith("/") or any(x in ("", ".", "..") for x in parts):
        raise ValueError(f"unsafe path in manifest: {p!r}")
    return p


def check_manifest(manifest: dict, model_id: str) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unknown manifest schema {manifest.get('schema')!r}")
    if manifest.get("model_id") != model_id:
        raise ValueError(f"manifest is for {manifest.get('model_id')!r}, not {model_id!r}")
    for f in manifest["files"]:
        safe_relpath(f["path"])
        if not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
            raise ValueError(f"bad sha256 for {f['path']}")


def is_baseline(model_id: str | None) -> bool:
    return model_id in BASELINE_IDS


# ---------------------------------------------------------------------------
# fetch


def log(msg: str) -> None:
    print(f"opf-models: {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _verified_cache(cache: Path, include: tuple[str, ...]) -> Path | None:
    """A cache dir whose recorded manifest still matches every included file."""
    try:
        manifest = json.loads((cache / "manifest.json").read_text())
        for f in manifest["files"]:
            if not f["path"].startswith(include):
                continue
            p = cache / safe_relpath(f["path"])
            if not p.is_file() or p.stat().st_size != f["size"] or sha256_file(p) != f["sha256"]:
                return None
        return cache
    except (OSError, ValueError, KeyError):
        return None


def _sync(s3: S3, prefix: str, model_id: str, manifest: dict, cache: Path, include: tuple[str, ...]) -> None:
    for f in manifest["files"]:
        if not f["path"].startswith(include):
            continue
        dest = cache / safe_relpath(f["path"])
        if dest.is_file() and dest.stat().st_size == f["size"] and sha256_file(dest) == f["sha256"]:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        log(f"downloading {model_id}/{f['path']} ({f['size']} bytes)")
        sha, size = s3.get_to_file(f"{prefix}/{model_id}/{f['path']}", part)
        if sha != f["sha256"] or size != f["size"]:
            part.unlink(missing_ok=True)
            raise ValueError(f"{f['path']}: sha256/size mismatch after download")
        os.replace(part, dest)


def resolve_and_fetch(*, model_id: str | None, channel: str | None, home: Path, bucket: str, prefix: str,
                      region: str, include: tuple[str, ...] = ("runtime/",), s3: S3 | None = None) -> tuple[str, Path] | None:
    """Return (model_id, cache dir) for a released model, or None for the baseline. Raises on failure."""
    root = home / "models"
    expected_sha = None
    if not model_id:
        if not CHANNEL_RE.match(channel):
            raise ValueError(f"bad channel {channel!r}")
        s3 = s3 or S3(bucket, region)
        pointer = json.loads(s3.get_bytes(f"{prefix}/channels/{channel}.json"))
        if pointer.get("schema") != CHANNEL_SCHEMA:
            raise ValueError(f"unknown channel schema {pointer.get('schema')!r}")
        model_id, expected_sha = pointer["model_id"], pointer.get("manifest_sha256")
        root.mkdir(parents=True, exist_ok=True)
        (root / f"channel-{channel}.json").write_text(json.dumps(pointer))
        log(f"channel {channel} -> {model_id}")
    if is_baseline(model_id):
        return None
    if not MODEL_ID_RE.match(model_id):
        raise ValueError(f"bad model id {model_id!r}")
    s3 = s3 or S3(bucket, region)
    raw = s3.get_bytes(f"{prefix}/{model_id}/manifest.json")
    if expected_sha and hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"{model_id}/manifest.json does not match the channel's manifest_sha256")
    manifest = json.loads(raw)
    check_manifest(manifest, model_id)
    cache = root / model_id
    with _lock(root):
        cache.mkdir(parents=True, exist_ok=True)
        _sync(s3, prefix, model_id, manifest, cache, include)
        (cache / "manifest.json").write_bytes(raw)
    return model_id, cache


def _offline_fallback(model_id: str | None, channel: str | None, home: Path, include: tuple[str, ...]) -> tuple[str, Path] | None:
    root = home / "models"
    if not model_id and channel:
        with contextlib.suppress(OSError, ValueError, KeyError):
            model_id = json.loads((root / f"channel-{channel}.json").read_text())["model_id"]
    if not model_id or is_baseline(model_id) or not MODEL_ID_RE.match(model_id):
        return None
    cache = _verified_cache(root / model_id, include)
    return (model_id, cache) if cache else None


def exports_for(model_id: str, cache: Path) -> dict[str, str]:
    manifest = json.loads((cache / "manifest.json").read_text())
    rt = manifest.get("runtime", {})
    env = {"OPF_API_MODEL_PATH": str(cache / rt.get("dir", "runtime")), "OPF_API_MODEL_RESOLVED": model_id}
    cal = rt.get("viterbi_calibration")
    if cal and (cache / cal).is_file():
        env["OPF_API_VITERBI_CALIBRATION_PATH"] = str(cache / cal)
    return env


def cmd_fetch(args) -> int:
    # --shadow reads OPF_API_SHADOW_MODEL_ID / _CHANNEL and exports OPF_API_SHADOW_*;
    # a shadow that resolves to the baseline still exports its name so it is enabled.
    var = "OPF_API_SHADOW_" if args.shadow else "OPF_API_"
    model_id = (args.model_id or os.environ.get(var + "MODEL_ID") or "").strip() or None
    channel = (args.channel or os.environ.get(var + "MODEL_CHANNEL") or "").strip() or None
    required = args.required or os.environ.get("OPF_MODELS_REQUIRED", "").lower() in ("1", "true", "yes")
    home = Path(args.home or os.environ.get("OPF_MODELS_HOME") or Path.home() / ".opf")
    bucket = os.environ.get("OPF_MODELS_BUCKET", DEFAULT_BUCKET)
    prefix = os.environ.get("OPF_MODELS_PREFIX", DEFAULT_PREFIX).strip("/")
    region = os.environ.get("OPF_MODELS_REGION", DEFAULT_REGION)
    # The ONNX backend serves onnx/ with card/ (labels, tokenizer, calibration);
    # it never needs the 2.8 GB torch checkpoint.
    onnx = os.environ.get("OPF_API_BACKEND", "").lower() == "onnx"
    include = tuple(args.include or (["onnx/", "card/"] if onnx else ["runtime/"]))
    if not model_id and not channel:
        return 0
    try:
        result = resolve_and_fetch(model_id=model_id, channel=channel, home=home, bucket=bucket,
                                   prefix=prefix, region=region, include=include)
    except Exception as exc:  # noqa: BLE001 - every failure has the same fallback
        result = _offline_fallback(model_id, channel, home, include)
        if result:
            log(f"WARNING: {exc}; using the verified cache of {result[0]}")
        elif required:
            log(f"ERROR: {exc} (OPF_MODELS_REQUIRED=true, not falling back)")
            return 1
        else:
            log(f"WARNING: {exc}; falling back to the upstream baseline")
            return 0
    if result is None:
        log("baseline selected (upstream checkpoint)")
        if args.shadow:
            print(f"{'export ' if args.exports else ''}OPF_API_SHADOW_MODEL_RESOLVED=opf-v0-baseline")
        return 0
    rid, cache = result
    exports = {"OPF_API_MODEL_PATH": str(cache), "OPF_API_MODEL_RESOLVED": rid} if onnx else exports_for(rid, cache)
    for k, v in exports.items():
        if args.shadow:
            k = k.replace("OPF_API_", "OPF_API_SHADOW_", 1)
        print(f"{'export ' if args.exports else ''}{k}={shlex.quote(v)}")
    log(f"{rid} ready at {cache}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="resolve OPF_API_MODEL_ID / OPF_API_MODEL_CHANNEL and download")
    f.add_argument("--model-id")
    f.add_argument("--channel")
    f.add_argument("--home", help="default ~/.opf (OPF_MODELS_HOME)")
    f.add_argument("--include", action="append", help="path prefixes to fetch (default runtime/)")
    f.add_argument("--required", action="store_true")
    f.add_argument("--exports", action="store_true", help="print `export K=V` lines for eval")
    f.add_argument("--shadow", action="store_true", help="resolve the shadow model (OPF_API_SHADOW_MODEL_ID / _CHANNEL)")
    f.set_defaults(func=cmd_fetch)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
