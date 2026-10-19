"""Export pinned remote branches for read-only analysis; never import their code."""
from pathlib import Path, PurePosixPath
import ast
import collections
import datetime
import hashlib
import io
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "docs/papers/hif4-public-references/test"
OUT = ROOT / "docs/papers/hif4-public-references/test-branch-snapshots-20260907"
AUDIT = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args])


def destination(base, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe repository member: {name!r}")
    return base.joinpath(*path.parts)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Snapshot already exists with different content: {path}")
    path.write_bytes(data)


def signature(data):
    result = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    try:
        tree = ast.parse(data.decode("utf-8-sig"))
        result["ast_sha256"] = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
        result["functions"] = [
            {"name": n.name, "start": n.lineno, "end": n.end_lineno,
             "ast_sha256": hashlib.sha256(ast.dump(n).encode()).hexdigest()}
            for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    except (SyntaxError, UnicodeError) as error:
        result["ast_error"] = str(error)
    return result


remote = git("ls-remote", "--heads", "origin").decode()
save(AUDIT / "remote-heads.tsv", remote.encode())
refs = sorted((ref.removeprefix("refs/heads/"), sha)
              for sha, ref in (line.split() for line in remote.splitlines()))
blob_cache = {}
branches = []
for branch, sha in refs:
    local = git("rev-parse", "refs/remotes/origin/" + branch).decode().strip()
    if local != sha:
        raise ValueError(f"Remote changed during snapshot: {branch}")
    slug = branch.replace("/", "--")
    base = OUT / slug
    info = {"branch": branch, "commit": sha, "snapshot": str(base.relative_to(ROOT)),
            "commit_metadata": git("show", "-s", "--format=%cI%n%s", sha).decode().strip(),
            "files": [], "archived_python": []}
    for record in git("ls-tree", "-rlz", sha).split(b"\0"):
        if not record:
            continue
        metadata, rawpath = record.split(b"\t", 1)
        mode, kind, oid, size = metadata.decode().split()
        name = rawpath.decode()
        item = {"path": name, "mode": mode, "kind": kind, "git_blob": oid,
                "bytes": int(size) if size.isdigit() else None}
        if kind != "blob":
            info["files"].append(item)
            continue
        data = blob_cache.setdefault(oid, None)
        if data is None:
            data = git("cat-file", "blob", oid)
            blob_cache[oid] = data
        try:
            decoded = data.decode("utf-8-sig")
            is_text = "\0" not in decoded
        except UnicodeError:
            is_text = False
        if is_text:
            save(destination(base, name), data)
            item["exported"] = True
            if name.endswith(".py"):
                item.update(signature(data))
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for member in archive.infolist():
                        if member.is_dir() or not member.filename.endswith(".py"):
                            continue
                        if member.file_size > 10_000_000:
                            raise ValueError("Unexpected oversized Python member")
                        code = archive.read(member)
                        relative = "_archives/" + name + "/" + member.filename
                        save(destination(base, relative), code)
                        info["archived_python"].append({"archive": name, "member": member.filename,
                            "export_path": relative, **signature(code)})
            except zipfile.BadZipFile:
                item["archive_error"] = "BadZipFile"
        info["files"].append(item)
    branches.append(info)

manifest = {"repository": "https://gitcode.com/Mrxxx7/test.git",
            "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "shallow": git("rev-parse", "--is-shallow-repository").decode().strip(),
            "remote_heads_verified": len(branches),
            "policy": "All Git branch objects fetched. UTF-8 text files and ZIP Python members exported; binary blobs remain available in Git. No repository code executed.",
            "branches": branches}
(AUDIT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
groups = collections.defaultdict(list)
for branch in branches:
    entry = next((x for x in branch["files"] if x["path"] == "solution/solution.py"), None)
    if entry:
        groups[entry.get("ast_sha256", entry["git_blob"])].append(branch["branch"])
print(json.dumps({"branches": len(branches), "unique_git_blobs": len(blob_cache),
    "exported_text_files": sum(bool(f.get("exported")) for b in branches for f in b["files"]),
    "archived_python_members": sum(len(b["archived_python"]) for b in branches),
    "main_solution_ast_groups": list(groups.values()),
    "snapshot_root": str(OUT)}, ensure_ascii=False, indent=2))
