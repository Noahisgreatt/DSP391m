"""VERSIONS (v1.0, v1.1, v2.0 ...): every time something changes, save a version.

    python -m src.release --note "model 1 trained"    # compare with the last version -> make the next one
    python -m src.release --list                      # all versions, one line each (also releases/README.md)
    python -m src.release --diff v1.0 v1.1            # exactly what changed between two versions

The number is decided for you, from what changed since the last version:
    MAJOR  v1.0 -> v2.0   a model was added (v1 = model 1 ... v4 = all four models)
    MINOR  v1.0 -> v1.1   same models, but the data, the code, a setting or the trained weights changed
    none                  nothing changed: no new version
A version = a git tag + releases/<version>.json (its metadata card, below). The tag pins the code and
dvc.lock; dvc.lock pins the data and every model file by md5 (stored on the DVC remote), so
    git checkout v1.0 && dvc pull        brings back exactly the code, data and models of v1.0
    dvc metrics diff v1.0 v2.0           DVC's own table of the numbers that changed
    dvc params diff v1.0 v2.0            DVC's own table of the settings that changed
Metadata card: date, git commit, note; data (md5 + size of each DVC-tracked dataset, videos, frames,
train / test split); code (md5 of src/ + api/); every setting (configs/*.yaml); per model: parameters,
md5 of the trained weights, test metrics (AUC micro / per-video / macro, EER, frames/s); behavior rules
metrics; software versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

MODELS = ("astt_baseline", "astt_improved", "aivad", "aivad_lite")
REL = Path("releases")
METRICS = ("auc_raw", "auc_norm", "auc_macro", "eer", "fps_scored")


def md5(paths) -> str:
    h = hashlib.md5()
    for p in sorted(paths):
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def read(p):
    p = Path(p)
    return (json.loads if p.suffix == ".json" else yaml.safe_load)(p.read_text()) if p.exists() else None


def git(*args) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"git {args[0]} failed: {r.stderr.strip()}")
    return r.stdout.strip()


# ---------------------------------------------------------------- what the project is right now
def snapshot(note="") -> dict:
    data = {}
    for f in sorted(Path("data/raw").glob("*.dvc")):              # DVC-tracked data (md5 = version of the data)
        o = read(f)["outs"][0]
        data[o["path"]] = {"md5": o["md5"], "size_gb": round(o.get("size", 0) / 1e9, 2), "files": o.get("nfiles")}
    sh, split = read("data/metadata/shanghaitech.json") or {}, read("data/metadata/split.json") or {}
    data["videos"] = {"shanghaitech": sh.get("videos"), "frames": sh.get("frames"),
                      "train": len(split.get("train", [])), "test": len(split.get("test", [])),
                      "manual": (read("data/metadata/manual.json") or {}).get("videos")}
    models = {}
    for m in MODELS:
        w, ev = Path(f"outputs/training/{m}/model.pth"), read(f"outputs/evaluation/{m}/metrics.json")
        if w.exists():
            models[m] = {"parameters": ((read(f"outputs/training/{m}/run.json") or {}).get("parameters") or {}).get("Total"),
                         "weights_md5": md5([w]), "test": {k: ev.get(k) for k in METRICS} if ev else None}
    try:
        import torch
        sw = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        sw = {"python": platform.python_version()}
    return {"note": note, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "--short", "HEAD"), "data": data,
            "code_md5": md5([*Path("src").rglob("*.py"), *Path("api").rglob("*.py")]),
            "settings": {f.stem: read(f) for f in sorted(Path("configs").glob("*.yaml"))},
            "models": models, "behavior": read("outputs/behavior/shanghaitech/metrics.json"), "software": sw}


# ---------------------------------------------------------------- comparing two cards
def flat(d, pre=""):
    out = {}
    for k, v in (d or {}).items():
        out.update(flat(v, f"{pre}{k}.") if isinstance(v, dict) else {f"{pre}{k}": v})
    return out


def changes(old: dict, new: dict) -> list[str]:
    """Human-readable list of everything that differs between two cards."""
    out = []
    for name in sorted(set(old["data"]) | set(new["data"])):
        a, b = old["data"].get(name), new["data"].get(name)
        if a != b:
            out.append(f"data {name}: {a} -> {b}")
    if old["code_md5"] != new["code_md5"]:
        out.append("code (src/, api/) changed")
    a, b = flat(old["settings"]), flat(new["settings"])
    out += [f"setting {k}: {a.get(k)} -> {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    for m in MODELS:
        a, b = old["models"].get(m), new["models"].get(m)
        if a and not b:
            out.append(f"{m}: removed")
        elif b and not a:
            auc = (b["test"] or {}).get("auc_raw")
            out.append(f"{m}: ADDED ({b['parameters'] or 0:,} parameters" + (f", AUC {auc * 100:.1f}%)" if auc is not None else ")"))
        elif a and b:
            if a["weights_md5"] != b["weights_md5"]:
                out.append(f"{m}: retrained (new weights)")
            ta, tb = a["test"] or {}, b["test"] or {}
            for k in METRICS:
                if ta.get(k) != tb.get(k) and tb.get(k) is not None:
                    pct = k != "fps_scored"
                    f = (lambda v: "-" if v is None else f"{v * 100:.1f}%") if pct else (lambda v: f"{v}")
                    delta = f" ({(tb[k] - ta[k]) * 100:+.1f})" if pct and ta.get(k) is not None else ""
                    out.append(f"{m}: {k} {f(ta.get(k))} -> {f(tb[k])}{delta}")
    return out


def versions() -> list[str]:
    key = lambda v: tuple(int(x) for x in v[1:].split("."))
    return sorted((f.stem for f in REL.glob("v*.json")), key=key)


def write_index():
    """releases/README.md: every version in one table (readable on GitHub)."""
    rows = ["# Versions", "", "Made with `python -m src.release`; details: `python -m src.release --diff vA vB`.", "",
            "| Version | Date | Built from commit | Models (AUC micro on the test set) | Data | Note |", "|---|---|---|---|---|---|"]
    for v in reversed(versions()):
        c = read(REL / f"{v}.json")
        ms = ", ".join(f"{m} {((d['test'] or {}).get('auc_raw') or 0) * 100:.1f}%" for m, d in c["models"].items()) or "-"
        rows.append(f"| **{v}** | {c['created_utc'][:10]} | `{c['git_commit']}` | {ms} | "
                    f"{c['data']['videos']['train']} train / {c['data']['videos']['test']} test videos | {c['note']} |")
    (REL / "README.md").write_text("\n".join(rows) + "\n")


# ---------------------------------------------------------------- commands
def release(note):
    trained = [f"evaluate@{m}" for m in MODELS if Path(f"outputs/training/{m}/model.pth").exists()]
    if trained and subprocess.run(["dvc", "status", "-q", *trained], capture_output=True).returncode:
        print("warning: some results are older than their code / settings (dvc status): run `dvc repro` first "
              "if you want this version to match")
    new, vs = snapshot(note), versions()
    if not vs:
        n = max(1, len(new["models"]))
        version, what = f"v{n}.0", ["first version"]
    else:
        old = read(REL / f"{vs[-1]}.json")
        what = changes(old, new)
        if not what:
            raise SystemExit(f"nothing changed since {vs[-1]}: no new version")
        major, minor = (int(x) for x in vs[-1][1:].split("."))
        added = [m for m in new["models"] if m not in old["models"]]
        version = f"v{major + len(added)}.0" if added else f"v{major}.{minor + 1}"
    REL.mkdir(exist_ok=True)
    (REL / f"{version}.json").write_text(json.dumps({"version": version, "changes": what, **new}, indent=2))
    write_index()
    print(f"{version}" + (f" (since {vs[-1]})" if vs else "") + ":\n  " + "\n  ".join(what))
    git("add", "-A", *[p for p in ("releases", "dvc.lock", "data/metadata", "data/raw", "outputs") if Path(p).exists()])
    git("commit", "-q", "-m", f"release {version}: {note or ', '.join(what)[:60]}")
    git("tag", "-a", version, "-m", note or version)
    print(f"\nsaved as git tag {version}. Upload it:  git push --follow-tags && dvc push")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--note", default="", help="what this version is, in a few words")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"))
    a = ap.parse_args()
    if a.list:
        write_index()
        print((REL / "README.md").read_text())
    elif a.diff:
        missing = [v for v in a.diff if not (REL / f"{v}.json").exists()]
        if missing:
            raise SystemExit(f"no such version: {missing} (see python -m src.release --list)")
        old, new = (read(REL / f"{v}.json") for v in a.diff)
        print(f"{a.diff[0]} -> {a.diff[1]}:\n  " + "\n  ".join(changes(old, new) or ["no differences"]))
    else:
        release(a.note)


if __name__ == "__main__":
    main()
