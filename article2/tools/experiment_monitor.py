#!/usr/bin/env python3
"""Read-only browser monitor for JSON-result experiment matrices."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>Federated experiment monitor</title><style>
body{margin:0;background:#101418;color:#e8edf2;font:14px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:28px}h1{margin:0 0 6px}.sub{color:#aeb9c5;margin-bottom:24px}.summary{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:22px}.card{background:#192128;border-radius:8px;padding:12px 16px;min-width:115px}.card b{display:block;font-size:22px}table{width:100%;border-collapse:collapse;background:#192128;border-radius:8px;overflow:hidden}th,td{padding:12px;text-align:left;border-bottom:1px solid #2d3944}th{color:#aeb9c5}.done{color:#77d89a}.running{color:#7bc8ff}.failed{color:#ff8e91}.pending{color:#d7b86f}@media(max-width:700px){main{padding:16px}table{font-size:12px}th,td{padding:8px 5px}}</style>
<main><h1>Federated experiment monitor</h1><div class=sub id=updated>Loading…</div><div class=summary id=summary></div><table><thead><tr><th>Attack</th><th>Defense</th><th>λ</th><th>Seed</th><th>Status</th><th>Round</th><th>TACC</th><th>ASR</th><th>RR</th></tr></thead><tbody id=rows></tbody></table></main>
<script>const f=v=>v==null?'—':(100*v).toFixed(2)+'%';const l=v=>v==null?'—':Number(v).toFixed(2);async function load(){let d=await fetch('/api/status',{cache:'no-store'}).then(r=>r.json());document.querySelector('#updated').textContent=`Updated ${new Date(d.updated_at).toLocaleTimeString()} · ${d.root}`;document.querySelector('#summary').innerHTML=Object.entries(d.counts).map(([k,v])=>`<div class=card><b>${v}</b>${k}</div>`).join('');document.querySelector('#rows').innerHTML=d.jobs.map(j=>`<tr><td>${j.attack}</td><td>${j.defense}</td><td>${l(j.lambda)}</td><td>${j.seed}</td><td class=${j.status}>${j.status}</td><td>${j.round}/${j.total_rounds}</td><td>${f(j.accuracy)}</td><td>${f(j.asr)}</td><td>${f(j.rr)}</td></tr>`).join('')}load();setInterval(load,5000)</script>"""


def _last_json_line(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 262144))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return {}


def _failed(log_path: Path) -> bool:
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16384))
            tail = stream.read().decode("utf-8", errors="replace")
        return "Traceback (most recent call last)" in tail
    except OSError:
        return False


def status(root: Path) -> dict:
    jobs = []
    for config_path in sorted(root.glob("_configs/**/*.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            attack = str(config["attacks"])
            defense = str(config["defenses"])
            seed = int(config["fed_config_overrides"]["seed"])
            total_rounds = int(config["fed_config_overrides"]["total_rounds"])
            task = str(config["task"])
            output_dir = Path(str(config["log_dir"]))
            lambda_value = config["fed_config_overrides"].get("svdd_lambda")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        directory = output_dir
        result_path = directory / f"{task}__{attack}__{defense}.json"
        record = {"attack": attack, "defense": defense, "seed": seed,
                  "total_rounds": total_rounds, "round": 0, "accuracy": None,
                  "asr": None, "rr": None, "lambda": lambda_value, "status": "pending"}
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rounds = result.get("rounds", [])
            if len(rounds) == total_rounds:
                evaluation = rounds[-1].get("evaluation", {})
                record.update(status="done", round=len(rounds), accuracy=evaluation.get("accuracy"),
                              asr=evaluation.get("backdoor_asr"), rr=evaluation.get("rr"))
                jobs.append(record)
                continue
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        streams = sorted(directory.glob(f"{task}__{attack}__{defense}__*.jsonl"))
        latest = _last_json_line(streams[-1]) if streams else {}
        if latest:
            evaluation = latest.get("evaluation", {})
            if not isinstance(evaluation, dict):
                evaluation = {}
            record.update(
                round=int(latest.get("round", 0)),
                accuracy=evaluation.get("accuracy"),
                asr=evaluation.get("backdoor_asr"),
                rr=evaluation.get("rr"),
            )
        if any(_failed(path) for path in (directory / f"{attack}.log", directory / "sensitivity.log", directory / "console.log")):
            record["status"] = "failed"
        elif latest:
            record["status"] = "running"
        jobs.append(record)
    counts = {key: sum(job["status"] == key for job in jobs) for key in ("done", "running", "failed", "pending")}
    return {"root": str(root), "updated_at": __import__("datetime").datetime.now().astimezone().isoformat(), "counts": counts, "jobs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18084)
    args = parser.parse_args()
    root = args.root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/status":
                body = json.dumps(status(root)).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
            else:
                body = PAGE.encode()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
