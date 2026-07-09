"""Static side-by-side replay artifacts for local world-model evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_world_model_watch(
    *,
    out_dir: str | Path,
    run_id: str,
    events: list[dict[str, Any]],
) -> dict[str, str | int]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    events_path = target / "events.jsonl"
    with events_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    report = {
        "schema_version": "learn-nethack.local-world-model-watch.v1",
        "run_id": run_id,
        "event_count": len(events),
        "events_path": str(events_path),
        "index_path": str(target / "index.html"),
    }
    (target / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "index.html").write_text(_watch_html(run_id), encoding="utf-8")
    return report


def terminal_text(chars) -> str:
    return "\n".join(bytes(row).decode("latin1") for row in chars)


def _watch_html(run_id: str) -> str:
    safe_run_id = json.dumps(run_id)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NetHack world-model proof</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #101214; color: #edf0f2; }}
    header {{ display: flex; align-items: center; gap: 12px; padding: 12px 16px;
      border-bottom: 1px solid #353b40; background: #171a1d; }}
    h1 {{ margin: 0; font-size: 16px; font-weight: 650; }}
    .run {{ color: #9aa4ab; font: 12px ui-monospace, monospace; }}
    .spacer {{ flex: 1; }}
    button {{ width: 34px; height: 30px; border: 1px solid #4b555d; background: #22272b;
      color: #edf0f2; cursor: pointer; font-size: 16px; }}
    button:disabled {{ opacity: .35; cursor: default; }}
    #counter {{ min-width: 72px; text-align: center; color: #c0c7cc; font-size: 13px; }}
    .meta {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1px;
      background: #353b40; border-bottom: 1px solid #353b40; }}
    .metric {{ background: #171a1d; padding: 9px 12px; min-height: 54px; }}
    .label {{ color: #8f9aa2; font-size: 11px; text-transform: uppercase; }}
    .value {{ margin-top: 4px; font: 13px ui-monospace, monospace; overflow-wrap: anywhere; }}
    main {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); min-height: calc(100vh - 109px); }}
    section {{ min-width: 0; border-right: 1px solid #353b40; }}
    section:last-child {{ border-right: 0; }}
    h2 {{ margin: 0; padding: 8px 10px; border-bottom: 1px solid #353b40;
      color: #c9d0d5; font-size: 12px; font-weight: 600; }}
    pre {{ margin: 0; padding: 10px; overflow: auto; white-space: pre; line-height: 1.08;
      font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; color: #d9dee1; }}
    @media (max-width: 1050px) {{
      main {{ grid-template-columns: 1fr 1fr; }}
      section {{ border-bottom: 1px solid #353b40; }}
      .meta {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 620px) {{
      main {{ grid-template-columns: 1fr; }}
      .meta {{ grid-template-columns: 1fr; }}
      pre {{ font-size: 9px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>NetHack world-model proof</h1>
    <span class="run" id="run"></span>
    <span class="spacer"></span>
    <button id="prev" title="Previous transition" aria-label="Previous transition">&#8592;</button>
    <span id="counter">0 / 0</span>
    <button id="next" title="Next transition" aria-label="Next transition">&#8594;</button>
  </header>
  <div class="meta">
    <div class="metric"><div class="label">Game / step</div><div class="value" id="position"></div></div>
    <div class="metric"><div class="label">Action</div><div class="value" id="action"></div></div>
    <div class="metric"><div class="label">Horizon</div><div class="value" id="horizon"></div></div>
    <div class="metric"><div class="label">Deterministic changed F1</div><div class="value" id="baseline-f1"></div></div>
    <div class="metric"><div class="label">Diffusion changed F1</div><div class="value" id="diffusion-f1"></div></div>
  </div>
  <main>
    <section><h2>Current terminal</h2><pre id="current"></pre></section>
    <section><h2>Ground truth</h2><pre id="truth"></pre></section>
    <section><h2>Deterministic</h2><pre id="baseline"></pre></section>
    <section><h2>Diffusion</h2><pre id="diffusion"></pre></section>
  </main>
  <script>
    const runId = {safe_run_id};
    let events = [];
    let index = 0;
    const byId = (id) => document.getElementById(id);
    const fixed = (value) => Number(value || 0).toFixed(4);
    function render() {{
      const event = events[index];
      byId('run').textContent = runId;
      byId('counter').textContent = events.length ? `${{index + 1}} / ${{events.length}}` : '0 / 0';
      byId('prev').disabled = index <= 0;
      byId('next').disabled = index >= events.length - 1;
      if (!event) return;
      byId('position').textContent = `${{event.gameid}} / ${{event.sequence_step}}`;
      byId('action').textContent = `${{event.action_id}} (${{event.key_label || 'unknown'}})`;
      byId('horizon').textContent = event.horizon;
      byId('baseline-f1').textContent = fixed(event.baseline_changed_f1);
      byId('diffusion-f1').textContent = fixed(event.diffusion_changed_f1);
      byId('current').textContent = event.current_frame;
      byId('truth').textContent = event.ground_truth_frame;
      byId('baseline').textContent = event.deterministic_frame;
      byId('diffusion').textContent = event.diffusion_frame;
    }}
    byId('prev').addEventListener('click', () => {{ index = Math.max(0, index - 1); render(); }});
    byId('next').addEventListener('click', () => {{ index = Math.min(events.length - 1, index + 1); render(); }});
    document.addEventListener('keydown', (event) => {{
      if (event.key === 'ArrowLeft') byId('prev').click();
      if (event.key === 'ArrowRight') byId('next').click();
    }});
    fetch('events.jsonl').then((response) => response.text()).then((text) => {{
      events = text.trim().split(/\\n+/).filter(Boolean).map((line) => JSON.parse(line));
      render();
    }});
  </script>
</body>
</html>
"""
