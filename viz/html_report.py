"""HTML report generation for evaluate_viz tracking results."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluate_viz import ProcessedSequenceSummary, RunMeta


_CSS = """
body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #0f0f0f; color: #e0e0e0; }
h1 { font-size: 1.4rem; margin-bottom: 0.25rem; color: #fff; }
h2 { font-size: 1.1rem; margin: 2rem 0 0.5rem; color: #ccc; border-bottom: 1px solid #333; padding-bottom: 0.25rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; font-size: 0.85rem; }
th, td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #2a2a2a; }
th { color: #888; font-weight: normal; width: 160px; }
td { color: #ddd; word-break: break-all; }
.stats { font-size: 0.85rem; color: #aaa; margin: 0.5rem 0 0.75rem; }
.bar-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.3rem 0; font-size: 0.82rem; }
.bar-label { width: 140px; color: #bbb; flex-shrink: 0; }
.bar-track { flex: 1; background: #1e1e1e; border-radius: 3px; height: 14px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 3px; }
.bar-count { width: 50px; text-align: right; color: #888; flex-shrink: 0; }
.bar-oob  { background: #ff4040; }
.bar-maxd { background: #ffa020; }
.bar-inv  { background: #a0a0a0; }
img.gif   { display: block; max-width: 100%; border: 1px solid #333; border-radius: 4px; margin-top: 0.75rem; }
.subtitle { color: #666; font-size: 0.8rem; margin-top: 0.1rem; }
"""


def _bar_html(label: str, css_class: str, count: int, total_deaths: int) -> str:
    pct = (count / total_deaths * 100) if total_deaths > 0 else 0.0
    return (
        f'<div class="bar-row">'
        f'<div class="bar-label">{html.escape(label)}</div>'
        f'<div class="bar-track"><div class="bar-fill {css_class}" style="width:{pct:.1f}%"></div></div>'
        f'<div class="bar-count">{count}</div>'
        f"</div>"
    )


def _sequence_section_html(summary: ProcessedSequenceSummary) -> str:
    total_deaths = summary.total_points - summary.final_alive
    alive_pct = summary.final_alive / summary.total_points * 100 if summary.total_points > 0 else 0.0
    dc = summary.death_counts
    oob = dc.get("out_of_bounds", 0)
    maxd = dc.get("max_displacement", 0)
    inv = dc.get("invalid_flow", 0)

    gif_name = html.escape(summary.output_path.name)
    seq_name = html.escape(summary.sequence_name)

    bars = (
        _bar_html("out_of_bounds", "bar-oob", oob, total_deaths)
        + _bar_html("max_displacement", "bar-maxd", maxd, total_deaths)
        + _bar_html("invalid_flow", "bar-inv", inv, total_deaths)
    )

    return (
        f"<h2>{seq_name}</h2>"
        f'<p class="stats">{summary.frame_count} frames &middot; '
        f"{summary.total_points} seeded &middot; "
        f'{summary.final_alive} alive ({alive_pct:.1f}%)</p>'
        f"{bars}"
        f'<img class="gif" src="{gif_name}" alt="{seq_name} tracking">'
    )


def write_html_report(
    summaries: list[ProcessedSequenceSummary],
    run_meta: RunMeta,
    output_dir: str | os.PathLike[str],
) -> Path:
    """Write a self-contained report.html to output_dir and return its path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_str = html.escape(run_meta.model)
    adapter_str = html.escape(run_meta.adapter)
    device_str = html.escape(run_meta.device)
    ts_str = html.escape(run_meta.timestamp)
    cmd_str = html.escape("python evaluate_viz.py " + " ".join(run_meta.argv))

    meta_rows = (
        f"<tr><th>Model</th><td>{model_str}</td></tr>"
        f"<tr><th>Adapter</th><td>{adapter_str}</td></tr>"
        f"<tr><th>Device</th><td>{device_str}</td></tr>"
        f"<tr><th>Timestamp</th><td>{ts_str}</td></tr>"
        f"<tr><th>Point method</th><td>{html.escape(run_meta.point_mode)}</td></tr>"
        f"<tr><th>Point count</th><td>{run_meta.point_count}</td></tr>"
        f"<tr><th>GIF duration</th><td>{run_meta.duration_ms} ms/frame</td></tr>"
        f"<tr><th>Max displacement</th><td>{'auto (25% diagonal)' if run_meta.max_displacement is None else f'{run_meta.max_displacement:.1f} px'}</td></tr>"
        f"<tr><th>Command</th><td><code>{cmd_str}</code></td></tr>"
    )

    sequence_sections = "\n".join(_sequence_section_html(s) for s in summaries)

    report_html = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>evaluate_viz report</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>evaluate_viz tracking report</h1>\n"
        f'<p class="subtitle">{ts_str}</p>\n'
        "<table>\n"
        f"{meta_rows}\n"
        "</table>\n"
        f"{sequence_sections}\n"
        "</body>\n"
        "</html>\n"
    )

    report_file = output_path / "report.html"
    report_file.write_text(report_html, encoding="utf-8")

    summary = {
        "run_meta": {
            "model": run_meta.model,
            "adapter": run_meta.adapter,
            "device": run_meta.device,
            "timestamp": run_meta.timestamp,
            "point_mode": run_meta.point_mode,
            "point_count": run_meta.point_count,
            "duration_ms": run_meta.duration_ms,
            "max_displacement": run_meta.max_displacement,
        },
        "sequences": [
            {
                "name": s.sequence_name,
                "frame_count": s.frame_count,
                "total_points": s.total_points,
                "final_alive": s.final_alive,
                "death_counts": s.death_counts,
            }
            for s in summaries
        ],
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return report_file
