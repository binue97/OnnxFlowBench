"""View evaluation results from one or more result directories.

Usages:

    # Open a single result report
    python view_eval_results.py results/dis

    # Compare two results side by side
    python view_eval_results.py results/dis results/ofnet

    # Compare up to three results side by side
    python view_eval_results.py results/dis results/ofnet results/raft
"""

from __future__ import annotations

import base64
import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_DIRS = 3

_CSS = """
body { font-family: system-ui, sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; background: #0f0f0f; color: #e0e0e0; }
h1 { font-size: 1.4rem; margin-bottom: 0.25rem; color: #fff; }
h2 { font-size: 1.1rem; margin: 2rem 0 0.5rem; color: #ccc; border-bottom: 1px solid #333; padding-bottom: 0.25rem; }
.subtitle { color: #666; font-size: 0.8rem; margin-top: 0.1rem; margin-bottom: 1.5rem; }
.run-headers { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.run-header { flex: 1; background: #1a1a1a; border-radius: 4px; padding: 0.75rem 1rem; border: 1px solid #2a2a2a; }
.run-header h3 { font-size: 0.95rem; margin: 0 0 0.25rem; color: #ddd; }
.run-header .path { font-size: 0.75rem; color: #666; word-break: break-all; }
.seq-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.seq-cell { flex: 1; }
.seq-cell img { display: block; width: 100%; border: 1px solid #333; border-radius: 4px; }
.seq-cell .label { font-size: 0.75rem; color: #888; margin-bottom: 0.35rem; }
.missing { background: #1a1a1a; border-radius: 4px; height: 80px; display: flex; align-items: center; justify-content: center; color: #555; font-size: 0.8rem; border: 1px solid #222; }
.run-meta { margin-top: 0.5rem; font-size: 0.72rem; color: #777; }
.run-meta div { margin: 0.1rem 0; }
.run-meta .key { color: #555; margin-right: 0.3rem; }
.seq-stats { font-size: 0.75rem; color: #999; margin: 0.4rem 0 0.3rem; }
.bar-row { display: flex; align-items: center; gap: 0.4rem; margin: 0.15rem 0; font-size: 0.7rem; }
.bar-label { width: 110px; color: #888; flex-shrink: 0; }
.bar-track { flex: 1; background: #1e1e1e; border-radius: 2px; height: 9px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 2px; }
.bar-count { width: 34px; text-align: right; color: #666; flex-shrink: 0; }
.bar-oob  { background: #ff4040; }
.bar-maxd { background: #ffa020; }
.bar-inv  { background: #a0a0a0; }
"""


def _gif_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/gif;base64,{data}"


def _find_gifs(directory: Path) -> dict[str, Path]:
    """Return {sequence_name: gif_path} for all GIFs in directory."""
    return {p.stem: p for p in sorted(directory.glob("*.gif"))}


def _load_summary(directory: Path) -> dict | None:
    """Load summary.json from a result directory, or None if absent."""
    path = directory / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_meta_html(meta: dict) -> str:
    max_disp = meta.get("max_displacement")
    disp_str = "auto (25% diagonal)" if max_disp is None else f"{max_disp:.1f} px"
    rows = [
        ("model", meta.get("model", "—")),
        ("adapter", meta.get("adapter", "—")),
        ("device", meta.get("device", "—")),
        ("timestamp", meta.get("timestamp", "—")),
        ("points", f"{meta.get('point_count', '—')} ({meta.get('point_mode', '—')})"),
        ("max disp", disp_str),
        ("gif speed", f"{meta.get('duration_ms', '—')} ms/frame"),
    ]
    inner = "".join(
        f'<div><span class="key">{html.escape(k)}:</span>{html.escape(str(v))}</div>'
        for k, v in rows
    )
    return f'<div class="run-meta">{inner}</div>'


def _bar_html(label: str, css_class: str, count: int, total: int) -> str:
    pct = (count / total * 100) if total > 0 else 0.0
    return (
        f'<div class="bar-row">'
        f'<div class="bar-label">{html.escape(label)}</div>'
        f'<div class="bar-track"><div class="bar-fill {css_class}" style="width:{pct:.1f}%"></div></div>'
        f'<div class="bar-count">{count}</div>'
        f"</div>"
    )


def _seq_stats_html(seq: dict) -> str:
    total = seq.get("total_points", 0)
    alive = seq.get("final_alive", 0)
    frames = seq.get("frame_count", 0)
    pct = alive / total * 100 if total > 0 else 0.0
    dc = seq.get("death_counts", {})
    total_deaths = total - alive
    stats = f'<div class="seq-stats">{frames} frames &middot; {total} seeded &middot; {alive} alive ({pct:.1f}%)</div>'
    bars = (
        _bar_html("out_of_bounds", "bar-oob", dc.get("out_of_bounds", 0), total_deaths)
        + _bar_html("max_displacement", "bar-maxd", dc.get("max_displacement", 0), total_deaths)
        + _bar_html("invalid_flow", "bar-inv", dc.get("invalid_flow", 0), total_deaths)
    )
    return stats + bars


def _open_with_xdg(path: str) -> None:
    subprocess.run(["xdg-open", path], check=False)


def _open_single(result_dir: Path) -> None:
    report = result_dir / "report.html"
    if not report.exists():
        sys.exit(f"error: no report.html found in {result_dir}")
    _open_with_xdg(str(report))


def _build_compare_html(dirs: list[Path]) -> str:
    labels = [html.escape(d.name) for d in dirs]
    gif_maps = [_find_gifs(d) for d in dirs]
    summaries = [_load_summary(d) for d in dirs]

    all_names = sorted(set().union(*[m.keys() for m in gif_maps]))

    run_headers_html = ""
    for label, d, summary in zip(labels, dirs, summaries):
        meta_html = ""
        if summary and "run_meta" in summary:
            meta_html = _run_meta_html(summary["run_meta"])
        run_headers_html += (
            f'<div class="run-header">'
            f"<h3>{label}</h3>"
            f'<div class="path">{html.escape(str(d.resolve()))}</div>'
            f"{meta_html}"
            f"</div>"
        )

    sequences_html = ""
    for name in all_names:
        sequences_html += f"<h2>{html.escape(name)}</h2>\n"
        sequences_html += '<div class="seq-row">\n'
        for label, gif_map, summary in zip(labels, gif_maps, summaries):
            seq_data = None
            if summary:
                seq_data = next(
                    (s for s in summary.get("sequences", []) if s["name"] == name),
                    None,
                )
            sequences_html += '<div class="seq-cell">\n'
            sequences_html += f'<div class="label">{label}</div>\n'
            if name in gif_map:
                uri = _gif_data_uri(gif_map[name])
                sequences_html += (
                    f'<img src="{uri}" alt="{html.escape(name)} — {label}">\n'
                )
            else:
                sequences_html += '<div class="missing">not found</div>\n'
            if seq_data:
                sequences_html += _seq_stats_html(seq_data) + "\n"
            sequences_html += "</div>\n"
        sequences_html += "</div>\n"

    runs_label = " vs ".join(labels)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>compare: {runs_label}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>Comparison: {runs_label}</h1>\n"
        f'<p class="subtitle">{len(all_names)} sequence(s) &middot; {len(dirs)} runs</p>\n'
        f'<div class="run-headers">{run_headers_html}</div>\n'
        f"{sequences_html}"
        "</body>\n"
        "</html>\n"
    )


def _open_compare(dirs: list[Path]) -> None:
    compare_html = _build_compare_html(dirs)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(compare_html)
        tmp_path = f.name
    _open_with_xdg(tmp_path)
    print(f"Compare report: {tmp_path}")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    if not args:
        print(
            "usage: python view_eval_results.py <result_dir> [<result_dir2> [<result_dir3>]]",
            file=sys.stderr,
        )
        return 1

    if len(args) > MAX_DIRS:
        print(
            f"error: maximum supported comparison is {MAX_DIRS} result directories, got {len(args)}",
            file=sys.stderr,
        )
        return 1

    dirs: list[Path] = []
    for arg in args:
        p = Path(arg).expanduser().resolve()
        if not p.is_dir():
            print(f"error: not a directory: {p}", file=sys.stderr)
            return 1
        dirs.append(p)

    if len(dirs) == 1:
        _open_single(dirs[0])
    else:
        _open_compare(dirs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
