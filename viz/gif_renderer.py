"""GIF frame rendering for flow-driven point tracking visualization."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Canonical color map: None = alive trail, key = death reason color.
# Used for both trail lines and the dot color at the point of death.
DEATH_REASON_COLORS: dict[str | None, tuple[int, int, int]] = {
    None: (255, 196, 64),           # yellow — alive trail
    "out_of_bounds": (255, 64, 64),    # red
    "max_displacement": (255, 160, 32), # orange
    "invalid_flow": (160, 160, 160),   # grey
}

_ALIVE_DOT_COLOR: tuple[int, int, int] = (80, 255, 96)   # green — currently alive dot


def _draw_tracked_points(
    frame: np.ndarray,
    tracked_points: list[np.ndarray],
    alive_masks: list[np.ndarray],
    death_masks: list[np.ndarray],
    death_reasons: list[str | None],
    frame_index: int,
    trail_length: int,
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    start_index = max(0, frame_index - trail_length + 1)

    current_points = np.asarray(tracked_points[frame_index], dtype=np.float32)
    current_alive = np.asarray(alive_masks[frame_index], dtype=bool)
    current_dead = np.asarray(death_masks[frame_index], dtype=bool)

    history_arrays = [
        np.asarray(tracked_points[i], dtype=np.float32)
        for i in range(start_index, frame_index + 1)
    ]

    for point_index in range(len(current_points)):
        # Trail color uses the *final* death reason even for pre-death frames — by design.
        reason = death_reasons[point_index]
        trail_color = DEATH_REASON_COLORS.get(reason, DEATH_REASON_COLORS[None])

        trail = []
        for history_index in range(start_index, frame_index + 1):
            history_points = history_arrays[history_index - start_index]
            if point_index >= len(history_points):
                continue
            x_coord, y_coord = history_points[point_index]
            if not np.isfinite(x_coord) or not np.isfinite(y_coord):
                continue
            trail.append((float(x_coord), float(y_coord)))

        if len(trail) >= 2:
            draw.line(trail, fill=trail_color, width=1)

        x_coord, y_coord = current_points[point_index]
        if not np.isfinite(x_coord) or not np.isfinite(y_coord):
            continue

        radius = 2
        if current_dead[point_index]:
            # Died this step — use the reason color for the dot
            color = DEATH_REASON_COLORS.get(reason, DEATH_REASON_COLORS[None])
        elif current_alive[point_index]:
            color = _ALIVE_DOT_COLOR
        else:
            # Previously dead — keep showing at last position with reason color
            color = DEATH_REASON_COLORS.get(reason, DEATH_REASON_COLORS[None])

        draw.ellipse(
            (
                float(x_coord) - radius,
                float(y_coord) - radius,
                float(x_coord) + radius,
                float(y_coord) + radius,
            ),
            fill=color,
        )

    return image


def write_tracking_gif(
    frames: list[np.ndarray],
    tracked_points: list[np.ndarray],
    alive_masks: list[np.ndarray],
    death_masks: list[np.ndarray],
    death_reasons: list[str | None],
    output_path: str | os.PathLike[str],
    trail_length: int = 8,
    duration_ms: int = 100,
) -> None:
    """Render tracked points on frames and write an animated GIF."""
    if not frames:
        raise ValueError("frames must not be empty")
    if trail_length <= 0:
        raise ValueError("trail_length must be positive")
    if duration_ms <= 0:
        raise ValueError("duration_ms must be positive")

    count = len(frames)
    if not (len(tracked_points) == len(alive_masks) == len(death_masks) == count):
        raise ValueError("frames, tracked_points, alive_masks, and death_masks must match length")
    if len(death_reasons) != len(tracked_points[0]):
        raise ValueError("death_reasons length must match number of tracked points")

    rendered_frames = [
        _draw_tracked_points(
            frame=frames[index],
            tracked_points=tracked_points,
            alive_masks=alive_masks,
            death_masks=death_masks,
            death_reasons=death_reasons,
            frame_index=index,
            trail_length=trail_length,
        )
        for index in range(count)
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_frames[0].save(
        output,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=duration_ms,
        loop=0,
    )
