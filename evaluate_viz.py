"""Visualize feature tracking over image sequences without ground truth.

Usage:
    # Quick test with DIS optical flow (no ONNX model needed)
    python evaluate_viz.py --adapter dis --input resources/test_sequences --output results/dis

    # Run with an ONNX model
    python evaluate_viz.py --model resources/models/ofnet_v1.onnx --adapter ofnet --input resources/test_sequences --output results/ofnet

		# Visualize result (View single evaluation report)
		python view_eval_results.py results/ofnet/ 

		# Visualize result (Compare multiple evaluation reports)
		python view_eval_results.py results/dis/ results/raft/ results/ofnet/
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from utils.frame_utils import read_gen
from viz.gif_renderer import write_tracking_gif
from viz.html_report import write_html_report


IMAGE_EXTENSIONS = {
	".bmp",
	".jpeg",
	".jpg",
	".png",
	".tif",
	".tiff",
	".webp",
}


@dataclass(frozen=True)
class SequenceInfo:
	"""One ordered image sequence ready for evaluation."""

	name: str
	image_paths: list[Path]


@dataclass(frozen=True)
class TrackingSequenceResult:
	"""Tracked point state across a sequence."""

	tracked_points: list[np.ndarray]
	alive_masks: list[np.ndarray]
	death_masks: list[np.ndarray]
	death_reasons: list[str | None]
	death_counts: dict[str, int]


@dataclass(frozen=True)
class ProcessedSequenceSummary:
	"""High-level summary for one processed sequence."""

	sequence_name: str
	frame_count: int
	total_points: int
	final_alive: int
	death_counts: dict[str, int]
	output_path: Path


@dataclass(frozen=True)
class RunMeta:
	"""Metadata about a CLI evaluation run."""

	model: str
	adapter: str
	device: str
	timestamp: str            # formatted datetime
	point_mode: str           # fast | hom
	point_count: int          # number of seeded points
	duration_ms: int          # GIF frame delay in ms
	max_displacement: float | None  # None = auto (25% of frame diagonal)
	argv: tuple[str, ...]     # sys.argv[1:] for reproducibility


def _is_image_file(path: Path) -> bool:
	return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _sorted_image_paths(directory: Path) -> list[Path]:
	return sorted(path for path in directory.iterdir() if _is_image_file(path))


def discover_sequences(root: str | os.PathLike[str]) -> list[SequenceInfo]:
	"""Discover ordered frame sequences under a root directory."""
	root_path = Path(root).expanduser().resolve()

	if not root_path.exists():
		raise FileNotFoundError(f"Sequence root does not exist: {root_path}")
	if not root_path.is_dir():
		raise NotADirectoryError(f"Sequence root is not a directory: {root_path}")

	sequences: list[SequenceInfo] = []
	child_dirs = sorted(path for path in root_path.iterdir() if path.is_dir())

	for child_dir in child_dirs:
		image_paths = _sorted_image_paths(child_dir)
		if len(image_paths) >= 2:
			sequences.append(SequenceInfo(name=child_dir.name, image_paths=image_paths))

	if sequences:
		return sequences

	root_images = _sorted_image_paths(root_path)
	if len(root_images) >= 2:
		return [SequenceInfo(name=root_path.name, image_paths=root_images)]

	raise ValueError(f"No image sequences found under: {root_path}")


def default_max_displacement(width: int, height: int, ratio: float = 0.25) -> float:
	"""Scale the displacement threshold with the image diagonal."""
	if width <= 0 or height <= 0:
		raise ValueError("width and height must be positive")
	if ratio <= 0:
		raise ValueError("ratio must be positive")
	return math.hypot(width, height) * ratio


def compute_death_reasons(
	previous_points: np.ndarray,
	next_points: np.ndarray,
	sampled_flow: np.ndarray,
	width: int,
	height: int,
	max_displacement: float,
) -> list[str | None]:
	"""Return per-point death reasons for the proposed update step.

	Points are expected in ``(x, y)`` order.
	"""
	if width <= 0 or height <= 0:
		raise ValueError("width and height must be positive")
	if max_displacement <= 0:
		raise ValueError("max_displacement must be positive")

	previous = np.asarray(previous_points, dtype=np.float32)
	proposed = np.asarray(next_points, dtype=np.float32)
	flow = np.asarray(sampled_flow, dtype=np.float32)

	if previous.shape != proposed.shape or previous.shape != flow.shape:
		raise ValueError("previous_points, next_points, and sampled_flow must share shape")
	if previous.ndim != 2 or previous.shape[1] != 2:
		raise ValueError("point arrays must have shape (N, 2)")

	reasons: list[str | None] = []
	for previous_point, next_point, flow_vector in zip(previous, proposed, flow, strict=True):
		if not np.all(np.isfinite(flow_vector)) or not np.all(np.isfinite(next_point)):
			reasons.append("invalid_flow")
			continue

		next_x, next_y = float(next_point[0]), float(next_point[1])
		if next_x < 0 or next_x > width - 1 or next_y < 0 or next_y > height - 1:
			reasons.append("out_of_bounds")
			continue

		displacement = float(np.linalg.norm(next_point - previous_point))
		if displacement > max_displacement:
			reasons.append("max_displacement")
			continue

		reasons.append(None)

	return reasons


def sample_flow_at_points(flow: np.ndarray, points: np.ndarray) -> np.ndarray:
	"""Sample dense flow at floating-point ``(x, y)`` locations.

	Points outside the image receive ``NaN`` flow values.
	"""
	flow_array = np.asarray(flow, dtype=np.float32)
	point_array = np.asarray(points, dtype=np.float32)

	if flow_array.ndim != 3 or flow_array.shape[2] != 2:
		raise ValueError("flow must have shape (H, W, 2)")
	if point_array.ndim != 2 or point_array.shape[1] != 2:
		raise ValueError("points must have shape (N, 2)")

	height, width = flow_array.shape[:2]
	sampled = np.full((len(point_array), 2), np.nan, dtype=np.float32)

	for index, (x_coord, y_coord) in enumerate(point_array):
		if not np.isfinite(x_coord) or not np.isfinite(y_coord):
			continue
		if x_coord < 0 or x_coord > width - 1 or y_coord < 0 or y_coord > height - 1:
			continue

		x0 = int(np.floor(x_coord))
		y0 = int(np.floor(y_coord))
		x1 = min(x0 + 1, width - 1)
		y1 = min(y0 + 1, height - 1)
		wx = float(x_coord - x0)
		wy = float(y_coord - y0)

		top = (1.0 - wx) * flow_array[y0, x0] + wx * flow_array[y0, x1]
		bottom = (1.0 - wx) * flow_array[y1, x0] + wx * flow_array[y1, x1]
		sampled[index] = (1.0 - wy) * top + wy * bottom

	return sampled


def propagate_points(previous_points: np.ndarray, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Advect points by sampling the dense flow field at their locations."""
	sampled_flow = sample_flow_at_points(flow, previous_points)
	next_points = np.asarray(previous_points, dtype=np.float32) + sampled_flow
	return sampled_flow, next_points


def track_sequence(
	model,
	frames: list[np.ndarray],
	initial_points: np.ndarray,
	max_displacement: float,
) -> TrackingSequenceResult:
	"""Track one set of points through a sequence using optical flow predictions."""
	if len(frames) < 2:
		raise ValueError("frames must contain at least two images")

	points = np.asarray(initial_points, dtype=np.float32)
	if points.ndim != 2 or points.shape[1] != 2:
		raise ValueError("initial_points must have shape (N, 2)")

	height, width = np.asarray(frames[0]).shape[:2]
	current_points = points.copy()
	current_alive = np.ones(len(points), dtype=bool)
	final_reasons: list[str | None] = [None] * len(points)
	death_counts = {
		"invalid_flow": 0,
		"out_of_bounds": 0,
		"max_displacement": 0,
	}

	tracked_points = [current_points.copy()]
	alive_masks = [current_alive.copy()]
	death_masks = [np.zeros(len(points), dtype=bool)]

	for frame_index in range(len(frames) - 1):
		flow = model.predict(frames[frame_index], frames[frame_index + 1])
		sampled_flow, proposed_points = propagate_points(current_points, flow)
		step_reasons = compute_death_reasons(
			previous_points=current_points,
			next_points=proposed_points,
			sampled_flow=sampled_flow,
			width=width,
			height=height,
			max_displacement=max_displacement,
		)

		next_points = current_points.copy()
		next_alive = current_alive.copy()
		step_dead = np.zeros(len(points), dtype=bool)

		for point_index, reason in enumerate(step_reasons):
			if not current_alive[point_index]:
				continue

			if reason is None:
				next_points[point_index] = proposed_points[point_index]
				continue

			next_alive[point_index] = False
			step_dead[point_index] = True
			final_reasons[point_index] = reason
			death_counts[reason] += 1

		current_points = next_points
		current_alive = next_alive
		tracked_points.append(current_points.copy())
		alive_masks.append(current_alive.copy())
		death_masks.append(step_dead)

	return TrackingSequenceResult(
		tracked_points=tracked_points,
		alive_masks=alive_masks,
		death_masks=death_masks,
		death_reasons=final_reasons,
		death_counts=death_counts,
	)


def load_rgb_image(path: str | os.PathLike[str]) -> np.ndarray:
	"""Load an image as ``(H, W, 3)`` RGB ``uint8``."""
	image = read_gen(str(path))
	if hasattr(image, "convert"):
		rgb_image = image.convert("RGB")
		return np.asarray(rgb_image, dtype=np.uint8)

	array = np.asarray(image, dtype=np.uint8)
	if array.ndim == 2:
		return np.repeat(array[..., None], 3, axis=2)
	return array[..., :3]


def load_sequence_frames(sequence: SequenceInfo) -> list[np.ndarray]:
	"""Load all frames for a discovered sequence."""
	return [load_rgb_image(path) for path in sequence.image_paths]


def initialize_points(
	frame: np.ndarray,
	point_mode: str,
	point_count: int,
	fast_threshold: int,
) -> np.ndarray:
	"""Seed points on the first frame and return them in ``(x, y)`` order."""
	import cv2

	height, width = frame.shape[:2]

	if point_mode == "fast":
		gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		detector = cv2.FastFeatureDetector_create(threshold=fast_threshold)
		keypoints = detector.detect(gray)
		if not keypoints:
			raise RuntimeError("FAST detected zero keypoints; try lowering --fast-threshold")
		keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
		points = np.array([kp.pt for kp in keypoints[:point_count]], dtype=np.float32)
	elif point_mode == "hom":
		aspect = width / height
		rows = max(1, int(np.sqrt(point_count / aspect)))
		cols = max(1, int(point_count / rows))
		ys = np.linspace(0, height - 1, rows + 2)[1:-1]
		xs = np.linspace(0, width - 1, cols + 2)[1:-1]
		yy, xx = np.meshgrid(ys, xs, indexing="ij")
		pts_yx = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
		if len(pts_yx) > point_count:
			idx = np.random.default_rng(42).choice(len(pts_yx), point_count, replace=False)
			pts_yx = pts_yx[idx]
		points = pts_yx[:, ::-1]  # (y, x) -> (x, y)
	else:
		raise ValueError(f"Unknown point_mode: {point_mode!r}")

	return points


def process_sequence(
	model,
	sequence: SequenceInfo,
	output_dir: str | os.PathLike[str],
	initial_points: np.ndarray | None = None,
	max_displacement: float | None = None,
	trail_length: int = 8,
	duration_ms: int = 200,
	point_mode: str = "fast",
	point_count: int = 200,
	fast_threshold: int = 20,
) -> ProcessedSequenceSummary:
	"""Run tracking and GIF rendering for one sequence."""
	frames = load_sequence_frames(sequence)
	if len(frames) < 2:
		raise ValueError(f"Sequence {sequence.name!r} must contain at least two frames")

	if initial_points is None:
		points = initialize_points(
			frame=frames[0],
			point_mode=point_mode,
			point_count=point_count,
			fast_threshold=fast_threshold,
		)
	else:
		points = np.asarray(initial_points, dtype=np.float32)

	height, width = frames[0].shape[:2]
	displacement_threshold = (
		max_displacement
		if max_displacement is not None
		else default_max_displacement(width=width, height=height)
	)

	tracking = track_sequence(
		model=model,
		frames=frames,
		initial_points=points,
		max_displacement=displacement_threshold,
	)

	output_path = Path(output_dir) / f"{sequence.name}.gif"
	write_tracking_gif(
		frames=frames,
		tracked_points=tracking.tracked_points,
		alive_masks=tracking.alive_masks,
		death_masks=tracking.death_masks,
		death_reasons=tracking.death_reasons,
		output_path=output_path,
		trail_length=trail_length,
		duration_ms=duration_ms,
	)

	return ProcessedSequenceSummary(
		sequence_name=sequence.name,
		frame_count=len(frames),
		total_points=len(points),
		final_alive=int(tracking.alive_masks[-1].sum()),
		death_counts=dict(tracking.death_counts),
		output_path=output_path,
	)





def default_sequence_root() -> Path:
	"""Return the default bundled sequence root."""
	return Path(__file__).resolve().parent / "resources" / "test_sequences"


def positive_int(value: str) -> int:
	parsed = int(value)
	if parsed <= 0:
		raise argparse.ArgumentTypeError("must be a positive integer")
	return parsed


def positive_float(value: str) -> float:
	parsed = float(value)
	if parsed <= 0:
		raise argparse.ArgumentTypeError("must be a positive number")
	return parsed


def build_argparser() -> argparse.ArgumentParser:
	"""Build the evaluator CLI parser."""
	from core.registry import list_adapters

	parser = argparse.ArgumentParser(
		description="Visualize flow-driven feature tracking on image sequences without ground truth"
	)
	parser.add_argument("--model", default=None, help="Path to the .onnx model")
	parser.add_argument(
		"--adapter",
		default="raft",
		help=f"Adapter name ({', '.join(list_adapters())})",
	)
	parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

	parser.add_argument(
		"--input",
		default=str(default_sequence_root()),
		help="Sequence root directory. Child folders are treated as separate sequences.",
	)
	parser.add_argument(
		"--output",
		default="results/evaluate_viz",
		help="Directory for output GIF files",
	)
	parser.add_argument(
		"--sequence",
		action="append",
		help="Specific sequence name to process. Repeat the flag to select multiple sequences.",
	)
	parser.add_argument("--point-mode", choices=["fast", "hom"], default="fast")
	parser.add_argument("--point-count", type=positive_int, default=200)
	parser.add_argument("--fast-threshold", type=positive_int, default=20)
	parser.add_argument("--max-displacement", type=positive_float, default=None)
	parser.add_argument("--trail-length", type=positive_int, default=8)
	parser.add_argument("--duration-ms", type=positive_int, default=200)
	return parser


def _select_sequences(
	sequences: list[SequenceInfo],
	selected_names: list[str] | None,
) -> list[SequenceInfo]:
	if not selected_names:
		return sequences

	wanted = set(selected_names)
	filtered = [sequence for sequence in sequences if sequence.name in wanted]
	missing = sorted(wanted - {sequence.name for sequence in filtered})
	if missing:
		raise ValueError(f"Unknown sequence names: {', '.join(missing)}")
	return filtered


def _format_death_counts(death_counts: dict[str, int]) -> str:
	return ", ".join(f"{name}={count}" for name, count in death_counts.items())


def run_cli(args: argparse.Namespace) -> list[ProcessedSequenceSummary]:
	"""Execute the evaluator from parsed CLI arguments."""
	sequences = _select_sequences(discover_sequences(args.input), args.sequence)

	from core.registry import get_adapter

	adapter = get_adapter(args.adapter)
	if args.model is None:
		model = adapter
	else:
		from core.flow_model import FlowModel
		model = FlowModel(args.model, adapter=adapter, device=args.device)

	summaries: list[ProcessedSequenceSummary] = []
	for sequence in sequences:
		summary = process_sequence(
			model=model,
			sequence=sequence,
			output_dir=args.output,
			max_displacement=args.max_displacement,
			trail_length=args.trail_length,
			duration_ms=args.duration_ms,
			point_mode=args.point_mode,
			point_count=args.point_count,
			fast_threshold=args.fast_threshold,
		)
		summaries.append(summary)

		print(f"Sequence: {summary.sequence_name}")
		print(f"  Frames: {summary.frame_count}")
		print(f"  Seeded points: {summary.total_points}")
		print(f"  Final alive: {summary.final_alive}")
		print(f"  Death counts: {_format_death_counts(summary.death_counts)}")
		print(f"  Output: {summary.output_path}")

	run_meta = RunMeta(
		model=args.model or args.adapter,
		adapter=args.adapter,
		device=args.device if args.model else "cpu",
		timestamp=datetime.now().strftime("%Y-%m-%d, %H:%M:%S"),
		point_mode=args.point_mode,
		point_count=args.point_count,
		duration_ms=args.duration_ms,
		max_displacement=args.max_displacement,
		argv=tuple(sys.argv[1:]),
	)
	report_path = write_html_report(summaries, run_meta, args.output)
	print(f"Report: {report_path}")

	return summaries


def main(argv: list[str] | None = None) -> int:
	parser = build_argparser()
	args = parser.parse_args(argv)
	run_cli(args)
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
