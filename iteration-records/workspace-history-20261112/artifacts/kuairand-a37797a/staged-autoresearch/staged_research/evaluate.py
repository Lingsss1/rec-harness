"""One entry: preflight -> immutable registration -> resumable matrix -> report."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from .engine import Trajectory
from .local import gpu_lease, make_runtime
from .report import report
from .storage import ProtocolError, locked, now, read, write
from .study import check_config, freeze, load_registration, matrix


def evaluate_study(root, output, *, config_path=None, check_only=False, keep_going=False,
                   runtime_factory=make_runtime, progress=print):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or output in root.parents or root in output.parents:
        raise ProtocolError("Report output must be disjoint from the study")
    with locked(root / ".evaluation.lock"):
        draft = root / "study.draft.json"
        registered = root / "registration.json"
        supplied = read(config_path) if config_path is not None else None
        if registered.exists():
            registration = load_registration(root)
            config = registration["config"]
        elif draft.exists():
            registration = None
            config = read(draft)
        elif supplied is not None:
            registration = None
            config = supplied
            write(draft, config, exclusive=True)
        else:
            raise ProtocolError("Provide --config or initialize study.draft.json first")
        if supplied is not None and supplied != config:
            raise ProtocolError("Config differs from this study; use a new root, never overwrite a registered experiment")
        errors = check_config(config, root, check_images=True)
        summary = {"ready": not errors, "errors": errors, "backend": config.get("execution", {"backend": "docker"}),
                   "trajectories": 60, "validation_calls": 6000, "check_only": check_only}
        if errors or check_only:
            return summary
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProtocolError("Export ANTHROPIC_API_KEY before evaluation; --check-only does not require a key")
        if registration is None:
            registration = freeze(root)
        prior_report = output / "summary.json"
        if prior_report.exists() and read(prior_report).get("registration_id") != registration["registration_id"]:
            raise ProtocolError("Output belongs to a different registration; use another report directory")
        runtime = runtime_factory(config, root)
        # Coordinate this runner across studies on this host as well as within one
        # study. This lock does not control unrelated training programs/schedulers.
        failures = []
        with gpu_lease(config):
            try:
                for index, cell in enumerate(matrix(config), 1):
                    label = "%s/%s/%d" % (cell["task"], cell["arm"], cell["replicate"])
                    progress("[%d/60] %s" % (index, label))
                    try:
                        state = Trajectory(root, registration, cell["task"], cell["arm"], cell["replicate"], runtime).run()
                        if state.get("test", {}).get("status") != "completed":
                            raise ProtocolError("Trajectory ended without a completed hidden test")
                    except Exception as exc:
                        failures.append({"cell": label, "error": str(exc), "at": now()})
                        write(root / "execution_failures" / (uuid.uuid4().hex + ".json"), failures[-1], exclusive=True)
                        progress("FAILED " + label + ": " + str(exc))
                        if not keep_going:
                            raise
            finally:
                # Export even on interruption; partial data remain explicitly partial.
                result = report(root, registration, output)
        summary.update(registration_id=registration["registration_id"],
                       completed_trajectories=result["completed_trajectories"], failures=failures,
                       output=str(output))
        return summary
