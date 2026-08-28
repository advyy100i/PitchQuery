"""Phase 5: MLflow, run entirely on this machine.

Backing store is a SQLite file and artefacts are a local folder, which is the
whole configuration. MLflow's hosted product is paid; nothing here touches it,
and nothing here needs a server running either — the SQLite URI works against
the file directly, so a training run logs whether or not `mlflow server` is up.
Start the server when you want to look at the runs.

Two things this module owns beyond "call mlflow.log_metric":

  * The promotion rule. `promote_if_better` reads the current champion's metric
    out of the registry and only moves the alias if the new run beats it. That
    turns "is this model better" from something you remember to check into
    something the training script cannot skip.

  * Degrading. If mlflow is not installed the training script still trains, and
    says it is not tracking. The alternative is a model pipeline that cannot run
    without an experiment tracker, which is backwards.

Run the UI:
  mlflow server --backend-store-uri sqlite:///mlflow.db \\
                --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
"""
import contextlib
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import REPO_ROOT  # noqa: E402

MODEL_NAME = "pitchquery-xg"
RANKER_MODEL_NAME = "pitchquery-ranker"
CHAMPION = "champion"

# sqlite:/// wants a POSIX-ish path even on Windows, and an absolute one so that
# the store does not follow the working directory around.
DEFAULT_URI = "sqlite:///" + (REPO_ROOT / "mlflow.db").as_posix()


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI)


def available() -> bool:
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def git_commit() -> str:
    """The commit the run was trained at, or 'unknown' outside a checkout.

    Worth a tag rather than a note in a filename: two runs with the same metrics
    and different commits is the situation this answers.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=REPO_ROOT, capture_output=True, text=True,
                             timeout=10)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                   capture_output=True, text=True, timeout=10)
            return f"{sha}-dirty" if dirty.stdout.strip() else sha
    except Exception:
        pass
    return "unknown"


@contextlib.contextmanager
def run(experiment: str, run_name: str = None):
    """Yield something with log_param(s)/log_metric(s)/log_artifact/set_tag(s).

    Yields a no-op recorder when mlflow is missing, so call sites do not need to
    branch on whether tracking is on.
    """
    if not available():
        print("mlflow is not installed — training, but not tracking. "
              "pip install mlflow to record this run.")
        yield _NullRun()
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        yield _MlflowRun(mlflow, active)


class _NullRun:
    run_id = None
    tracking = False

    def log_params(self, *_a, **_k): pass
    def log_metrics(self, *_a, **_k): pass
    def set_tags(self, *_a, **_k): pass
    def log_artifact(self, *_a, **_k): pass
    def log_dict(self, *_a, **_k): pass
    def log_model(self, *_a, **_k): return None


class _MlflowRun:
    tracking = True

    def __init__(self, mlflow, active):
        self._mlflow = mlflow
        self.run_id = active.info.run_id

    def log_params(self, params: dict):
        # MLflow stores params as strings and rejects a value over 6000 chars;
        # the feature list goes in as a JSON artefact instead (see log_dict).
        self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict):
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                  if v is not None})

    def set_tags(self, tags: dict):
        self._mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_artifact(self, path):
        p = Path(path)
        if p.exists():
            self._mlflow.log_artifact(str(p))

    def log_dict(self, obj, name: str):
        self._mlflow.log_dict(obj, name)

    def log_model(self, model, artifact_path: str, **kwargs):
        # MLflow 3 serialises sklearn models with skops, which refuses to write
        # a file referencing types it does not recognise — LightGBM's Booster
        # and sklearn's calibration wrappers among them. Naming them is the
        # documented escape hatch, and the list is short because it is the list
        # of things this script just fitted itself. Anything not on it is a
        # model shape that changed, which is worth a failure rather than a
        # blanket allow.
        kwargs.setdefault("skops_trusted_types", TRUSTED_TYPES)
        return self._mlflow.sklearn.log_model(model, name=artifact_path, **kwargs)


TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "lightgbm.sklearn.LGBMRanker",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.isotonic.IsotonicRegression",
    "scipy.interpolate._interpolate.interp1d",
    "numpy.dtype",
]


# --- registry and promotion ---------------------------------------------------

def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    return MlflowClient()


def champion_metric(model_name: str, metric: str):
    """The champion's value for `metric`, or None if there is no champion yet."""
    if not available():
        return None
    from mlflow.exceptions import MlflowException

    client = _client()
    try:
        version = client.get_model_version_by_alias(model_name, CHAMPION)
    except MlflowException:
        return None
    data = client.get_run(version.run_id).data
    return data.metrics.get(metric)


def promote_if_better(model_name: str, run_id: str, artifact_path: str,
                      metric: str, value: float, lower_is_better: bool = True) -> dict:
    """Register this run and move `champion` only if it actually beats the champion.

    The rule is code and not a habit on purpose. A registry where promotion is a
    manual step in a UI is a registry that eventually points at whichever model
    somebody clicked last, and the tie case matters too: an equal score is not an
    improvement, so it does not promote. Returns a dict the caller can print.
    """
    if not available():
        return {"promoted": False, "reason": "mlflow not installed"}

    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    client = _client()
    incumbent = champion_metric(model_name, metric)

    version = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", model_name)
    client.set_model_version_tag(model_name, version.version, metric, f"{value:.6f}")
    client.set_model_version_tag(model_name, version.version, "git_commit", git_commit())

    better = (incumbent is None
              or (value < incumbent if lower_is_better else value > incumbent))
    if better:
        client.set_registered_model_alias(model_name, CHAMPION, version.version)
    return {"promoted": bool(better), "version": version.version,
            "metric": metric, "value": value, "incumbent": incumbent}


def describe_promotion(result: dict) -> str:
    """One line for the training log, saying what happened and why."""
    if not result.get("version"):
        return f"registry: not updated ({result.get('reason', 'unknown')})"
    m, new, old = result["metric"], result["value"], result["incumbent"]
    if result["promoted"]:
        was = "no champion yet" if old is None else f"beats {old:.4f}"
        return (f"registry: version {result['version']} promoted to champion "
                f"({m} {new:.4f}, {was})")
    return (f"registry: version {result['version']} logged but NOT promoted "
            f"({m} {new:.4f} vs champion {old:.4f}) — the champion stands")
