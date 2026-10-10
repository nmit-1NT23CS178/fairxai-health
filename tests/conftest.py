"""Test fixtures. All data here is SYNTHETIC (generated) and isolated in a temp dir - never real patient data."""
import os, sys, tempfile, pathlib, time
TMP = pathlib.Path(tempfile.mkdtemp(prefix="fairxai_test_"))
os.environ.update(FAIRXAI_ROOT=str(TMP), DATA_DIR=str(TMP / "data"), ARTIFACT_DIR=str(TMP / "artifacts"), APP_ENV="development", MONGO_URI="", ADMIN_TOKEN="")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pytest
from sklearn.datasets import make_classification


def synthetic_frame(n=600, seed=0):
    X, y = make_classification(n_samples=n, n_features=5, n_informative=4, n_redundant=0, class_sep=1.0, flip_y=0.05, random_state=seed)
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(X, columns=["glucose", "bmi", "bloodpressure", "marker_a", "marker_b"])
    df["glucose"] = 100 + df["glucose"] * 25; df["bmi"] = 28 + df["bmi"] * 5; df["bloodpressure"] = 75 + df["bloodpressure"] * 10
    df["age"] = rng.randint(25, 80, n); df["sex"] = rng.choice(["F", "M"], n); df["Outcome"] = y
    df.insert(0, "Patient ID", range(1, n + 1))
    return df


@pytest.fixture(scope="session")
def raw_dir():
    from backend.ml.common import RAW_DIR
    return RAW_DIR


@pytest.fixture(scope="session")
def trained_app(raw_dir):
    """Writes a synthetic 'diabetes' dataset, trains it for real, returns the Flask app."""
    d = raw_dir / "diabetes"; d.mkdir(parents=True, exist_ok=True)
    synthetic_frame().to_csv(d / "synthetic_diabetes.csv", index=False)
    from backend.ml.train import train_disease
    train_disease("diabetes", search_iter=2)
    from backend.app import create_app
    app = create_app(); app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(trained_app):
    return trained_app.test_client()
