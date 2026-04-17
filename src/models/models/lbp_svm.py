from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def detect_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / ".git").exists() or ((p / "features").exists() and (p / "src").exists()):
            return p
    return start


def load_xy(pkl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing features file: {pkl_path}")
    with open(pkl_path, "rb") as f:
        X, y = pickle.load(f)
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    return X, y


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_splits(features_dir: Path) -> SplitData:
    X_train, y_train = load_xy(features_dir / "train_lbp_features.pkl")
    X_valid, y_valid = load_xy(features_dir / "valid_lbp_features.pkl")
    X_test, y_test = load_xy(features_dir / "test_lbp_features.pkl")
    return SplitData(X_train, y_train, X_valid, y_valid, X_test, y_test)


def build_search(random_state: int, scoring: str) -> GridSearchCV:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    param_grid = [
        {
            "svm__kernel": ["rbf"],
            "svm__C": [0.1, 1, 3, 10, 30, 100],
            "svm__gamma": ["scale", 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        },
        {
            "svm__kernel": ["linear"],
            "svm__C": [0.1, 1, 3, 10, 30, 100],
        },
        {
            "svm__kernel": ["poly"],
            "svm__C": [0.1, 1, 3, 10],
            "svm__gamma": ["scale", 1e-3, 1e-2],
            "svm__degree": [2, 3],
            "svm__coef0": [0.0, 0.5, 1.0],
        },
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    return GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )


def format_eval(name: str, model, X: np.ndarray, y: np.ndarray) -> str:
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred)
    cm = confusion_matrix(y, pred)
    report = classification_report(y, pred, target_names=["real", "fake"])
    return (
        f"== {name} ==\n"
        f"accuracy: {acc}\n"
        f"f1: {f1}\n"
        f"confusion_matrix:\n{cm}\n"
        f"report:\n{report}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/tune SVM on LBP features and save best model.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to repository root (auto-detected if omitted).",
    )
    parser.add_argument(
        "--scoring",
        default="accuracy",
        choices=["accuracy", "f1", "f1_macro", "f1_weighted"],
        help="GridSearch scoring metric.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--save-model",
        default="models/svm_lbp_model.pkl",
        help="Where to save the trained model pickle (relative to repo root by default).",
    )
    parser.add_argument(
        "--save-report",
        default="reports/svm_lbp_results.txt",
        help="Where to save the text metrics report (relative to repo root by default).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else detect_repo_root(Path.cwd())
    features_dir = repo_root / "features"
    model_path = (repo_root / args.save_model).resolve() if not Path(args.save_model).is_absolute() else Path(args.save_model)
    report_path = (repo_root / args.save_report).resolve() if not Path(args.save_report).is_absolute() else Path(args.save_report)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    splits = load_splits(features_dir)

    print("Shapes:")
    print("  train:", splits.X_train.shape, splits.y_train.shape)
    print("  valid:", splits.X_valid.shape, splits.y_valid.shape)
    print("  test :", splits.X_test.shape, splits.y_test.shape)

    search = build_search(random_state=args.random_state, scoring=args.scoring)
    search.fit(splits.X_train, splits.y_train)

    lines: list[str] = []
    lines.append(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"repo_root: {repo_root}")
    lines.append(f"scoring: {args.scoring}")
    lines.append(f"best_cv_score: {search.best_score_}")
    lines.append("best_params:")
    for k, v in search.best_params_.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    best_model = search.best_estimator_
    lines.append(format_eval("VALID", best_model, splits.X_valid, splits.y_valid))
    lines.append(format_eval("TEST", best_model, splits.X_test, splits.y_test))

    report_text = "\n".join(lines).strip() + "\n"
    print("\n" + report_text)

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Saved model:  {model_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

