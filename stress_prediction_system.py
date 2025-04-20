
from __future__ import annotations

import argparse, logging, math, joblib, os, warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataLoader:
    """Loads CSV data and offers basic summary."""
    def __init__(self, path: str):
        self.path = path
        self.df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        logging.info("Reading %s", self.path)
        self.df = pd.read_csv(self.path)
        logging.info("Loaded %d rows, %d columns", *self.df.shape)
        return self.df

    def summary(self):
        if self.df is None:
            raise ValueError("Call load() first")
        logging.info("Missing values per column:\n%s", self.df.isnull().sum())

class FeatureCreator(BaseEstimator, TransformerMixin):
    """Creates extra numeric ratios and interaction terms."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xc = X.copy()
        # Derived ratios / interactions
        if {"Sleep Duration (Hours per night)", "Study Hours Per Week"}.issubset(Xc.columns):
            Xc["Sleep_to_Study"] = Xc["Sleep Duration (Hours per night)"] / Xc["Study Hours Per Week"].replace(0, np.nan)
            Xc["Sleep_to_Study"].fillna(0, inplace=True)
        if {"Social Media Usage (Hours per day)", "Sleep Duration (Hours per night)"}.issubset(Xc.columns):
            Xc["SM_to_Sleep"] = Xc["Social Media Usage (Hours per day)"] / Xc["Sleep Duration (Hours per night)"].replace(0, np.nan)
            Xc["SM_to_Sleep"].fillna(0, inplace=True)
        if {"Physical Exercise (Hours per week)", "Age"}.issubset(Xc.columns):
            Xc["Exercise_per_Age"] = Xc["Physical Exercise (Hours per week)"] / Xc["Age"]
        # Combine GPA and study hours
        if {"Academic Performance (GPA)", "Study Hours Per Week"}.issubset(Xc.columns):
            Xc["GPA_x_Study"] = Xc["Academic Performance (GPA)"] * Xc["Study Hours Per Week"]
        return Xc

class Preprocessor:
    """ColumnTransformer with KNNImputer, scaling, encoding."""
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df
        self.target = target
        self.numeric: List[str] = []
        self.categorical: List[str] = []
        self.pipeline: ColumnTransformer | None = None

    def _detect(self):
        for col in self.df.columns:
            if col == self.target:
                continue
            (self.categorical if self.df[col].dtype == 'object' else self.numeric).append(col)

    def build(self):
        self._detect()
        num_pipe = Pipeline([("imputer", KNNImputer(3)), ("scaler", StandardScaler())])
        cat_pipe = Pipeline([("encoder", OneHotEncoder(handle_unknown='ignore'))])
        self.pipeline = ColumnTransformer([("num", num_pipe, self.numeric),
                                           ("cat", cat_pipe, self.categorical)])
        return self.pipeline

class ModelTrainer:
    """StackingRegressor (RF + GBR -> RidgeCV)."""
    def __init__(self, preproc: ColumnTransformer, **rf_params):
        self.preproc = preproc
        self.model: Pipeline | None = None
        self.rf_params = rf_params

    def build(self):
        base = [
            ("rf", RandomForestRegressor(random_state=42, **self.rf_params)),
            ("gbr", GradientBoostingRegressor(random_state=42))
        ]
        stack = StackingRegressor(estimators=base, final_estimator=RidgeCV())
        self.model = Pipeline([
            ("create", FeatureCreator()),
            ("preproc", self.preproc),
            ("stack", stack)
        ])
        return self.model

    def train(self, X, y):
        if self.model is None:
            self.build()
        self.model.fit(X, y)
        return self.model

    def evaluate(self, X, y) -> Dict[str, float]:
        p = self.model.predict(X)
        return {
            "MAE": mean_absolute_error(y, p),
            "RMSE": math.sqrt(mean_squared_error(y, p)),
            "R2": r2_score(y, p)
        }

    def save(self, path: str):
        joblib.dump(self.model, path)
        logging.info("Saved model to %s", path)

class RecommendationEngine:
    """Rule‑based tips from raw record."""
    def __init__(self, model: Pipeline, target: str):
        self.model = model
        self.target = target

    def _tips(self, r):
        tips = []
        if r["Sleep Duration (Hours per night)"] < 7: tips.append("Increase sleep to 7–8 h")
        if r["Physical Exercise (Hours per week)"] < 3: tips.append("Add weekly exercise")
        if r["Social Media Usage (Hours per day)"] > 4: tips.append("Limit social media")
        if r["Family Support  "] < 3: tips.append("Lean on support network")
        return tips

    def recommend(self, X: pd.DataFrame):
        preds = self.model.predict(X)
        out = X.copy()
        out["Predicted Stress"] = preds
        out["Recommendations"] = out.apply(self._tips, axis=1)
        return out[["Predicted Stress", "Recommendations"]]

class FeatureImportance:
    """Permutation importance helper."""
    def __init__(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, top_k: int = 20):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.top_k = top_k
        self.importances_: pd.DataFrame | None = None

    def compute(self):
        result = permutation_importance(self.model, self.X_test, self.y_test,
                                        n_repeats=5, random_state=42, n_jobs=-1)
        imp = pd.DataFrame({"feature": self.X_test.columns,
                            "importance": result.importances_mean})
        self.importances_ = imp.sort_values("importance", ascending=False).head(self.top_k)
        return self.importances_

    def print(self):
        if self.importances_ is None:
            raise ValueError("Run compute first.")
        print("\nTop feature importances:")
        for _, row in self.importances_.iterrows():
            print(f"{row.feature:<40}{row.importance:>10.4f}")

def kfold_cv(df: pd.DataFrame, target: str, folds: int = 5) -> Dict[str, float]:
    idx = np.arange(len(df))
    np.random.seed(42)
    np.random.shuffle(idx)
    splits = np.array_split(idx, folds)
    metrics = {"MAE": [], "RMSE": [], "R2": []}
    for i in range(folds):
        test_idx = splits[i]
        train_idx = np.hstack([splits[j] for j in range(folds) if j != i])
        tr, te = df.iloc[train_idx], df.iloc[test_idx]
        prep = Preprocessor(tr, target).build()
        trainer = ModelTrainer(prep, n_estimators=200)
        trainer.train(tr.drop(columns=[target]), tr[target])
        m = trainer.evaluate(te.drop(columns=[target]), te[target])
        for k, v in m.items():
            metrics[k].append(v)
        logging.info("Fold %d: %s", i+1, m)
    return {k: float(np.mean(v)) for k, v in metrics.items()}

def split_df(df: pd.DataFrame, target: str, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=[target]); y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed)


class HyperparameterSearch:
    """Performs exhaustive grid search for RandomForest parameters.

    Usage:
        hs = HyperparameterSearch()
        best_params, report = hs.search(X_train, y_train)
    """
    def __init__(self):
        self.param_grid = {
            "rf__n_estimators": [100, 200, 400],
            "rf__max_depth": [None, 10, 20],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [1, 2]
        }
        self.best_params_: Dict[str, Any] | None = None
        self.cv_results_: pd.DataFrame | None = None

    def _build_pipe(self):
        # Minimal pipeline for speed
        return Pipeline([
            ("rf", RandomForestRegressor(random_state=42))
        ])

    def search(self, X, y, cv=3):
        pipe = self._build_pipe()
        gs = GridSearchCV(pipe, self.param_grid, cv=cv,
                          scoring="neg_root_mean_squared_error",
                          n_jobs=-1, verbose=1)
        gs.fit(X, y)
        self.best_params_ = gs.best_params_
        self.cv_results_ = pd.DataFrame(gs.cv_results_)
        return self.best_params_, self.cv_results_

    def save_results(self, path: str):
        if self.cv_results_ is None:
            raise ValueError("Run search() first.")
        self.cv_results_.to_csv(path, index=False)
        logging.info("Grid search results saved to %s", path)

class SHAPPlotter:
    """Generates SHAP value summary plots if `shap` is installed.

    The class gracefully degrades when `shap` is unavailable, so the core
    pipeline remains dependency‑light.
    """
    def __init__(self, model: Pipeline, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        try:
            import shap
            self.shap = shap
            self.explainer = shap.Explainer(self.model.predict, feature_names=self.feature_names)
        except ImportError:
            self.shap = None
            logging.warning("Optional dependency 'shap' not installed; SHAP plots disabled.")

    def summary_plot(self, X_sample: pd.DataFrame):
        if self.shap is None:
            print("Install shap to enable SHAP summary plots.")
            return
        values = self.explainer(X_sample)
        self.shap.summary_plot(values, X_sample, show=True)

def main():
    ap = argparse.ArgumentParser(description="Stress prediction system")
    ap.add_argument("--data", required=True, help="CSV path")
    ap.add_argument("--target", default="Mental Stress Level")
    ap.add_argument("--cv", type=int, help="Run k‑fold CV with given folds")
    ap.add_argument("--save_model")
    ap.add_argument("--save_importances")
    args = ap.parse_args()

    df = DataLoader(args.data).load()

    # Optional CV
    if args.cv:
        cv_res = kfold_cv(df, args.target, args.cv)
        print(f"\n{args.cv}-fold CV average:", cv_res)

    # Standard train/test
    prep = Preprocessor(df, args.target).build()
    trainer = ModelTrainer(prep, n_estimators=300, max_depth=None)
    X_train, X_test, y_train, y_test = split_df(df, args.target)
    trainer.train(X_train, y_train)
    eval_metrics = trainer.evaluate(X_test, y_test)
    print("\nHold‑out metrics:", eval_metrics)

    if args.save_model:
        trainer.save(args.save_model)

    # Feature importance
    fi = FeatureImportance(trainer.model, X_test, y_test)
    fi.compute()
    fi.print()
    if args.save_importances:
        fi.importances_.to_csv(args.save_importances, index=False)

    # Sample recommendations
    rec_engine = RecommendationEngine(trainer.model, args.target)
    print("\nRecommendations for 5 samples:")
    print(rec_engine.recommend(X_test.head(5)).to_string(index=False))

if __name__ == '__main__':
    main()
