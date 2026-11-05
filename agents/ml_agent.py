
import io
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


RANDOM_STATE = 42
DEFAULT_TRAINING_ROWS = 30000
FEATURE_SAMPLE_SIZE = 20000


def get_feature_suggestions(df, target, sample_size=FEATURE_SAMPLE_SIZE):
    suggested, excluded = [], {}
    sample = df.sample(n=min(len(df), sample_size), random_state=RANDOM_STATE) if len(df) > sample_size else df

    for col in df.columns:
        if col == target:
            continue

        series = sample[col]
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / max(series.notna().sum(), 1)
        name = str(col).lower()

        if df[col].isna().all():
            excluded[col] = "All values are missing"
        elif unique_count <= 1:
            excluded[col] = "Constant in inspected sample"
        elif any(term in name for term in ("email", "password", "phone", "mobile", "address")):
            excluded[col] = "Potentially sensitive information"
        elif name.endswith("_id") or name == "id" or name.endswith(" identifier"):
            excluded[col] = "Identifier column"
        elif unique_count > 100 and unique_ratio > 0.05:
            excluded[col] = "High-cardinality column"
        else:
            suggested.append(col)

    return suggested, excluded


def prepare_features(X):
    X = X.copy()
    numeric, categorical = [], []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            numeric.append(col)
        else:
            converted = pd.to_numeric(X[col], errors="coerce")
            ratio = converted.notna().sum() / max(X[col].notna().sum(), 1)

            if ratio >= 0.95:
                X[col] = converted.replace([np.inf, -np.inf], np.nan)
                numeric.append(col)
            else:
                X[col] = X[col].astype("string").fillna("Missing").astype(str)
                categorical.append(col)

    return X, numeric, categorical


def remove_unusable_features(X):
    all_missing = X.columns[X.isna().all()].tolist()
    constant = [
        col for col in X.columns
        if col not in all_missing and X[col].nunique(dropna=True) <= 1
    ]
    removed = all_missing + constant
    return X.drop(columns=removed), removed


def build_pipeline(numeric, categorical, estimator):
    transformers = []

    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))

    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
        ("model", estimator)
    ])


def get_models(task):
    if task == "Regression":
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(
                max_depth=12, min_samples_leaf=2, random_state=RANDOM_STATE
            ),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=60, max_depth=12, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=2
            )
        }

    return {
        "Logistic Regression": LogisticRegression(
            solver="saga", max_iter=300, tol=1e-3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=60, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=2
        )
    }


def sample_training_data(X, y, training_rows, task):
    if training_rows >= len(X):
        return X, y

    stratify = None
    if task == "Classification":
        counts = y.value_counts()
        if counts.min() >= 2 and training_rows >= len(counts):
            stratify = y

    try:
        indices, _ = train_test_split(
            X.index, train_size=training_rows,
            random_state=RANDOM_STATE, stratify=stratify
        )
    except ValueError:
        indices = X.sample(n=training_rows, random_state=RANDOM_STATE).index

    return X.loc[indices], y.loc[indices]


def split_data(X, y, task, test_size):
    stratify = None

    if task == "Classification":
        counts = pd.Series(y).value_counts()
        test_count = int(np.ceil(len(y) * test_size))
        train_count = len(y) - test_count

        if counts.min() >= 2 and test_count >= len(counts) and train_count >= len(counts):
            stratify = y

    return train_test_split(
        X, y, test_size=test_size,
        random_state=RANDOM_STATE, stratify=stratify
    )


def prepare_training_data(df, feature_columns, y, valid_indices, training_rows, task):
    X = df.loc[valid_indices, feature_columns]
    X, y = sample_training_data(X, y, training_rows, task)
    X, numeric, categorical = prepare_features(X)
    X, removed = remove_unusable_features(X)

    if X.empty:
        raise ValueError("No usable input features remain after removing missing and constant columns.")

    # Keep transformer column lists synchronized with the retained DataFrame.
    numeric = [col for col in numeric if col in X.columns]
    categorical = [col for col in categorical if col in X.columns]

    encoder = None
    if task == "Classification":
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)

        if y.nunique() < 2:
            raise ValueError("The selected sample contains fewer than two target classes.")
    else:
        y = pd.to_numeric(y, errors="coerce").astype(float)

    return X, y, numeric, categorical, encoder, removed


def show_ml():
    st.header("🤖 Machine Learning")
    st.caption("Configure your prediction, train models and understand their results.")

    df = st.session_state.get("cleaned_df")

    if df is None:
        st.info("Upload a dataset first.")
        return

    if df.empty or len(df.columns) < 2:
        st.warning("At least two columns and valid records are required.")
        return

    st.subheader("01 · Define Your Prediction")
    st.markdown(
        "**Target (y):** The output the model must learn to predict.  \n"
        "**Features (X):** The input columns used to predict that output."
    )

    target = st.selectbox("Select Target Column (Output)", df.columns.tolist(), key="ml_target")
    suggested, excluded = get_feature_suggestions(df, target)

    feature_columns = st.multiselect(
        "Select Feature Columns (Inputs)",
        [col for col in df.columns if col != target],
        default=suggested,
        help="Potential identifiers, sensitive fields, constant columns and high-cardinality columns are excluded from the default selection. You can manually select other columns.",
        key="ml_features"
    )

    if excluded:
        with st.expander(f"Why were {len(excluded)} columns excluded by default?"):
            st.dataframe(
                pd.DataFrame([
                    {"Column": col, "Reason": reason}
                    for col, reason in excluded.items()
                ]),
                use_container_width=True, hide_index=True
            )

    if not feature_columns:
        st.warning("Select at least one input feature.")
        return

    y = df[target].copy()
    numeric_target = pd.api.types.is_numeric_dtype(y)

    if not numeric_target:
        converted = pd.to_numeric(y, errors="coerce")
        ratio = converted.notna().sum() / max(y.notna().sum(), 1)

        if ratio >= 0.95:
            y, numeric_target = converted, True

    if numeric_target:
        suggested_task = "Classification" if 2 <= y.nunique(dropna=True) <= 10 else "Regression"
        task = st.radio(
            "Prediction Type",
            ["Classification", "Regression"],
            index=["Classification", "Regression"].index(suggested_task),
            horizontal=True,
            help="Classification predicts categories. Regression predicts numerical values.",
            key="ml_task"
        )
    else:
        task = "Classification"
        st.info("Categorical target detected. Classification is selected automatically.")

    if numeric_target:
        y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)

    valid_indices = y.index[y.notna()]
    y = y.loc[valid_indices]

    if len(y) < 10:
        st.warning("At least 10 valid target records are recommended.")
        return

    if task == "Classification":
        counts = y.value_counts()

        if len(counts) < 2:
            st.warning("Classification requires at least two target classes.")
            return

        if counts.min() < 2:
            st.warning("Every target class needs at least two records.")
            return
    elif not numeric_target:
        st.warning("Regression requires a numerical target column.")
        return
    elif y.nunique() < 2:
        st.warning("Regression requires at least two distinct target values.")
        return

    st.divider()
    st.subheader("02 · Understand Your Data")
    st.caption(
        f"Target: {target} · Task: {task} · "
        f"Rows excluded due to missing target: {len(df) - len(y):,}"
    )

    st.divider()
    st.subheader("03 · Configure Training")

    models = get_models(task)
    model_name = st.selectbox("Select Algorithm", list(models), key="ml_model")
    test_size = st.slider("Test data percentage", 0.1, 0.5, 0.2, 0.05, key="ml_test_size")

    max_rows = len(y)
    training_rows = st.slider(
        "Maximum records used",
        min_value=min(1000, max_rows),
        max_value=max_rows,
        value=min(DEFAULT_TRAINING_ROWS, max_rows),
        step=max(1, min(5000, max_rows // 20)),
        key="ml_training_rows"
    )

    st.caption(
        f"Training and evaluation will use a reproducible sample of "
        f"{training_rows:,} out of {len(y):,} valid records."
    )

    if training_rows < max_rows:
        st.info("A representative sample is used for training. The original dataset remains unchanged.")

    with st.expander("Preview input features and target"):
        preview_indices = y.head(10).index
        st.markdown("**Input data (X)**")
        st.dataframe(df.loc[preview_indices, feature_columns], use_container_width=True)
        st.markdown(f"**Expected output (y): {target}**")
        st.dataframe(y.head(10).rename(target).to_frame(), use_container_width=True)

    if task == "Classification":
        with st.expander("Target class distribution"):
            st.dataframe(
                y.value_counts().rename_axis("Class").reset_index(name="Records"),
                use_container_width=True, hide_index=True
            )

    st.divider()
    st.subheader("04 · Compare Algorithms")

    if st.button("Compare All Models", use_container_width=True, key="compare_models"):
        try:
            with st.spinner("Preparing training sample..."):
                X_model, y_model, numeric, categorical, encoder, removed = prepare_training_data(
                    df, feature_columns, y, valid_indices, training_rows, task
                )

            if removed:
                st.warning("Removed unusable features: " + ", ".join(map(str, removed)))

            X_train, X_test, y_train, y_test = split_data(
                X_model, y_model, task, test_size
            )

            results, failures = [], []
            progress, status = st.progress(0), st.empty()

            for i, (name, estimator) in enumerate(models.items()):
                start = time.perf_counter()
                status.info(f"Training {name} ({i + 1}/{len(models)})...")

                try:
                    pipeline = build_pipeline(numeric, categorical, estimator)
                    pipeline.fit(X_train, y_train)
                    predictions = pipeline.predict(X_test)
                    elapsed = round(time.perf_counter() - start, 2)

                    if task == "Regression":
                        results.append({
                            "Model": name,
                            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                            "MAE": mean_absolute_error(y_test, predictions),
                            "R²": r2_score(y_test, predictions) if len(y_test) > 1 else np.nan,
                            "Training Time (s)": elapsed
                        })
                    else:
                        results.append({
                            "Model": name,
                            "Accuracy": accuracy_score(y_test, predictions),
                            "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
                            "Macro F1": f1_score(
                                y_test, predictions, average="macro", zero_division=0
                            ),
                            "Training Time (s)": elapsed
                        })
                except Exception as error:
                    failures.append(f"{name}: {error}")

                progress.progress((i + 1) / len(models))

            status.empty()
            progress.empty()

            if results:
                result_df = pd.DataFrame(results)
                st.dataframe(result_df.round(4), use_container_width=True, hide_index=True)

                if task == "Regression":
                    best = result_df.loc[result_df["RMSE"].idxmin(), "Model"]
                    st.success(f"Best result by RMSE: {best}")
                else:
                    best = result_df.loc[result_df["Macro F1"].idxmax(), "Model"]
                    st.success(f"Best result by Macro F1: {best}")
            else:
                st.error("No model could be trained. Check the selected data.")

            for failure in failures:
                st.warning(failure)

        except Exception as error:
            st.error(f"Model comparison failed: {error}")

    st.divider()
    st.subheader("05 · Train Selected Algorithm")

    if st.button("Train Model", type="primary", use_container_width=True, key="train_model"):
        try:
            with st.spinner("Preparing training data..."):
                X_model, y_model, numeric, categorical, encoder, removed = prepare_training_data(
                    df, feature_columns, y, valid_indices, training_rows, task
                )

            if removed:
                st.warning("Removed unusable features: " + ", ".join(map(str, removed)))

            X_train, X_test, y_train, y_test = split_data(
                X_model, y_model, task, test_size
            )

            model = build_pipeline(numeric, categorical, models[model_name])
            progress, status = st.progress(0), st.empty()
            status.info(f"Training {model_name}...")

            start = time.perf_counter()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            elapsed = time.perf_counter() - start

            progress.progress(1.0)
            status.empty()
            progress.empty()

            st.success(f"{model_name} trained successfully in {elapsed:.2f} seconds.")

            metric_cols = st.columns(4)
            metric_cols[0].metric("Training Records", f"{len(X_train):,}")
            metric_cols[1].metric("Testing Records", f"{len(X_test):,}")

            if task == "Regression":
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions) if len(y_test) > 1 else np.nan

                metric_cols[2].metric("RMSE", f"{rmse:.4f}")
                metric_cols[3].metric("R² Score", f"{r2:.4f}" if np.isfinite(r2) else "N/A")
                st.caption(f"MAE: {mae:.4f}. Lower RMSE and MAE are generally better.")

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.scatter(y_test, predictions, alpha=0.4, s=12)
                low = min(y_test.min(), predictions.min())
                high = max(y_test.max(), predictions.max())
                ax.plot([low, high], [low, high], "--", color="red")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            else:
                accuracy = accuracy_score(y_test, predictions)
                balanced = balanced_accuracy_score(y_test, predictions)
                precision = precision_score(
                    y_test, predictions, average="macro", zero_division=0
                )
                recall = recall_score(
                    y_test, predictions, average="macro", zero_division=0
                )
                f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

                metric_cols[2].metric("Accuracy", f"{accuracy:.4f}")
                metric_cols[3].metric("Macro F1", f"{f1:.4f}")
                st.caption(
                    f"Balanced accuracy: {balanced:.4f} · "
                    f"Macro precision: {precision:.4f} · Macro recall: {recall:.4f}"
                )

                labels = np.arange(len(encoder.classes_))
                report = classification_report(
                    y_test, predictions, labels=labels,
                    target_names=encoder.classes_.astype(str),
                    output_dict=True, zero_division=0
                )

                with st.expander("Detailed classification report"):
                    st.dataframe(
                        pd.DataFrame(report).transpose().round(4),
                        use_container_width=True
                    )

                cm = confusion_matrix(y_test, predictions, labels=labels)
                fig, ax = plt.subplots(figsize=(7, 5))
                sns.heatmap(
                    cm, annot=len(labels) <= 12, fmt="d", cmap="Blues",
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.divider()
            st.subheader("06 · Understand Feature Importance")
            estimator = model.named_steps["model"]

            if hasattr(estimator, "feature_importances_"):
                names = model.named_steps["preprocessor"].get_feature_names_out()
                importance = pd.DataFrame({
                    "Feature": names,
                    "Importance": estimator.feature_importances_
                }).sort_values("Importance", ascending=False)

                st.dataframe(
                    importance.head(20).round(5),
                    use_container_width=True, hide_index=True
                )

                fig, ax = plt.subplots(figsize=(7, 4))
                top = importance.head(15).sort_values("Importance")
                ax.barh(top["Feature"], top["Importance"])
                ax.set_title("Top 15 Feature Importances")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption("Feature importance measures model reliance, not causation.")
            else:
                st.info("This algorithm does not provide built-in feature importance.")

            st.divider()
            st.subheader("07 · Download Model and Predictions")

            model_buffer = io.BytesIO()
            joblib.dump({
                "pipeline": model,
                "target_encoder": encoder,
                "target_column": target,
                "feature_columns": X_model.columns.tolist(),
                "problem_type": task
            }, model_buffer)

            st.download_button(
                "Download Trained Model",
                data=model_buffer.getvalue(),
                file_name="trained_model.pkl",
                mime="application/octet-stream",
                use_container_width=True,
                key="download_trained_model"
            )

            if task == "Classification":
                actual = encoder.inverse_transform(np.asarray(y_test, dtype=int))
                predicted = encoder.inverse_transform(np.asarray(predictions, dtype=int))
            else:
                actual, predicted = y_test.to_numpy(), predictions

            output = pd.DataFrame(
                {"Actual": actual, "Predicted": predicted},
                index=X_test.index
            )

            with st.expander("Preview predictions"):
                st.dataframe(output.head(20), use_container_width=True)

            st.download_button(
                "Download Predictions (CSV)",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_predictions"
            )

        except Exception as error:
            st.error(f"Model training failed: {error}")

    st.divider()
    st.caption(
        "Available algorithms: Linear Regression, Logistic Regression, "
        "Decision Tree Regressor, Decision Tree Classifier, "
        "Random Forest Regressor and Random Forest Classifier."
    )
