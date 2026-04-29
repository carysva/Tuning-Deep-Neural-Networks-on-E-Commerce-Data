"""
Group Assignment 2 - Tuning deep neural networks on pricing data.

Overview:
- Uses TensorFlow deep neural networks
- Includes a parameter tuning function across:
  batch size, hidden layers/neurons, activations, optimizers, LR schedule
- Uses the required inputs:
  sku, price, order, duration, category
- Predicts response:
  quantity

"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


DATA_FILE = "pricing (1).csv"
TARGET = "quantity"
CAT_COLS = ["sku", "category"]
NUM_COLS = ["price", "order", "duration"]


@dataclass
class TrialResult:
    config_name: str
    config: dict[str, Any]
    best_epoch: int
    val_r2: float
    val_rmse: float
    val_mae: float
    fit_time_s: float


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def load_data(base_dir: Path) -> pd.DataFrame:
    path = base_dir / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Could not find '{DATA_FILE}' in {base_dir}")

    df = pd.read_csv(path)
    needed = CAT_COLS + NUM_COLS + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only required columns
    df = df[needed].copy()

    # Ensure types
    for c in CAT_COLS:
        df[c] = df[c].astype(np.int64)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    # Train/val/test = 80/10/10
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_features_targets(df: pd.DataFrame):
    X_cat = df[CAT_COLS].to_numpy(dtype=np.int64)
    X_num = df[NUM_COLS].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X_cat, X_num, y


def build_model(
    cfg: dict[str, Any],
    n_sku: int,
    n_category: int,
    num_mean: np.ndarray,
    num_var: np.ndarray,
) -> tf.keras.Model:
    # Inputs
    in_sku = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name="sku")
    in_cat = tf.keras.layers.Input(shape=(1,), dtype=tf.int64, name="category")
    in_num = tf.keras.layers.Input(shape=(len(NUM_COLS),), dtype=tf.float32, name="num")

    # Embeddings for categorical features
    sku_emb_dim = cfg.get("sku_emb_dim", 16)
    cat_emb_dim = cfg.get("cat_emb_dim", 4)

    sku_emb = tf.keras.layers.Embedding(input_dim=n_sku + 1, output_dim=sku_emb_dim, name="sku_emb")(in_sku)
    sku_flat = tf.keras.layers.Flatten()(sku_emb)

    cat_emb = tf.keras.layers.Embedding(input_dim=n_category + 1, output_dim=cat_emb_dim, name="cat_emb")(in_cat)
    cat_flat = tf.keras.layers.Flatten()(cat_emb)

    # Numeric normalization using train stats
    norm = tf.keras.layers.Normalization(name="num_norm")
    norm.build((None, len(NUM_COLS)))
    norm.set_weights([num_mean.astype(np.float32), num_var.astype(np.float32), np.array(0, dtype=np.int64)])
    num_norm = norm(in_num)

    x = tf.keras.layers.Concatenate(name="feature_concat")([sku_flat, cat_flat, num_norm])

    # Hidden stack
    hidden_units = cfg["hidden_units"]  # e.g. [256, 128, 64]
    activation_name = cfg["activation"]  # relu, elu, tanh, sigmoid, leaky_relu, prelu
    dropout_rate = cfg.get("dropout", 0.0)
    l2_reg = cfg.get("l2", 0.0)

    kernel_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    for i, u in enumerate(hidden_units):
        x = tf.keras.layers.Dense(u, kernel_regularizer=kernel_reg, name=f"dense_{i+1}")(x)

        if activation_name == "leaky_relu":
            x = tf.keras.layers.LeakyReLU(alpha=0.1, name=f"act_{i+1}_lrelu")(x)
        elif activation_name == "prelu":
            x = tf.keras.layers.PReLU(name=f"act_{i+1}_prelu")(x)
        else:
            x = tf.keras.layers.Activation(activation_name, name=f"act_{i+1}_{activation_name}")(x)

        if cfg.get("batch_norm", False):
            x = tf.keras.layers.BatchNormalization(name=f"bn_{i+1}")(x)

        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f"drop_{i+1}")(x)

    # Predict log1p(quantity), then invert externally for metrics
    out = tf.keras.layers.Dense(1, activation="linear", name="out")(x)
    model = tf.keras.Model(inputs=[in_sku, in_cat, in_num], outputs=out, name=cfg["name"])

    # Optimizer choices
    lr = cfg.get("lr", 1e-3)
    if cfg.get("lr_schedule", False):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=2000,
            decay_rate=0.96,
            staircase=True,
        )

    opt_name = cfg["optimizer"]
    if opt_name == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "rmsprop":
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif opt_name == "adagrad":
        opt = tf.keras.optimizers.Adagrad(learning_rate=lr)
    elif opt_name == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif opt_name == "sgd_momentum":
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif opt_name == "sgd_nesterov":
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    model.compile(loss="mse", optimizer=opt)
    return model


def to_dataset(X_cat, X_num, y_log, batch_size: int, training: bool):
    x_dict = {
        "sku": X_cat[:, 0],
        "category": X_cat[:, 1],
        "num": X_num,
    }
    ds = tf.data.Dataset.from_tensor_slices((x_dict, y_log))
    if training:
        ds = ds.shuffle(200_000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_on_original_scale(model, X_cat, X_num, y_true):
    x_dict = {"sku": X_cat[:, 0], "category": X_cat[:, 1], "num": X_num}
    pred_log = model.predict(x_dict, batch_size=4096, verbose=0).reshape(-1)
    pred = np.expm1(pred_log)
    pred = np.clip(pred, a_min=0, a_max=None)

    r2 = r2_score(y_true, pred)
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    mae = float(np.mean(np.abs(y_true - pred)))
    return r2, rmse, mae


def tune_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    tune_rows: int = 220_000,
    max_epochs: int = 20,
) -> tuple[dict[str, Any], list[TrialResult]]:
    # Subsample for fast tuning
    tune_train = train_df.sample(n=min(tune_rows, len(train_df)), random_state=SEED).reset_index(drop=True)
    tune_val = val_df.sample(n=min(max(50_000, tune_rows // 4), len(val_df)), random_state=SEED).reset_index(drop=True)

    X_cat_tr, X_num_tr, y_tr = make_features_targets(tune_train)
    X_cat_va, X_num_va, y_va = make_features_targets(tune_val)

    # Train on log target for stability
    y_tr_log = np.log1p(y_tr).astype(np.float32)
    y_va_log = np.log1p(y_va).astype(np.float32)

    num_mean = X_num_tr.mean(axis=0)
    num_var = X_num_tr.var(axis=0)

    n_sku = int(max(train_df["sku"].max(), val_df["sku"].max()))
    n_category = int(max(train_df["category"].max(), val_df["category"].max()))

    # Candidate tuning grid (covers assignment dimensions)
    configs: list[dict[str, Any]] = [
        {
            "name": "relu_adam_baseline",
            "hidden_units": [256, 128, 64],
            "activation": "relu",
            "optimizer": "adam",
            "lr": 1e-3,
            "batch_size": 1024,
            "dropout": 0.10,
            "batch_norm": True,
            "l2": 1e-6,
        },
        {
            "name": "elu_adam_schedule",
            "hidden_units": [384, 192, 96, 48],
            "activation": "elu",
            "optimizer": "adam",
            "lr": 8e-4,
            "lr_schedule": True,
            "batch_size": 1024,
            "dropout": 0.10,
            "batch_norm": True,
            "l2": 1e-6,
        },
        {
            "name": "prelu_rmsprop",
            "hidden_units": [384, 256, 128, 64],
            "activation": "prelu",
            "optimizer": "rmsprop",
            "lr": 8e-4,
            "batch_size": 2048,
            "dropout": 0.15,
            "batch_norm": True,
            "l2": 5e-6,
        },
        {
            "name": "lrelu_nesterov",
            "hidden_units": [256, 128, 64, 32],
            "activation": "leaky_relu",
            "optimizer": "sgd_nesterov",
            "lr": 3e-3,
            "batch_size": 2048,
            "dropout": 0.05,
            "batch_norm": True,
            "l2": 1e-5,
        },
        {
            "name": "tanh_adagrad",
            "hidden_units": [256, 128, 64],
            "activation": "tanh",
            "optimizer": "adagrad",
            "lr": 1.5e-2,
            "batch_size": 1024,
            "dropout": 0.0,
            "batch_norm": False,
            "l2": 0.0,
        },
        {
            "name": "sigmoid_rmsprop",
            "hidden_units": [128, 64, 32],
            "activation": "sigmoid",
            "optimizer": "rmsprop",
            "lr": 1e-3,
            "batch_size": 1024,
            "dropout": 0.0,
            "batch_norm": False,
            "l2": 0.0,
        },
    ]

    results: list[TrialResult] = []
    best_cfg: Optional[dict[str, Any]] = None
    best_val_r2 = -np.inf

    for cfg in configs:
        print(f"\n=== Tuning trial: {cfg['name']} ===")
        tf.keras.backend.clear_session()

        model = build_model(cfg, n_sku, n_category, num_mean, num_var)
        train_ds = to_dataset(X_cat_tr, X_num_tr, y_tr_log, cfg["batch_size"], training=True)
        val_ds = to_dataset(X_cat_va, X_num_va, y_va_log, cfg["batch_size"], training=False)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=3,
                restore_best_weights=True,
                verbose=0,
            )
        ]

        t0 = time.time()
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max_epochs,
            verbose=0,
            callbacks=callbacks,
        )
        fit_time_s = time.time() - t0
        best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)

        val_r2, val_rmse, val_mae = evaluate_on_original_scale(model, X_cat_va, X_num_va, y_va)
        print(
            f"val_r2={val_r2:.6f} | val_rmse={val_rmse:.4f} | "
            f"val_mae={val_mae:.4f} | best_epoch={best_epoch} | fit_s={fit_time_s:.1f}"
        )

        results.append(
            TrialResult(
                config_name=cfg["name"],
                config=cfg,
                best_epoch=best_epoch,
                val_r2=float(val_r2),
                val_rmse=float(val_rmse),
                val_mae=float(val_mae),
                fit_time_s=float(fit_time_s),
            )
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_cfg = cfg

    assert best_cfg is not None
    return best_cfg, results


def train_best_full(
    best_cfg: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    X_cat_tr, X_num_tr, y_tr = make_features_targets(train_df)
    X_cat_va, X_num_va, y_va = make_features_targets(val_df)
    X_cat_te, X_num_te, y_te = make_features_targets(test_df)

    y_tr_log = np.log1p(y_tr).astype(np.float32)
    y_va_log = np.log1p(y_va).astype(np.float32)

    num_mean = X_num_tr.mean(axis=0)
    num_var = X_num_tr.var(axis=0)

    n_sku = int(max(train_df["sku"].max(), val_df["sku"].max(), test_df["sku"].max()))
    n_category = int(max(train_df["category"].max(), val_df["category"].max(), test_df["category"].max()))

    tf.keras.backend.clear_session()
    model = build_model(best_cfg, n_sku, n_category, num_mean, num_var)

    train_ds = to_dataset(X_cat_tr, X_num_tr, y_tr_log, best_cfg["batch_size"], training=True)
    val_ds = to_dataset(X_cat_va, X_num_va, y_va_log, best_cfg["batch_size"], training=False)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=40,
        verbose=1,
        callbacks=callbacks,
    )

    tr_r2, tr_rmse, tr_mae = evaluate_on_original_scale(model, X_cat_tr, X_num_tr, y_tr)
    va_r2, va_rmse, va_mae = evaluate_on_original_scale(model, X_cat_va, X_num_va, y_va)
    te_r2, te_rmse, te_mae = evaluate_on_original_scale(model, X_cat_te, X_num_te, y_te)

    metrics = {
        "train": {"r2": tr_r2, "rmse": tr_rmse, "mae": tr_mae},
        "val": {"r2": va_r2, "rmse": va_rmse, "mae": va_mae},
        "test": {"r2": te_r2, "rmse": te_rmse, "mae": te_mae},
        "best_epoch": int(np.argmin(hist.history["val_loss"]) + 1),
    }
    return model, metrics


def main():
    base_dir = get_base_dir()
    out_dir = base_dir

    print("Loading data...")
    df = load_data(base_dir)
    train_df, val_df, test_df = split_data(df)
    print(f"Rows: total={len(df):,} train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    print("\nTuning candidate DNN configurations...")
    best_cfg, trials = tune_models(train_df, val_df, tune_rows=220_000, max_epochs=20)
    print(f"\nBest config from tuning: {best_cfg['name']}")

    print("\nTraining best config on full train set...")
    model, metrics = train_best_full(best_cfg, train_df, val_df, test_df)

    # Save model
    model_path = out_dir / "ga2_best_pricing_dnn.keras"
    model.save(model_path)

    # Save detailed trial table
    trial_rows = []
    for t in trials:
        row = {
            "config_name": t.config_name,
            "best_epoch": t.best_epoch,
            "val_r2": t.val_r2,
            "val_rmse": t.val_rmse,
            "val_mae": t.val_mae,
            "fit_time_s": t.fit_time_s,
            "config_json": json.dumps(t.config),
        }
        trial_rows.append(row)
    trial_df = pd.DataFrame(trial_rows).sort_values("val_r2", ascending=False)
    trial_csv = out_dir / "ga2_tuning_results.csv"
    trial_df.to_csv(trial_csv, index=False)

    # Save summary report
    report_path = out_dir / "ga2_model_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Group Assignment 2 - Pricing model report\n")
        f.write("=========================================\n\n")
        f.write(f"Data file: {DATA_FILE}\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}\n\n")

        f.write("Best tuned config:\n")
        f.write(json.dumps(best_cfg, indent=2))
        f.write("\n\n")

        f.write("Final metrics (original quantity scale):\n")
        f.write(json.dumps(metrics, indent=2))
        f.write("\n\n")
        f.write(f"Saved model: {model_path.name}\n")
        f.write(f"Saved trials: {trial_csv.name}\n")

    print("\n=== FINAL METRICS (original quantity scale) ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model -> {model_path}")
    print(f"Saved tuning table -> {trial_csv}")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()

