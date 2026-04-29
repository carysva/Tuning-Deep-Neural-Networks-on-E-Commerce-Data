import time
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout, Flatten
from tensorflow.keras.models import Model

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------
df = pd.read_csv("pricing.csv")

numeric_features = ["price", "order", "duration"]

# Output
y = df["quantity"].values

# Split first
X_train_full, X_test, y_train_full, y_test = train_test_split(
    df[["sku", "category"] + numeric_features],
    y,
    test_size=0.2,
    random_state=42
)

# Validation split from training set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# 2. Scale numeric features
# --------------------------------------------------
scaler = StandardScaler()
X_train.loc[:, numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_val.loc[:, numeric_features] = scaler.transform(X_val[numeric_features])
X_test.loc[:, numeric_features] = scaler.transform(X_test[numeric_features])

# --------------------------------------------------
# 3. Safe embedding vocab sizes
# --------------------------------------------------
sku_vocab_size = df["sku"].max() + 2
category_vocab_size = df["category"].max() + 2

sku_unknown_idx = sku_vocab_size - 1
cat_unknown_idx = category_vocab_size - 1

X_train.loc[:, "sku"] = X_train["sku"].clip(upper=sku_unknown_idx - 1)
X_val.loc[:, "sku"] = np.where(X_val["sku"] < sku_unknown_idx, X_val["sku"], sku_unknown_idx)
X_test.loc[:, "sku"] = np.where(X_test["sku"] < sku_unknown_idx, X_test["sku"], sku_unknown_idx)

X_train.loc[:, "category"] = X_train["category"].clip(upper=cat_unknown_idx - 1)
X_val.loc[:, "category"] = np.where(X_val["category"] < cat_unknown_idx, X_val["category"], cat_unknown_idx)
X_test.loc[:, "category"] = np.where(X_test["category"] < cat_unknown_idx, X_test["category"], cat_unknown_idx)

# --------------------------------------------------
# 4. Build input dictionaries
# --------------------------------------------------
train_inputs = {
    "sku_input": X_train["sku"].values,
    "category_input": X_train["category"].values,
    "numeric_input": X_train[numeric_features].values
}

val_inputs = {
    "sku_input": X_val["sku"].values,
    "category_input": X_val["category"].values,
    "numeric_input": X_val[numeric_features].values
}

test_inputs = {
    "sku_input": X_test["sku"].values,
    "category_input": X_test["category"].values,
    "numeric_input": X_test[numeric_features].values
}

# --------------------------------------------------
# 5. Embedding dimensions
# --------------------------------------------------
sku_emb_dim = min(20, max(1, sku_vocab_size // 100))
cat_emb_dim = min(10, max(1, category_vocab_size // 100))

# --------------------------------------------------
# 6. Function to build model
# --------------------------------------------------
def build_model(hidden_1, hidden_2, activation, dropout_rate, learning_rate):
    sku_input = Input(shape=(1,), name="sku_input")
    category_input = Input(shape=(1,), name="category_input")
    numeric_input = Input(shape=(len(numeric_features),), name="numeric_input")

    sku_emb = Embedding(input_dim=sku_vocab_size, output_dim=sku_emb_dim)(sku_input)
    sku_emb = Flatten()(sku_emb)

    cat_emb = Embedding(input_dim=category_vocab_size, output_dim=cat_emb_dim)(category_input)
    cat_emb = Flatten()(cat_emb)

    x = Concatenate()([sku_emb, cat_emb, numeric_input])

    x = Dense(hidden_1, activation=activation)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(hidden_2, activation=activation)(x)

    output = Dense(1, activation="linear")(x)

    model = Model(
        inputs=[sku_input, category_input, numeric_input],
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse")
        ]
    )

    return model

# --------------------------------------------------
# 7. Parameter grid
# --------------------------------------------------
param_grid = {
    "hidden_1": [64, 128],
    "hidden_2": [32, 64],
    "activation": ["relu", "tanh"],
    "dropout_rate": [0.1],
    "learning_rate": [0.001],
    "batch_size": [64, 128],
    "epochs": [20]
}

all_combinations = list(itertools.product(
    param_grid["hidden_1"],
    param_grid["hidden_2"],
    param_grid["activation"],
    param_grid["dropout_rate"],
    param_grid["learning_rate"],
    param_grid["batch_size"],
    param_grid["epochs"]
))

# --------------------------------------------------
# 8. Run experiments
# --------------------------------------------------
results = []

for i, combo in enumerate(all_combinations, start=1):
    hidden_1, hidden_2, activation, dropout_rate, learning_rate, batch_size, epochs = combo

    print(f"\nRunning Experiment {i}/{len(all_combinations)}")
    print(f"hidden_1={hidden_1}, hidden_2={hidden_2}, activation={activation}, "
          f"dropout={dropout_rate}, lr={learning_rate}, batch_size={batch_size}, epochs={epochs}")

    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = build_model(
        hidden_1=hidden_1,
        hidden_2=hidden_2,
        activation=activation,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    start_time = time.time()

    history = model.fit(
        train_inputs,
        y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    end_time = time.time()
    training_time = end_time - start_time

    # Predictions
    y_train_pred = model.predict(train_inputs, verbose=0).flatten()
    y_val_pred = model.predict(val_inputs, verbose=0).flatten()
    y_test_pred = model.predict(test_inputs, verbose=0).flatten()

    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    epochs_ran = len(history.history["loss"])

    results.append({
        "experiment_id": i,
        "hidden_1": hidden_1,
        "hidden_2": hidden_2,
        "activation": activation,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_ran": epochs_ran,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "training_time_sec": training_time
    })

    print(f"Train MAE: {train_mae:.4f} | Train RMSE: {train_rmse:.4f}")
    print(f"Val MAE:   {val_mae:.4f} | Val RMSE:   {val_rmse:.4f}")
    print(f"Test MAE:  {test_mae:.4f} | Test RMSE:  {test_rmse:.4f}")
    print(f"Training Time: {training_time:.2f} sec")

# --------------------------------------------------
# 9. Save results
# --------------------------------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by=["val_mae", "val_rmse"], ascending=True)

print("\nTop Results:")
print(results_df.head())

results_df.to_csv("tuning_results.csv", index=False)
print("\nResults saved to tuning_results.csv")