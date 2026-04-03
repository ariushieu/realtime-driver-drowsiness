import os

import numpy as np
import tensorflow as tf


MODEL_CANDIDATES = [
    ("B1_16_batches.weights.keras", "B1"),
    ("B0_16_batches.weights.keras", "B0"),
]


def build_model(variant):
    if variant == "B1":
        efficientnet_cls = tf.keras.applications.EfficientNetB1
    else:
        efficientnet_cls = tf.keras.applications.EfficientNetB0

    # Training used 224x224 for ALL variants
    input_size = 224

    base_model = efficientnet_cls(
        weights=None,
        include_top=False,
        input_shape=(input_size, input_size, 3),
    )
    # DO NOT pre-build — it causes variable name mismatches
    # that break weight loading for the Normalization layer.
    base_model.trainable = True

    model = tf.keras.models.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(6, activation="softmax"),
        ]
    )
    return model


def main():
    selected = next(((p, v) for p, v in MODEL_CANDIDATES if os.path.exists(p)), None)
    if selected is None:
        raise FileNotFoundError(
            "No model weights file found. Tried: "
            + ", ".join(path for path, _ in MODEL_CANDIDATES)
        )

    model_path, variant = selected
    model = build_model(variant)
    model.load_weights(model_path)
    print(f"Loaded weights from: {model_path} (EfficientNet{variant})")

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        print(f"\nLayer: {layer.name}  ({len(weights)} weight tensors)")

        # For the base EfficientNet layer, drill into sublayers
        # to give a clearer picture of what was actually loaded.
        if hasattr(layer, "layers"):
            # Summarise all sublayer weight tensors
            all_vals = np.concatenate([w.flatten() for w in weights])
            n_zero = np.sum(all_vals == 0)
            print(f" - Total params across sublayers: {len(all_vals):,}")
            print(f" - Global stats: mean={np.mean(all_vals):.6f}, "
                  f"std={np.std(all_vals):.6f}, "
                  f"min={np.min(all_vals):.6f}, max={np.max(all_vals):.6f}")
            print(f" - Zero params: {n_zero:,} / {len(all_vals):,} "
                  f"({n_zero/len(all_vals)*100:.1f}%)")

            # Show the first tensor (Normalization layer) explicitly
            w0 = weights[0]
            print(f" - weights[0] (Normalization mean): shape={w0.shape}, "
                  f"values={w0.tolist()}")
            w1 = weights[1]
            print(f" - weights[1] (Normalization variance): shape={w1.shape}, "
                  f"values={w1.tolist()}")

            # Show a few conv kernel tensors to confirm they are non-zero
            for idx, w in enumerate(weights):
                if len(w.shape) == 4 and w.shape[0] >= 3:
                    print(f" - weights[{idx}] (first conv kernel found): "
                          f"shape={w.shape}, mean={np.mean(w):.6f}, "
                          f"std={np.std(w):.6f}")
                    break
        else:
            w = weights[0]
            print(f" - Weights shape: {w.shape}")
            print(f" - Max weight: {np.max(w)}, Min weight: {np.min(w)}")
            print(f" - Mean weight: {np.mean(w)}, Std: {np.std(w)}")

            if len(weights) > 1:
                b = weights[1]
                print(
                    f" - Bias shape: {b.shape}, Max: {np.max(b)}, Min: {np.min(b)}, "
                    f"Mean: {np.mean(b)}, Std: {np.std(b)}"
                )


if __name__ == "__main__":
    main()