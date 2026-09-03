import tensorflow as tf
import os
import shutil

# Check TensorFlow version first
if not tf.__version__.startswith("2.15"):
    raise RuntimeError(f"Script requires TensorFlow 2.15, found {tf.__version__}")

# Original model filenames
original_models = [f"model_{i+1}.keras" for i in range(5)]

# Folder for converted models
converted_dir = "converted_models"
os.makedirs(converted_dir, exist_ok=True)

for model_file in original_models:
    if not os.path.exists(model_file):
        print(f"Warning: {model_file} not found, skipping.")
        continue

    print(f"Loading {model_file} ...")
    model = tf.keras.models.load_model(model_file, compile=False)

    # Optional: recompile with the same settings (ensures no TF 2.15-only objects remain)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.00183)
    model.compile(optimizer=optimizer, loss="mse")

    new_model_file = os.path.join(converted_dir, "model_new_" + os.path.basename(model_file))
    print(f"Saving converted model to {new_model_file} ...")
    model.save(new_model_file)

print("All models converted safely. Originals are untouched.")
