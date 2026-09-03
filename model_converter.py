import os
import tensorflow as tf

# Directory containing your current .keras files
source_dir = "converted_models"  # adjust if different
# Directory to save the new SavedModel folders
target_dir = "models_saved"

os.makedirs(target_dir, exist_ok=True)

# Find all .keras files
keras_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".keras")])

for file in keras_files:
    model_path = os.path.join(source_dir, file)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Save as a folder (SavedModel format)
    folder_name = os.path.splitext(file)[0]  # e.g., "model_1"
    save_path = os.path.join(target_dir, folder_name)
    model.save(save_path)
    print(f"Saved {file} -> {save_path}")
print("All models converted to SavedModel format.")