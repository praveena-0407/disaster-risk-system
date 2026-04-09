"""
image_classifier.py ─ Disaster Image Classifier
=================================================
Loads the trained CNN model and predicts disaster type from an image.

Usage:
    python image_classifier.py                        # uses built-in test
    python image_classifier.py path/to/image.jpg      # classify any image
"""

import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "cnn_model/disaster_cnn.h5"
IMAGE_SIZE   = (224, 224)

CLASSES = [
    "Damaged_Infrastructure",
    "Fire_Disaster",
    "Human_Damage",
    "Land_Disaster",
    "Non_Damage",
    "Water_Disaster",
]

# Maps CNN classes → user-friendly display labels + emoji
DISPLAY_MAP = {
    "Damaged_Infrastructure": ("🏚️",  "DAMAGED INFRASTRUCTURE"),
    "Fire_Disaster":          ("🔥",  "FIRE DISASTER"),
    "Human_Damage":           ("🚨",  "HUMAN DAMAGE"),
    "Land_Disaster":          ("⛰️",  "LAND DISASTER (Landslide/Earthquake)"),
    "Non_Damage":             ("✅",  "NO DISASTER DETECTED"),
    "Water_Disaster":         ("🌊",  "WATER DISASTER (Flood)"),
}


# ── Core Functions ────────────────────────────────────────────────────────────

def load_classifier():
    """Load the saved CNN model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        print(f"❌ Failed to load model from '{MODEL_PATH}': {e}")
        print("   Make sure you have run train_cnn.py first.")
        sys.exit(1)


def preprocess_image(img_path: str) -> np.ndarray:
    """Load and preprocess an image for prediction."""
    try:
        img = keras_image.load_img(img_path, target_size=IMAGE_SIZE)
        arr = keras_image.img_to_array(img) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        print(f"❌ Could not load image '{img_path}': {e}")
        sys.exit(1)


def classify_image(img_path: str, model=None) -> dict:
    """
    Classify a disaster image.

    Returns:
        dict with keys: label, display_label, emoji, confidence, all_probs
    """
    if model is None:
        model = load_classifier()

    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array, verbose=0)[0]

    top_idx        = int(np.argmax(predictions))
    label          = CLASSES[top_idx]
    confidence     = float(predictions[top_idx]) * 100
    emoji, display = DISPLAY_MAP[label]

    all_probs = {
        CLASSES[i]: round(float(predictions[i]) * 100, 2)
        for i in range(len(CLASSES))
    }

    return {
        "label":         label,
        "display_label": display,
        "emoji":         emoji,
        "confidence":    round(confidence, 2),
        "all_probs":     all_probs,
    }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("⚠️  No image path provided.")
        print("Usage: python image_classifier.py path/to/image.jpg\n")
        print("Running with a quick model load test instead...\n")
        model = load_classifier()
        print(f"✅ Model loaded successfully!")
        print(f"   Input shape : {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Classes     : {CLASSES}")
        return

    img_path = sys.argv[1]
    print(f"\n🔍 Classifying: {img_path}\n")

    model  = load_classifier()
    result = classify_image(img_path, model)

    print("─" * 45)
    print(f"{result['emoji']}  Result: {result['display_label']}")
    print(f"   Confidence : {result['confidence']:.1f}%")
    print("─" * 45)
    print("\n📊 All class probabilities:")
    for cls, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob / 5)
        print(f"   {cls:<28} {prob:5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()