import os

MODEL_PATH = "model.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            return joblib.load(MODEL_PATH)
        except Exception as e:
            print("Failed to load model.pkl:", e)
            return None
    else:
        print("No model file found - using dummy model.")
        return None

def predict_from_model(model, features):
    if model is not None:
        try:
            pred = model.predict([features])
            return pred[0].tolist() if hasattr(pred[0], "tolist") else pred[0]
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")

    try:
        s = sum(map(float, features))
        return {"dummy_sum": s, "count": len(features)}
    except Exception:
        return {"echo": features}
