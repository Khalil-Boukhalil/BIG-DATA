import torch
from ultralytics import YOLO

if __name__ == "__main__":  # ✅ Add this for multiprocessing safety
    # Load trained model
    model_path = r"C:\Users\khali\runs\detect\train67\weights\best.pt"  # Update path if needed
    model = YOLO(model_path)

    # Run validation (Reduce workers to avoid multiprocessing issues)
    data_path = r"C:\Users\khali\PycharmProjects\BIG_DATA_T3\PKlot\data.yaml"
    results = model.val(data=data_path, workers=0)  # ✅ Set workers=0 for Windows fix

    # Print results summary
    print(results)
