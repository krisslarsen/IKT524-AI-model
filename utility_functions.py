import os, requests, cv2, matplotlib.pyplot as plt
from ultralytics import YOLO

def utils_image_find(local_path, url=None):
    """
    Ensure an image exists locally.
    - If it exists: return local path.
    - If not, and url is given: download and return local path.
    - If not, and no url: return message.
    """
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        return local_path

    if url:  # try download
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            return local_path
        except Exception as e:
            return f"Download failed: {e}"

    return "Image not found, add url argument"

def run_yolo(img_path, model_path="yolo11n.pt", results_dir="results"):
    """Run YOLO on img_path, save annotated result, and show it."""
    os.makedirs(results_dir, exist_ok=True)
    model = YOLO(model_path)
    results = model(img_path)[0]
    annotated = results.plot()  # BGR
    out_path = os.path.join(results_dir, f"results_{os.path.basename(img_path)}")
    cv2.imwrite(out_path, annotated)
    print("Saved:", out_path)

    # Show image
    plt.imshow(annotated[:, :, ::-1])  # BGR -> RGB
    plt.axis("off")
    plt.show()
    return out_path