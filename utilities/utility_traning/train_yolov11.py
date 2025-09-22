import os
import argparse
from ultralytics import YOLO
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv11 with custom parameters")
    # Data / model
    p.add_argument('--data', type=str, default=None,
                   help='Path to data.yaml. If omitted, chosen by --model_type.')
    p.add_argument('--weights', type=str, default='yolo11m.pt',
                   help='YOLOv11 weights: yolo11n/s/m/l/x.pt or a local .pt')
    p.add_argument('--model_type', type=str, default='character',
                   choices=['license_plates', 'character'],
                   help='Profile that sets default data path + augmentations')

    # Training basics
    p.add_argument('--imgsz', type=int, default=640, help='Training image size')
    p.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    p.add_argument('--batch', type=int, default=16, help='Batch size')
    p.add_argument('--name', type=str, default='run_yolo11', help='Run name')
    p.add_argument('--device', type=str, default='0',
                   help='CUDA device like "0", "0,1", or "cpu"')
    p.add_argument('--resume', action='store_true', help='Resume last training')

    # Aug overrides (optional)
    p.add_argument('--hsv_h', type=float, default=None)
    p.add_argument('--hsv_s', type=float, default=None)
    p.add_argument('--hsv_v', type=float, default=None)
    p.add_argument('--degrees', type=float, default=None)
    p.add_argument('--translate', type=float, default=None)
    p.add_argument('--scale', type=float, default=None)
    p.add_argument('--shear', type=float, default=None)
    p.add_argument('--perspective', type=float, default=None)
    p.add_argument('--flipud', type=float, default=None)
    p.add_argument('--fliplr', type=float, default=None)
    p.add_argument('--mosaic', type=float, default=None)
    p.add_argument('--mixup', type=float, default=None)

    return p.parse_args()

def main():
    args = parse_args()

    # === Defaults per profile (mirrors your v8 script) ===
    # Update these dataset paths to your repo layout if needed.
    if args.data is None:
        if args.model_type == 'license_plates':
            args.data = 'all_datasets/dataset_1_licenseplates/data.yaml'
        else:
            args.data = 'all_datasets/dataset_2_characters/data.yaml'

    if args.model_type == 'character':
        # gentler augs (from your v8 script)
        defaults = dict(
            hsv_h=0, hsv_s=0, hsv_v=0,
            degrees=1.0, translate=0.0, scale=0.1, shear=0.0, perspective=0.001,
            flipud=0.0, fliplr=0.0, mosaic=0.0, mixup=0.0,
        )
    else:
        # stronger augs (from your v8 script)
        defaults = dict(
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.6,
            degrees=10.0, translate=0.1, scale=0.3, shear=3.0, perspective=0.002,
            flipud=0.0, fliplr=0.5, mosaic=0.5, mixup=0.1,
        )

    # Allow CLI to override any default aug
    aug = {k: (getattr(args, k) if getattr(args, k) is not None else v)
           for k, v in defaults.items()}

    # Load YOLOv11 weights
    model = YOLO(args.weights)

    # Train
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        device=args.device,
        resume=args.resume,
        save=True,
        plots=True,
        lr0=1e-3,
        **aug
    )

if __name__ == "__main__":
    main()
