from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-seg.pt")

    model.train(
        data="core/dataset/data.yaml",
        epochs=200,
        imgsz=748,
        batch=2,
        project="runs",
        name="mest_segmenter",
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
    )


if __name__ == "__main__":
    main()