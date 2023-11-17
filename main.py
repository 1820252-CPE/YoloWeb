import cv2
import argparse
import numpy
from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 480],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def plot_boxes(results, frame, confidence_threshold=0.8):
    confidences = []
    class_ids = []

    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys = boxes.xyxy

        for i in range(len(boxes)):
            confidence = boxes[i].conf
            confidences.append(confidence)

            # Print confidence for each detection
            print(f"Detection {i + 1}: Confidence = {confidence}")

            if confidence >= confidence_threshold:
                xyxy = xyxys[i]
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                class_ids.append(boxes[i].cls)

    print(confidences)
    return frame



def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("C:\\Users\\cdomi\\OneDrive\\Desktop\\Webapp\\customdatasetsemi.pt")
    model.to('cuda')

    while True:
        ret, frame = cap.read()
        results = model(frame, show=True)

        # Set confidence threshold to 0.8
        frame = plot_boxes(results, frame, confidence_threshold=0.8)

        # cv2.imshow("", frame)
        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()
