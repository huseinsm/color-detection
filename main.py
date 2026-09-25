import argparse

import cv2

from util import color_mask, get_hsv_ranges, parse_color

BOX_COLOR = (0, 255, 0)
WINDOW = "color-detection"


def detect(frame, ranges, min_area):
    """Draw a box around every blob of the target color and return the boxes."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = color_mask(hsv, ranges)

    # Remove speckle noise before looking for blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
    return boxes, mask


def open_source(source):
    # A bare number means a webcam index, anything else is a video file path.
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def main():
    parser = argparse.ArgumentParser(description="Track a color in real time using HSV masking.")
    parser.add_argument("--color", default="yellow", help='preset name (red, yellow, blue, ...) or "R,G,B"')
    parser.add_argument("--source", default="0", help="webcam index or path to a video file")
    parser.add_argument("--min-area", type=int, default=500, help="ignore blobs smaller than this many pixels")
    parser.add_argument("--show-mask", action="store_true", help="also display the binary mask")
    args = parser.parse_args()

    ranges = get_hsv_ranges(parse_color(args.color))
    cap = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f"could not open video source {args.source!r}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, mask = detect(frame, ranges, args.min_area)
            cv2.putText(frame, f"{args.color}: {len(boxes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOR, 2)

            cv2.imshow(WINDOW, frame)
            if args.show_mask:
                cv2.imshow("mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
