from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    detected_image = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Run object detection
            results = model(filepath)

            # Read image
            image = cv2.imread(filepath)

            # Draw detections
            for result in results:
                boxes = result.boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    label = model.names[cls]

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(
                        image,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            output_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                "detected_" + file.filename
            )

            cv2.imwrite(output_path, image)

            detected_image = output_path

    return render_template("index.html", detected_image=detected_image)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 