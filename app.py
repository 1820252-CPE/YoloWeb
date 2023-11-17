from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2

from transformers import BertForQuestionAnswering, BertTokenizer
from transformers import pipeline
import textwrap

app = Flask(__name__)

# Load YOLO model
yolo_model = YOLO("C:\\Users\\cdomi\\OneDrive\\Desktop\\Webapp\\customdatasetsemi.pt")

# Open camera
cap = cv2.VideoCapture(0)

# Load BERT model for question answering
bert_model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
bert_tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
bert_qna = pipeline('question-answering', model=bert_model, tokenizer=bert_tokenizer)

# Read context from a text file
context_path = 'C:\\Users\\cdomi\\OneDrive\\Desktop\\Webapp\\templates\\context.txt'
with open(context_path, 'r', encoding='utf-8') as context_file:
    context = context_file.read()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = yolo_model.track(frame, persist=True)

            if results and results[0].boxes:
                frame = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def get_detected_objects(results):
    detected_objects = []

    if results and results[0].names:
        names_attr = results[0].names[0]

        if isinstance(names_attr, dict):
            # Case when 'names' is a dictionary
            for class_id, label in names_attr.items():
                # Check if the label is an integer
                if isinstance(label, int):
                    # If it's an integer, convert it to a string
                    label = str(label)

                detected_objects.append(label)
        elif isinstance(names_attr, str):
            # Case when 'names' is a string
            detected_objects.append(names_attr)

        print("Full Results:", results)  # Print the full results for further analysis

    print("Detected Objects:", detected_objects)
    return detected_objects

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_objects')
def detect_objects():
    # Read a new frame from the camera
    ret, frame = cap.read()

    # Get results for the current frame
    results = yolo_model.track(frame, persist=True)
    detected_objects = get_detected_objects(results)

    # Return the detected objects as JSON
    return jsonify({'detected_objects': detected_objects})

@app.route('/answer_question', methods=['POST'])
def answer_question():
    question = request.form['question']
    answer_info = bert_qna({'question': question, 'context': context})
    return render_template('index.html', inquiry=question, answer_info=answer_info)

if __name__ == '__main__':
    app.run(debug=True)