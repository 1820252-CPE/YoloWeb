from flask import Flask, render_template, Response, jsonify, request
from flask_mysqldb import MySQL
from ultralytics import YOLO
import cv2
from transformers import BertForQuestionAnswering, BertTokenizer, pipeline

app = Flask(__name__)

app.config['MYSQL_HOST'] = "localhost"
app.config['MYSQL_USER'] = "root"
app.config['MYSQL_PASSWORD'] = ""
app.config['MYSQL_DB'] = "detection"
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

yolo_model = None
cap = None

bert_model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
bert_tokenizer = BertTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
bert_qna = pipeline('question-answering', model=bert_model, tokenizer=bert_tokenizer)

context_path = 'C:\\Users\\cdomi\\OneDrive\\Desktop\\Webapp\\templates\\context.txt'
with open(context_path, 'r', encoding='utf-8') as context_file:
    context = context_file.read()

def initialize_yolo_and_camera():
    global yolo_model, cap
    yolo_model = YOLO("C:\\Users\\cdomi\\OneDrive\\Desktop\\Webapp\\customdatasetsemi.pt")
    cap = cv2.VideoCapture(0)

if __name__ == '__main__':
    initialize_yolo_and_camera()

    with app.app_context():
        with mysql.connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    object_type VARCHAR(255) NOT NULL,
                    count INT DEFAULT 0,
                    UNIQUE KEY (object_type)
                )
                """
            )
            mysql.connection.commit()

def insert_detection_count(object_type, count):
    with app.app_context():
        cursor = mysql.connection.cursor()
        cursor.execute(
            "INSERT INTO detection_data (object_type, count) VALUES (%s, %s) ON DUPLICATE KEY UPDATE count = %s",
            (object_type, count, count)
        )
        mysql.connection.commit()
        cursor.close()
        print(f"Inserted count {count} for object type {object_type}")

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            results = yolo_model.track(frame, persist=True)
            print("YOLO Results:", results)

            detected_objects = get_detected_objects(results)

            for obj_type in set(detected_objects):
                obj_count = detected_objects.count(obj_type)
                insert_detection_count(obj_type, obj_count)

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
            for class_id, label in names_attr.items():
                if isinstance(label, int):
                    label = str(label)

                detected_objects.append(label)
        elif isinstance(names_attr, str):
            detected_objects.append(names_attr)

    print("Detected Objects:", detected_objects)
    return detected_objects

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_objects')
def detect_objects():
    ret, frame = cap.read()
    results = yolo_model.track(frame, persist=True)
    detected_objects = get_detected_objects(results)
    return jsonify({'detected_objects': detected_objects})

@app.route('/answer_question', methods=['POST'])
def answer_question():
    question = request.form['question']
    answer_info = bert_qna({'question': question, 'context': context})
    return render_template('index.html', inquiry=question, answer_info=answer_info)

if __name__ == '__main__':
    app.run(debug=True)
