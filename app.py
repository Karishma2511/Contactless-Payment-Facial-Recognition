from flask import Flask, render_template, Response, redirect
from flask_socketio import SocketIO
import cv2
import numpy as np
import face_recognition
import os
import stripe
from flask import request
import time


app = Flask(__name__)
socketio = SocketIO(app)

stripe.api_key = 'sk_test_51OBCeJIBgmBLNR5CTO8JKf4UNE9KJrHr3l2xOZFBMJayZyE8Arv92cR0miRmu1LCQvByYEpBIBvM0guEtDwGaHbq00m9mK6lqH'

recognized = False
dataset_dir = 'faces'
known_face_encodings = []
known_face_names = []
recognizer = cv2.face.LBPHFaceRecognizer_create()
dataset_dir = 'faces'
known_face_encodings = []
known_face_names = []


def handle_payment_status(intent):
    if intent['status'] == 'succeeded':
        print(f"Payment successful! Payment ID: {intent['id']}")
        socketio.emit('payment_status', {'status': 'success'})
    else:
        socketio.emit('payment_status', {'status': 'failure'})


def reverify_face(name):
    print(f"Please reverify your face for {name}.")
    time.sleep(3)

    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(1)
    minW = 0.1 * video_capture.get(3)
    minH = 0.1 * video_capture.get(4)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_encodings = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])
            confidence = 100

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    return
                else:
                    print("Face reverification failed. Please try again.")

        cv2.imshow('Reverification', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def initialize_payment(name):
    print(f"Payment method initialized for {name}.")
    user_response = input(f"Do you want to make a payment for {name}? (yes/no): ")

    if user_response.lower() == 'yes':
        amount = float(input("Enter the payment amount: "))
        currency = 'usd'
        start_payment(name, amount, currency)

        card_details_file = os.path.join('faces', name, 'card_details.txt')
        if os.path.exists(card_details_file):
            with open(card_details_file, 'r') as file:
                card_details = file.read()
                print("Card Details:")
                print(card_details)
        else:
            print("Card details not found.")

        reverify_face(name)
    else:
        print("Payment canceled by user.")


def start_payment(name, amount, currency):
    try:
        intent = stripe.PaymentIntent.create(
            amount=int(amount * 100),
            currency=currency,
            payment_method_types=['pm_card_visa'],
            confirm=True,
        )
        handle_payment_status(intent)
        return redirect('/payment_completed')

    except stripe.error.StripeError as e:
        return "Payment Failed"


def load_known_faces():
    global known_face_encodings, known_face_names
    for root, dirs, files in os.walk('faces'):
        for directory in dirs:
            dataset_dir = os.path.join(root, directory)
            for filename in os.listdir(dataset_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(dataset_dir, filename)
                    try:
                        image = face_recognition.load_image_file(path)
                        encoding = face_recognition.face_encodings(image)
                        if len(encoding) > 0:
                            known_face_encodings.append(encoding[0])
                            known_face_names.append(directory)
                    except Exception as e:
                        pass


load_known_faces()

def prepare_training_data(data_folder):
    faces = []
    labels = []
    label_id = {}
    current_id = 0

    for root, dirs, files in os.walk(data_folder):
        for directory in dirs:
            if directory not in label_id:
                label_id[directory] = current_id
                current_id += 1

            subject_dir = os.path.join(root, directory)
            for filename in os.listdir(subject_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(subject_dir, filename)
                    label = label_id[directory]
                    image = cv2.imread(path)
                    if image is not None:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        faces.append(gray)
                        labels.append(label)
                    else:
                        print(f"Error loading image: {path}")

    return faces, np.array(labels), label_id


if __name__ == '__main__':
    faces, labels, label_id = prepare_training_data(dataset_dir)
    recognizer.train(faces, labels)
    recognizer.save('trainer/trainer.yml')

    with open('label_mapping.txt', 'w') as file:
        for name, label in label_id.items():
            file.write(f"{name}: {label}\n")


def extract_features(img_path):
    img = face_recognition.load_image_file(img_path,'RGB')
    encodings = face_recognition.face_encodings(img)
    if len(encodings) > 0:
        return encodings[0]
    else:
        return None

def draw_text_and_rectangle(frame, face_locations, name, probability):
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name} ({probability:.2f})"
        cv2.putText(frame, label, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def generate_frames():
    global recognized
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    video_capture = cv2.VideoCapture(1)
    minW = 0.1 * video_capture.get(3)
    minH = 0.1 * video_capture.get(4)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_encodings = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])
            name = 'Unknown'
            confidence = 100

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]

                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = 100 - int(face_distances[best_match_index] * 100)

                    if not recognized:
                        initialize_payment(name)
                        recognized = True
                        amount = 55
                        currency = 'usd'
                        process_payment(amount, currency)


            cv2.putText(frame, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@socketio.on('start')
def start():
    global recognized
    recognized = False
    socketio.emit('status', {'status': 'Started'})


@socketio.on('stop')
def stop():
    global recognized
    recognized = True
    socketio.emit('status', {'status': 'Stopped'})

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('payment')
def process_payment(amount, currency):
    try:
        intent = stripe.PaymentIntent.create(
            amount= amount,
            currency= currency,
            payment_method='pm_card_visa',
            confirmation_method='manual',
            confirm=True,
            return_url='http://127.0.0.1:8000/payment_completed'
        )

        handle_payment_status(intent)

    except stripe.error.StripeError as e:
        socketio.emit('payment_status', {'status': 'failure'})



@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@socketio.on('payment_status_request')
def payment_status_request():
    socketio.emit('payment_status', {'status': 'Requested'})

@socketio.on('payment_initiate')
def handle_payment_initiate(data):
    payment_method_id = data.get('payment_method_id')
    amount = request.json.get('amount')
    currency = request.json.get('currency')
    amount = amount
    currency = currency
    process_payment(amount, currency, payment_method_id)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socketio.run(app, debug=True, port=8000)