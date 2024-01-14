from flask import Flask, render_template, Response, redirect
from flask_socketio import SocketIO
import cv2
import numpy as np
import face_recognition
import os
import stripe
from flask import request


app = Flask(__name__)
socketio = SocketIO(app)

# Replace with your Stripe secret key
stripe.api_key = 'sk_test_51OBCeJIBgmBLNR5CTO8JKf4UNE9KJrHr3l2xOZFBMJayZyE8Arv92cR0miRmu1LCQvByYEpBIBvM0guEtDwGaHbq00m9mK6lqH'

recognized = False
dataset_dir = 'faces'
known_face_encodings = []
known_face_names = []
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Define the dataset directory
dataset_dir = 'faces'
known_face_encodings = []
known_face_names = []


def handle_payment_status(intent):
    if intent['status'] == 'succeeded':
        print(f"Payment successful! Payment ID: {intent['id']}")
        socketio.emit('payment_status', {'status': 'success'})
    else:
        print(f"Payment failed. Reason: {intent['last_payment_error']['message']}")
        socketio.emit('payment_status', {'status': 'failure'})

def initialize_payment(name):
    print(f"Payment method initialized for {name}.")
    socketio.emit('payment_initiated', {'name': name})


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
                        pass  # Removing logging, just silently handle exceptions


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

    return faces, np.array(labels), label_id  # Return the label mapping dictionary as well

if __name__ == '__main__':
    # Use captured images to train the recognizer
    faces, labels, label_id = prepare_training_data(dataset_dir)

    # Train with integer labels
    recognizer.train(faces, labels)
    recognizer.save('trainer/trainer.yml')  # Save the trained model to a file

    # Save label mapping to a file (optional)
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
    video_capture = cv2.VideoCapture(0)
    minW = 0.1 * video_capture.get(3)
    minH = 0.1 * video_capture.get(4)

    correct_recognitions = 0
    total_recognitions = 0

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

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    recognized_name = known_face_names[best_match_index]
                    actual_name = 'Expected Name'  # Replace with the actual expected name

                    if recognized_name == actual_name:
                        correct_recognitions += 1

                    total_recognitions += 1

        accuracy = (correct_recognitions / total_recognitions) * 100 if total_recognitions > 0 else 0
        print(f"Accuracy: {accuracy}%")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

train_dir = 'train_faces'
test_dir = 'test_faces'

def load_known_faces(directory):
    known_face_encodings = []
    known_face_names = []
    for root, dirs, files in os.walk(directory):
        for sub_dir in dirs:
            dataset_dir = os.path.join(root, sub_dir)
            for filename in os.listdir(dataset_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(dataset_dir, filename)
                    try:
                        image = face_recognition.load_image_file(path)
                        encoding = face_recognition.face_encodings(image)
                        if len(encoding) > 0:
                            known_face_encodings.append(encoding[0])
                            known_face_names.append(sub_dir)
                    except Exception as e:
                        pass  # Silently handle exceptions
    return known_face_encodings, known_face_names

known_face_encodings_train, known_face_names_train = load_known_faces(train_dir)
known_face_encodings_test, known_face_names_test = load_known_faces(test_dir)

correct_recognitions = 0
total_recognitions = 0

for face_encoding_test, name_test in zip(known_face_encodings_test, known_face_names_test):
    matches = face_recognition.compare_faces(known_face_encodings_train, face_encoding_test)
    if any(matches):
        index = matches.index(True)
        recognized_name_train = known_face_names_train[index]
        if recognized_name_train == name_test:
            correct_recognitions += 1
        total_recognitions += 1

accuracy = (correct_recognitions / total_recognitions) * 100 if total_recognitions > 0 else 0
print(f"Accuracy: {accuracy}%")


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

@app.route('/start_payment', methods=['POST'])
def start_payment():
    amount = request.json.get('amount')
    currency = request.json.get('currency')
    payment_token = request.json.get('payment_token')

    try:
        intent = stripe.PaymentIntent.create(
            amount=amount,
            currency=currency,
            payment_method_types=['card'],
            payment_method_data={'type': 'card', 'card': {'token': payment_token}},
            confirm=True,
        )
        handle_payment_status(intent)
        return redirect('/payment_completed')

    except stripe.error.StripeError as e:
        print(f"Payment failed: {e.error.message}")
        return "Payment Failed"

@socketio.on('payment')
def process_payment(amount, currency):
    try:
        # Create a PaymentIntent on Stripe without a payment method
        intent = stripe.PaymentIntent.create(
            amount=5500,  # The amount in cents, for example, $55.00
            currency='usd',
            payment_method='pm_card_visa',  # Replace with an actual payment method ID or token
            confirmation_method='manual',
            confirm=True,
            return_url='http://127.0.0.1:8000/payment_completed'  # Your return URL
        )

        # Handle payment status based on Stripe intent status
        handle_payment_status(intent)

    except stripe.error.StripeError as e:
        # Payment failed for various reasons
        print(f"Payment failed: {e.error.message}")
        socketio.emit('payment_status', {'status': 'failure'})


@app.route('/payment_completed')
def payment_completed():
    return render_template('payment_completed.html')

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
    # Assuming you have the desired amount and currency
    amount = 55
    currency = 'usd'
    process_payment(amount, currency, payment_method_id)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socketio.run(app, debug=True, port=8000)