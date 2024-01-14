import cv2
import os

dataset_dir = 'faces'

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

def is_person_registered(person_name):
    person_dir = os.path.join(dataset_dir, person_name)
    return os.path.exists(person_dir)

def capture_faces():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    person_name = input('\nEnter the name of the person and press <return>: ')

    person_dir = os.path.join(dataset_dir, person_name)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        card_number = input('\nEnter the card number: ')
        expiry_date = input('Enter the expiry date (MM/YY): ')
        cvv = input('Enter the CVV: ')
        card_details_path = os.path.join(person_dir, 'card_details.txt')
        with open(card_details_path, 'w') as card_file:
            card_file.write(f'Card Number: {card_number}\nExpiry Date: {expiry_date}\nCVV: {cvv}')

    print("\n[INFO] Initializing face capture. Look at the camera and wait...")

    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            img_path = os.path.join(person_dir, f"{person_name}_{count}.jpg")
            cv2.imwrite(img_path, gray[y:y+h, x:x+w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 250:
            break

    print("\n[INFO] Exiting program and cleaning up")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    choice = input("Enter '1' to capture images for an existing person or '2' to capture new faces: ")

    if choice == '1':
        person_name = input("Enter the name of the existing person: ")
        if is_person_registered(person_name):
            print(f"{person_name} is already registered. Capturing images...")
            capture_faces()
        else:
            print(f"Person {person_name} is not registered.")
    elif choice == '2':
        capture_faces()
    else:
        print("Invalid choice.")