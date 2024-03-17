from flask import Flask, render_template, request, jsonify
import os
import face_recognition
import cv2
from io import BytesIO

import cv2_imshow
from PIL import Image

DATABASE_PATH = r"C:\Users\Pc\Desktop\CH\Lebanese University\M2\AI Python\Grumpy\Database"
app = Flask(__name__, template_folder=os.path.abspath('templates'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form['user_message']
    # Here you can process the user's message and generate a response
    # For simplicity, let's just echo the user's message
    bot_response = user_message
    if 'image' in request.files:
        img_file = request.files['image']
        if img_file.filename != '':
            img_bytes = img_file.read()
            img = Image.open(BytesIO(img_bytes))
            img.show()  # Display the image
            return jsonify({'bot_response': 'rcvd'})

    return jsonify({'bot_response': 'not rcvd'})

if __name__ == '__main__':
    #dataset = fiftyone.zoo.load_zoo_dataset("coco-2017")

    app.run(debug=True)
def recognize_familiar_face(image_path):


    # Load known face encodings and names (from your dataset)
    known_face_encodings = []  # List of known face encodings
    known_face_names = []  # List of corresponding person names

    confidence_threshold = 0.5  # Adjust as needed

    # Iterate through your dataset and populate the lists
    for person_folder in os.listdir(DATABASE_PATH):
        person_path = os.path.join(DATABASE_PATH, person_folder)

        if os.path.isdir(person_path):
            images = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for image in images:
                image_path = os.path.join(person_path, image)
                face_image = face_recognition.load_image_file(image_path)

                # Assume there's only one face per image for simplicity
                face_encodings = face_recognition.face_encodings(face_image)

                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    known_face_names.append(person_folder)
                # Append the encoding and corresponding name to the lists
                # known_face_encodings.append(face_encoding)
                # known_face_names.append(person_folder)

    # Load an unknown image for face recognition
    unknown_image_path = r'C:\Users\Pc\Desktop\CH\Lebanese University\M2\AI Python\Grumpy\Test\einstein.jpeg'
    unknown_image = face_recognition.load_image_file(unknown_image_path)

    # Find face locations and encode faces in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Compare faces in the unknown image with known faces
    # Change this part of the code
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        confidence = 0.0
        name = "Unknown"  # Default name if no match is found

        # If a match is found, use the name of the known face
        print("matches : " + str(matches))
        if any(matches):
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            confidence = 1.0 - face_distances[first_match_index]
            similarity_score = 1 / (1 + face_distances[first_match_index])  # Reciprocal transformation
        if confidence >= confidence_threshold:
            label = f"{name}: {confidence:.2%}"
            return name
        else:
            label = "Unknown Person"
            return label
        # Print the name and draw a rectangle around the face in the image
        print("Name:", name)
        cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(unknown_image, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # To
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if any face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        confidence = 0.0
        name = "Unknown"  # Default name if no match is found

        # If a match is found, use the name of the known face with the highest confidence
        if any(matches):
            max_confidence_index = matches.index(True)
            for i in range(len(matches)):
                if matches[i] and face_distances[i] > confidence:
                    max_confidence_index = i
                    confidence = face_distances[i]
                    print("conf is " + str(confidence))
            name = known_face_names[max_confidence_index]
            similarity_score = 1 / (1 + confidence)  # Reciprocal transformation
        print("conf is " + str(confidence))
        if confidence >= confidence_threshold:
            label = f"{name}: {confidence:.2%}"
        else:
            label = "Unknown Person"
            user_input_name = input("Enter a name for the unknown person: ")
            name = user_input_name.strip()
            os.makedirs(f'{DATABASE_PATH}/{name}')
            try:
                cv2.imwrite(f'{DATABASE_PATH}/{name}/{name}-1.jpg', unknown_image)
            except Exception as e:
                print("Error saving image:", str(e))

        # directory_path = '/content/Original Images/'
        # if not os.path.exists(directory_path):
        #     os.makedirs(directory_path)

        # Print the name and draw a rectangle around the face in the image
        print("Name:", name)
        cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(unknown_image, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the result
    # cv2.imshow('Face Recognition Result', cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR))

    cv2_imshow(cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR))

    cv2.waitKey(0)
    cv2.destroyAllWindows()