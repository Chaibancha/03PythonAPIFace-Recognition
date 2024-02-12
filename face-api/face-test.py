from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import pickle

know_face_names, known_face_encodings = pickle.load(open('faces.p', 'rb'))

# print(know_face_names, known_face_encoding)
image = Image.open('test/group2.jpg')

face_locations = face_recognition.face_locations(np.array(image))
face_encodings = face_recognition.face_encodings(np.array(image), face_locations)
# print(face_locations)

draw = ImageDraw.Draw(image)

for face_encoding, face_location in zip(face_encodings, face_locations):
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    name = know_face_names[best_match_index]
    
    top, right, bottom, left = face_location
    draw.rectangle([left, top, right, bottom])
    draw.text((left, top), name)
    print(name)
    
image.show()