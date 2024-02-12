import face_recognition
from PIL import Image, ImageDraw
import pickle

known_faces = [
    ['Lisa', 'lisa.jpg'],
    ['Rose', 'rose.jpg'],
    ['Jennie', 'jennie.jpg'],
    ['Jisoo', 'jisoo.jpg'],
]

known_faces_names = []
known_faces_encoding = []
for face in known_faces:
    known_faces_names.append(face[0])
    face_image = face_recognition.load_image_file(face[1])
    # ตรวจสอบว่ามีหน้าในรูปภาพหรือไม่ก่อนที่จะทำการดึง encoding
    face_encodings = face_recognition.face_encodings(face_image)
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        known_faces_encoding.append(face_encoding)
    else:
        print(f"No face found in {face[1]}")

print(known_faces_names)
print(known_faces_encoding)
pickle.dump((known_faces_names, known_faces_encoding), open('faces.p', 'wb'))