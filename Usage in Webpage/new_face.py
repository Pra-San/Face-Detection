import face_recognition
import numpy as np



def encode(new_name,image_path):
    new_name = np.array(new_name)
    new_image= face_recognition.load_image_file(image_path)
    new_encoding = face_recognition.face_encodings(new_image)[0]

    try:

        known_face_encodings=np.load('encodings.npy')
        known_face_names=np.load('names.npy')
        new_encoding = new_encoding.reshape(1, -1)
        known_face_names = np.append(known_face_names, new_name)
        known_face_encodings = np.append(known_face_encodings, new_encoding, axis=0)
    except FileNotFoundError:
        known_face_encodings=np.array([new_encoding])
        known_face_names=np.array(new_name)

    np.save('encodings.npy',known_face_encodings)
    np.save('names.npy',known_face_names)

