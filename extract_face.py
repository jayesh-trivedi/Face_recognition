'''
module to extract faces from images

'''

from PIL import Image
from mtcnn.mtcnn import MTCNN
import numpy as np


def extract_face(filename, required_size=(160, 160)):
    " to extract faces from pictures "
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1_coordinate, y1_coordinate, width, height = results[0]['box']
    x1_coordinate, y1_coordinate = abs(x1_coordinate), abs(y1_coordinate)
    x2_coordinate, y2_coordinate = x1_coordinate + width, y1_coordinate + height
    face = pixels[y1_coordinate:y2_coordinate, x1_coordinate:x2_coordinate]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array
