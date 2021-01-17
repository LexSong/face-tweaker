import face_recognition as fr
import numpy as np
from joblib import Memory
from PIL.Image import Image


class FaceNotFoundError(Exception):
    pass


memory = Memory(".face_encodings_cache", verbose=0)


@memory.cache
def _get_face_encoding(image):
    face_locations = fr.face_locations(image)
    if not face_locations:
        raise FaceNotFoundError

    def rect_area(args):
        top, right, bottom, left = args
        return (bottom - top) * (right - left)

    largest_face = max(face_locations, key=rect_area)
    face_encoding = fr.face_encodings(image, [largest_face])[0]
    return face_encoding


def get_face_encoding(image: Image):
    return _get_face_encoding(np.array(image))


def get_face_distance(a, b):
    return fr.face_distance([a], b)[0]
