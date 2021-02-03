import cv2
import numpy as np
import tensorflow as tf
from skimage import transform as trans

src = np.array(
    [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
     [33.5493, 92.3655], [62.7299, 92.2041]],
    dtype=np.float32)
SIZE = 112


def image_preprocessing(img):
    img = img - 127.5
    img = img * 0.0078125
    img = img.astype(np.float32)

    return img


def align(img, landmark, image_size):
    M = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def estimate_norm(lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()

    tform.estimate(lmk, src)
    M = tform.params[0:2, :]

    return M


def main():
    landmark_model = tf.keras.models.load_model('saved_model/face_landmark/1/')

    origin_image = cv2.imread('dataset/cele_tiny/9999667/1038077955,2622606147_align.jpg')
    h, w, _ = origin_image.shape
    img_for_landmark = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

    img_for_landmark = cv2.resize(img_for_landmark, (224, 224))
    img_for_landmark = image_preprocessing(img_for_landmark)
    landmarks = landmark_model.predict(np.expand_dims(img_for_landmark, axis=0))[0]

    landmarks = np.asarray([(int(landmark[0] * w),
                             int(landmark[1] * h))
                            for landmark in landmarks])

    img = align(origin_image, landmarks, SIZE)

    cv2.imwrite('align.jpg', img)


if __name__ == '__main__':
    main()
