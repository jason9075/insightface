import glob
import math
import os

import cv2
import numpy as np
import requests
import tensorflow as tf
import torch

import backbones
from align import align

SIM_THR = 0.3
RECOG_SIZE = 224
LANDMARK_SIZE = 112


def main():
    landmark_predictor = tf.keras.models.load_model('saved_model/face_landmark/1/')
    backbone = backbones.iresnet50(False)

    weights = torch.load("pytorch/partial_fc_glint360k_r50/16backbone.pth", map_location=torch.device('cpu'))
    backbone.load_state_dict(weights)
    backbone = backbone.float()
    backbone = backbone.eval()

    member_data = {}

    for member_path in glob.glob('imgs/member/*.jpg'):
        filename = member_path.split('/')[-1]
        origin_img = cv2.imread(f'imgs/member/{filename}')

        vector = gen_vector(landmark_predictor, origin_img, backbone)
        member_name = filename.split('.')[0]
        if not os.path.exists(f'imgs/result/{member_name}'):
            os.mkdir(f'imgs/result/{member_name}')
        member_data[member_name] = vector

    for test_path in glob.glob('imgs/log/*.jpg'):
        filename = test_path.split('/')[-1]
        origin_img = cv2.imread(f'imgs/log/{filename}')

        vector = gen_vector(landmark_predictor, origin_img, backbone)

        most_sim_value = 1.0
        most_sim_name = 'unknown'
        for member_name, member_vector in member_data.items():
            sim_value = cosine_dist(vector, member_vector)
            if sim_value < most_sim_value:
                most_sim_value = sim_value
                most_sim_name = member_name

        if most_sim_value < SIM_THR:
            print(f'imgs/result/{most_sim_name}/{most_sim_value:.2f}_{most_sim_name}_{filename}')
            cv2.imwrite(f'imgs/result/{most_sim_name}/{most_sim_value:.2f}_{most_sim_name}_{filename}', origin_img)
        else:
            print(f'imgs/result/unknown/{most_sim_value:.2f}_{most_sim_name}_{filename}')
            cv2.imwrite(f'imgs/result/unknown/{most_sim_value:.2f}_{most_sim_name}_{filename}', origin_img)



def gen_vector(landmark_predictor, origin_img, vector_predictor):
    h, w, _ = origin_img.shape
    img = cv2.resize(origin_img, (RECOG_SIZE, RECOG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img - 127.5
    img = img * 0.0078125
    landmarks = landmark_predictor.predict(np.expand_dims(img, axis=0))[0]
    landmarks = np.asarray([(int(landmark[0] * w),
                             int(landmark[1] * h))
                            for landmark in landmarks])
    align_face_img = align(origin_img, landmarks, LANDMARK_SIZE)
    # cv2.imwrite('output.jpg', align_face_img)
    # exit(0)
    align_face_img = cv2.cvtColor(align_face_img, cv2.COLOR_BGR2RGB)
    align_face_img = align_face_img - 127.5
    img = align_face_img * 0.0078125

    img = img.transpose([2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).float()
    img = torch.autograd.Variable(img, requires_grad=False).to('cpu')
    with torch.no_grad():
        vector = vector_predictor.forward(img)
    return np.asarray(vector)


def cosine_dist(v1, v2):
    dot = np.sum(np.multiply(np.asarray(v1), np.asarray(v2)), axis=1)
    norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(
        v2, axis=1
    )
    similarity = dot / norm
    distance = np.arccos(similarity) / math.pi

    return distance[0]


if __name__ == '__main__':
    main()
