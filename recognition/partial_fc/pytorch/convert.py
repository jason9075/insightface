import cv2

from align import image_preprocessing
from backend import get_model
import torch
import numpy as np

input_size = (112, 112)


def load_weight(model, torch_path, trainable=True, verbose=False):
    torch_weights = torch.load(torch_path, map_location=torch.device('cpu'))

    for layer in model.layers:
        if verbose:
            print(f'Set layer: {layer.name}')
        layer_prefix = layer.name.split('__')[0]
        layer_type = layer.name.split('__')[-1]

        if layer_type == 'conv':
            weight = np.array(torch_weights[f'{layer_prefix}.weight'])
            weight = np.transpose(weight, [2, 3, 1, 0])
            layer.set_weights([weight])
            layer.trainable = trainable
        elif layer_type == 'bn':
            gamma = np.array(torch_weights[f'{layer_prefix}.weight'])
            beta = np.array(torch_weights[f'{layer_prefix}.bias'])
            running_mean = np.array(torch_weights[f'{layer_prefix}.running_mean'])
            running_var = np.array(torch_weights[f'{layer_prefix}.running_var'])
            layer.set_weights([gamma, beta, running_mean, running_var])
            layer.trainable = trainable
        elif layer_type == 'fc':
            weight = np.array(torch_weights[f'{layer_prefix}.weight'])
            bias = np.array(torch_weights[f'{layer_prefix}.bias'])
            weight = np.transpose(weight, [1, 0])
            layer.set_weights([weight, bias])
            layer.trainable = trainable
        elif layer_type == 'prelu':
            weight = np.array(torch_weights[f'{layer_prefix}.weight'])
            weight = weight.reshape([1, 1, -1])
            layer.set_weights([weight])
            layer.trainable = trainable
        else:
            if verbose:
                print(f'Ignore layer: {layer.name}')


def main():
    import pickle
    torch_prediction = pickle.load(open("sample.pkl", "rb"))

    # print(torch_prediction)
    model = get_model(input_size, layers=[3, 4, 14, 3])
    load_weight(model, 'pytorch/partial_fc_glint360k_r50/16backbone.pth', trainable=False, verbose=False)
    # model.summary()

    img1 = cv2.imread('boy_1.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = image_preprocessing(img1)
    img1 = np.ones([112,112,3],dtype=np.float32)

    tf_result = model.predict(np.expand_dims(img1, axis=0))

    # torch_prediction = np.transpose(torch_prediction, [0, 2, 3, 1])
    diff = torch_prediction - tf_result

    print("torch")
    print(torch_prediction[0,:10])
    print("tf")
    print(tf_result[0,:10])
    print(np.max(np.abs(diff)))


if __name__ == '__main__':
    main()
