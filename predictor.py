import torchvision
from PIL import *
from model import fusionVGG19
import torch
from skimage import io, transform
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


config = AttrDict()
config.update({
    "batchSize": 1,
    "landmarkNum": 19,
    "image_scale": (800, 640),
    "use_gpu": 0,
    "spacing": 0.1,
    "R1": 41,
    "R2": 41,
    "epochs": 100,
    "data_enhanceNum": 1,
    "stage": "train",
    "saveName": "test1",
    "testName": "30cepha100_fusion_unsuper.pkl",
    "dataRoot": "../input/process-data/",
    "supervised_dataset_train": "cepha/",
    "supervised_dataset_test": "cepha/",
    "unsupervised_dataset": "cepha/",
    "trainingSetCsv": "cepha_train.csv",
    "testSetCsv": "cepha_test.csv",
    "unsupervisedCsv": "cepha_val.csv"
})
model = fusionVGG19(torchvision.models.vgg19_bn(pretrained=True), config).to("cpu")
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image = sample
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant')

        return {'image': img}


transform_origin = torchvision.transforms.Compose([
    Rescale(config.image_scale), ToTensor()
])


def image_loader(image):
    """load image, returns cuda tensor"""
    #image = io.imread(image_name)
    image = transform_origin(image)
    image = image.unsqueeze(0)
    return image.to('cpu')


def get_predicted_heatmaps(image):
    tensor = image_loader(image)
    tensor = tensor.to('cpu')
    with torch.no_grad():
        output = model.forward(tensor)
    return output,tensor


def regression_voting(heatmaps, R):
    topN = int(R * R * 3.1415926)
    heatmap = heatmaps[0]
    imageNum, featureNum, h, w = heatmap.size()
    landmarkNum = int(featureNum / 3)
    heatmap = heatmap.contiguous().view(imageNum, featureNum, -1)

    predicted_landmarks = torch.zeros((imageNum, landmarkNum, 2))
    Pmap = heatmap[:, 0:landmarkNum, :].data
    Xmap = torch.round(heatmap[:, landmarkNum:landmarkNum * 2, :].data * R).long() * w
    Ymap = torch.round(heatmap[:, landmarkNum * 2:landmarkNum * 3, :].data * R).long()
    topkP, indexs = torch.topk(Pmap, topN)
    for imageId in range(imageNum):
        for landmarkId in range(landmarkNum):

            topnXoff = Xmap[imageId][landmarkId][indexs[imageId][landmarkId]]  # offset in x direction
            topnYoff = Ymap[imageId][landmarkId][indexs[imageId][landmarkId]]  # offset in y direction
            VotePosi = (topnXoff + topnYoff + indexs[imageId][landmarkId]).cpu().numpy().astype("int")
            tem = VotePosi[VotePosi >= 0]
            maxid = 0
            if len(tem) > 0:
                maxid = np.argmax(np.bincount(tem))
            x = maxid // w
            y = maxid - x * w
            x, y = x / (h - 1), y / (w - 1)
            predicted_landmarks[imageId][landmarkId] = torch.Tensor([x, y])
    return predicted_landmarks


=======
import torchvision
from PIL import *
from model import fusionVGG19
import torch
from skimage import io, transform
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


config = AttrDict()
config.update({
    "batchSize": 1,
    "landmarkNum": 19,
    "image_scale": (800, 640),
    "use_gpu": 0,
    "spacing": 0.1,
    "R1": 41,
    "R2": 41,
    "epochs": 100,
    "data_enhanceNum": 1,
    "stage": "train",
    "saveName": "test1",
    "testName": "30cepha100_fusion_unsuper.pkl",
    "dataRoot": "../input/process-data/",
    "supervised_dataset_train": "cepha/",
    "supervised_dataset_test": "cepha/",
    "unsupervised_dataset": "cepha/",
    "trainingSetCsv": "cepha_train.csv",
    "testSetCsv": "cepha_test.csv",
    "unsupervisedCsv": "cepha_val.csv"
})
model = fusionVGG19(torchvision.models.vgg19_bn(pretrained=True), config).to("cpu")
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image = sample
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant')

        return {'image': img}


transform_origin = torchvision.transforms.Compose([
    Rescale(config.image_scale), ToTensor()
])


def image_loader(image):
    """load image, returns cuda tensor"""
    #image = io.imread(image_name)
    image = transform_origin(image)
    image = image.unsqueeze(0)
    return image.to('cpu')


def get_predicted_heatmaps(image):
    tensor = image_loader(image)
    tensor = tensor.to('cpu')
    with torch.no_grad():
        output = model.forward(tensor)
    return output,tensor


def regression_voting(heatmaps, R):
    topN = int(R * R * 3.1415926)
    heatmap = heatmaps[0]
    imageNum, featureNum, h, w = heatmap.size()
    landmarkNum = int(featureNum / 3)
    heatmap = heatmap.contiguous().view(imageNum, featureNum, -1)

    predicted_landmarks = torch.zeros((imageNum, landmarkNum, 2))
    Pmap = heatmap[:, 0:landmarkNum, :].data
    Xmap = torch.round(heatmap[:, landmarkNum:landmarkNum * 2, :].data * R).long() * w
    Ymap = torch.round(heatmap[:, landmarkNum * 2:landmarkNum * 3, :].data * R).long()
    topkP, indexs = torch.topk(Pmap, topN)
    for imageId in range(imageNum):
        for landmarkId in range(landmarkNum):

            topnXoff = Xmap[imageId][landmarkId][indexs[imageId][landmarkId]]  # offset in x direction
            topnYoff = Ymap[imageId][landmarkId][indexs[imageId][landmarkId]]  # offset in y direction
            VotePosi = (topnXoff + topnYoff + indexs[imageId][landmarkId]).cpu().numpy().astype("int")
            tem = VotePosi[VotePosi >= 0]
            maxid = 0
            if len(tem) > 0:
                maxid = np.argmax(np.bincount(tem))
            x = maxid // w
            y = maxid - x * w
            x, y = x / (h - 1), y / (w - 1)
            predicted_landmarks[imageId][landmarkId] = torch.Tensor([x, y])
    return predicted_landmarks


