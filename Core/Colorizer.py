import numpy as np
import cv2 as cv

class Colorizer:
    def __init__(self, image):
        self.inputImage = image
        imageData = self.inputImage
        (self.height, self.width) = imageData.shape[0], imageData.shape[1]
        
        self.colorModel = cv.dnn.readNetFromCaffe("Models/colorization_deploy_v2.prototxt", "Models/colorization_release_v2.caffemodel")

        clusterCenters = np.load("Models/pts_in_hull.npy")
        clusterCenters = clusterCenters.transpose().reshape(2, 313, 1, 1)

        self.colorModel.getLayer(self.colorModel.getLayerId('class8_ab')).blobs = [clusterCenters.astype(np.float32)]
        self.colorModel.getLayer(self.colorModel.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

    def processImage(self):
        self.image = self.inputImage
        self.image = cv.resize(self.image, (self.width, self.height))

        self.processFrame()
        return self.result

    def processFrame(self):
        imgNormalized = self.image.astype(np.float32) / 255.0

        imgLab = cv.cvtColor(imgNormalized, cv.COLOR_BGR2Lab)
        channelL = imgLab[:, :, 0]

        imgLabResized = cv.cvtColor(cv.resize(imgNormalized, (224, 224)), cv.COLOR_BGR2Lab)
        channelLResized = imgLabResized[:, :, 0]
        channelLResized -= 50

        self.colorModel.setInput(cv.dnn.blobFromImage(channelLResized))
        result = self.colorModel.forward()[0, :, :, :].transpose((1, 2, 0))

        resultResized = cv.resize(result, (self.width, self.height))

        self.result = np.concatenate((channelL[:, :, np.newaxis], resultResized), axis=2)
        self.result = np.clip(cv.cvtColor(self.result, cv.COLOR_Lab2BGR), 0, 1)
        self.result = np.array((self.result)*255, dtype=np.uint8)
