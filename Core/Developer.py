import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from .Utilities import Utility
from .Colorizer import Colorizer

class Developer:
    def __init__(self, image = '', colorize = False, displayPlot = False):
        self.image = image
        self.colorize = colorize
        self.displayPlot = displayPlot
        self.utility = Utility()
        self.shown = False
        self.iteration = 0

    def develop(self, image = ''):
        if image != '':
            self.image = image
        img = cv.imread(self.image)
        img = cv.bitwise_not(img)

        blackAndWhitePhoto = img

        height, width, channels = img.shape
        blue = np.zeros((height, width, channels), np.uint8)
        alpha = 0.8
        blue[:, :, 0] = 216 * alpha
        blue[:, :, 1] = 133 * alpha
        blue[:, :, 2] = 66 * alpha

        img = cv.subtract(img, blue)


        img = self.utility.autoColorCorrect(img)
        img = self.utility.autoEqualize(img)

        cuttOffs = []

        blackAndWhite = False

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])
            hist /= 20
            x = np.argmin(hist)
            y = np.max(hist)
            cuttOffs.append(x)
            if self.displayPlot:
                plt.plot(hist, color=col)
                plt.scatter(x, 100, color=col)
                plt.xlim([0, 256])
                plt.ylim([0, 256])
        if self.displayPlot:
            plt.grid(True)
            plt.show()

        for i in cuttOffs:
            if i > 3:
                blackAndWhite = True
                blackAndWhitePhoto = self.utility.blackAndWhitePhotos(blackAndWhitePhoto)
                blackAndWhitePhoto = cv.cvtColor(blackAndWhitePhoto, cv.COLOR_GRAY2BGR)
                break

        img = self.utility.adjustLevels(img, (0, 0, 0), (255, 255, 255), 1.5)
        blackAndWhitePhoto = self.utility.adjustLevels(blackAndWhitePhoto, (0, 0, 0), (255, 255, 255), 4.5)
        img = self.utility.temperature(img, 10)
        if not blackAndWhite:
            self.result = img
        else:
            self.result = blackAndWhitePhoto

        self.blackAndWhite = blackAndWhite

    def show(self):
        self.develop()
        if self.blackAndWhite and self.colorize:
            colorizer = Colorizer(self.result)
            self.result = colorizer.processImage()
        cv.imshow('Developed Image', self.result)
        cv.waitKey(0)
        cv.destroyAllWindows()
        self.shown = True

    def write(self, image='', path = "./DevelopedImages"):
        if not self.shown:
            self.develop(image)
            if self.blackAndWhite and self.colorize:
                colorizer = Colorizer(self.result)
                self.result = colorizer.processImage()
        if not os.path.exists(path):
            os.mkdir(path)
        print(f"Wrote to {self.image} to {path}/developedPhoto{self.iteration}.jpg")
        cv.imwrite(f'DevelopedImages/developedPhoto{self.iteration}.jpg', self.result)
        self.iteration += 1