import cv2 as cv
import numpy as np

class Utility:
    def temperature(self, img, beta):
        adjustedImg = img
        b, g, r = cv.split(img)

        if beta > 0:
            r = cv.add(r, beta)
            b = cv.subtract(b, beta)
        else:
            r  = cv.subtract(r, -beta)
            b = cv.add(b, -beta)

        r = np.clip(r, 0, 255)
        b = np.clip(b, 0, 255)

        adjustedImg = cv.merge([b, g, r])

        return adjustedImg

    def tint(self, img, beta):
        adjustedImg = img
        b, g, r = cv.split(img)

        if beta > 0:
            g = cv.add(g, beta)
        else:
            g = cv.subtract(g, -beta)
        
        g = np.clip(g, 0, 255)

        adjustedImg = cv.merge([b, g, r])
        return adjustedImg

    def exposure(self, img, beta):
        adjustedImg = img

        imgFloat = img.astype(np.float32)
        adjustedImg = imgFloat * beta
        adjustedImg = np.clip(adjustedImg, 0, 255).astype(np.uint8)

        return adjustedImg

    def adjustLevels(self, image, inBlack, inWhite, gamma):
        adjustedImage = np.zeros_like(image)
        for channel in range(image.shape[2]):
            ib = inBlack[channel] / 255.0
            iw = inWhite[channel] / 255.0

            lut = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                normalized = i / 255.0
                corrected = (normalized - ib) / (iw - ib)
                gammaCorrected = np.power(corrected, gamma)
                lut[i] = np.clip(gammaCorrected * 255.0, 0, 255)

            adjustedImage[:, :, channel] = cv.LUT(image[:, :, channel], lut)

        return adjustedImage

    def autoColorCorrect(self, image):
        result = cv.cvtColor(image, cv.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
        return result

    def autoEqualize(self, image):
        b, g, r, = cv.split(image)
        b = cv.equalizeHist(b)
        g = cv.equalizeHist(g)
        r = cv.equalizeHist(r)

        result = cv.merge([b, g, r])
        return result

    def blackAndWhitePhotos(self, image):
        result = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return result