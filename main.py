# This an example on what you can do with this library for now

from Core import Developer

path = input("Path to negative: ")
colorizeImg = input("Colorize image (if it's black and white) (y/n)? ")
colorizeImg = True if colorizeImg.lower == 'y' else False

developer = Developer(path, colorize=colorizeImg)
developer.write()
developer.show()