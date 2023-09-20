import sys

from loguru import logger

sys.path.append(".")

from util import load_image, visualize, save_image
import matplotlib
import matplotlib.pyplot as plt
from packed_model import Model, TransferEvent, prepare_models

matplotlib.use('module://backend_interagg')
image_path = './data/content/unsplash-rDEOVtE7vOs.jpg'

original_image = load_image(image_path)

plt.figure(figsize=(10, 10), dpi=30)
visualize(original_image[0])
plt.show()

prepare_models()
model = Model()

for ev in model.transfer(image_path, 26, True, False):
    if isinstance(ev, TransferEvent):
        logger.info("Transfer event: {}".format(ev.type))
        save_image(ev.data, f"{ev.type}.png")
        plt.figure(figsize=(10, 10), dpi=30)
        visualize(ev.data)
        plt.show()
