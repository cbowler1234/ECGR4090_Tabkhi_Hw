import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
dir = "/home/connor/Downloads/yolov5-master/data/images/"
imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg','birds.jpg','bird.jpg','boat.jpg','butterfly.jpg','car.jpg','earth.jpg','heart.jpg','leaf.jpg','man.jpg','monkey.jpg','phone.jpg','shoe.jpg',"sun.jpg",'water.jpg')]  # batched list of images

# Inference
import time

start = time.time()

results = model(imgs)

stop = time.time()

duration = stop - start
print(duration)
results.print()  # or .show(), .save()
