import os
import matplotlib.pyplot as plt
import json
from celluloid import Camera
import numpy as np

data = None
meAbsPath = os.path.dirname(os.path.realpath(__file__))
with open(meAbsPath+"/history.json", "r") as f:
    data = json.load(f)

fig = plt.figure()
camera = Camera(fig)

for i in range(5):
    
    j = str(i)
    bw = [x["bw"] for x in data[j]["population"].values()]
    gain = [x["gain"] for x in data[j]["population"].values()]
    t = plt.scatter(np.array(bw), np.array(gain))
    # plt.legend(t, [f'line {i}'])
    plt.show()
    camera.snap()
    
animation = camera.animate(interval=1000 , blit=True)
animation.save('celluloid_legends.gif', writer = 'imagemagick')

# import winsound
# duration = 10000  # milliseconds
# freq = 1000  # Hz
# winsound.Beep(freq, duration)
