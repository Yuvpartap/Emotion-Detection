from deepface import DeepFace
from PIL import Image
import cv2
import matplotlib.pyplot as plt
img_path = r'E:\Chandigarh Uni\FAI discussion\Exp-33\OIP'
img = cv2.imread(img_path)
plt.imshow(img[:, :, ::-1])
image = Image.open(img_path)
image.show()
demography = DeepFace.analyze(img_path)
print("Age: ", demography["age"])
print("Emotion: ", demography["dominant_emotion"])
