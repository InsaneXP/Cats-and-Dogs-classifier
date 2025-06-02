import cv2 

def load_and_preprocess_image(path, target_size=(224, 224)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    return img

import matplotlib.pyplot as plt  

def show_image(img_tensor, label=None):
    img = img_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img)
    if label is not None:
        plt.title(label)
    plt.axis('off')
    plt.show()
