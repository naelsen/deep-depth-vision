# Pour créer nos modèles
import tensorflow as tf
from tensorflow import keras

# Pour créer nos jeu d'entraînement et de test
#from sklearn.model_selection import train_test_split

# Pour les calculs rapides
import numpy as np

# Pour gérer les images
import cv2 as cv

# Pour gérer les fichiers
#import glob

# Pour plot rapidement des trucs
import matplotlib.pyplot as plt


def myAcc(Y_true, Y_pred):
    diff = keras.backend.abs(Y_true - Y_pred)
    correct = keras.backend.less(diff, 0.2)
    return keras.backend.mean(correct)

def create_model():
    DROP = 0.25
    activation = "relu"
    x_list = list()

    input = keras.layers.Input(shape=(480, 640, 3))
    MP = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")
    US = keras.layers.UpSampling2D((2, 2))

    down = False
    i = 0
    for N in [4, 8, 16, 32, 64, 128, 64, 32, 16, 8, 4]:
        if N == 128:
            down = True
        if not down:
            i += 1
            if N == 4:
                x = keras.layers.Conv2D(N ,kernel_size=(3, 3), activation=activation, padding="same")(input)
            else:
                x = keras.layers.Conv2D(N, kernel_size=(3, 3), activation=activation, padding="same")(x_list[-1][1])
            x = keras.layers.Conv2D(N, kernel_size=(3, 3), activation=activation, padding="same")(x)
            x_out = keras.layers.Dropout(DROP)(x)
            x_out = MP(x_out)
            x_list.append((x, x_out))
        else:
            if N == 128:
                x = keras.layers.SeparableConv2D(N, kernel_size=(3, 3), activation=activation, padding="same")(x_list[-1][1])
                x = keras.layers.SeparableConv2D(N, kernel_size=(3, 3), activation=activation, padding="same")(x)
            else:
                i -= 1
                x = keras.layers.Conv2D(N, kernel_size=(3, 3), activation=activation, padding="same")(x)
                x = keras.layers.Concatenate()([x_list[i][0], x])
                x = keras.layers.Conv2D(N, kernel_size=(3, 3), activation=activation, padding="same")(x)
            if N != 4:
                x = keras.layers.Dropout(DROP)(x)
                x = US(x)

    x = keras.layers.Conv2D(1, kernel_size=(1, 1), activation=activation, padding='same')(x)
    x = keras.layers.Reshape((480,640))(x)

    LEARNING_RATE = 0.001
    BETA_1 = 0.9
    BETA_2 = 0.99
    EPSILON = 0.1
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)

    model = keras.Model(inputs=input, outputs=x)
    model.compile(
        loss="mean_squared_error",
        optimizer=opt,
        metrics=[myAcc])

    #model.summary()
    return model

def to_img(img):
	mean_img = np.mean(img)
	std_img = np.std(img)
	img = (img - mean_img)/std_img
	min_img = np.min(img)
	img = img - min_img
	max_img = np.max(img)
	img = img*255/max_img
	return img.astype("uint8")

def draw_target(img):
    h, w = img.shape[:2]
    y_target = int(h/2)
    x_target = int(w/2)
    cv.line(img, pt1=(0, y_target), pt2=(w ,y_target), color=(0,255,0), thickness=2)
    cv.line(img, pt1=(x_target, 0), pt2=(x_target, h), color=(0,255,0), thickness=2)
    cv.circle(img, center=(x_target, y_target), radius=10, color=(0,0,255), thickness=2)
    return(img)

model = create_model()
#model.summary()
model.load_weights("weight.h5")

cam = cv.VideoCapture(0)

while True:
    # ===== Get frame
    isTrue, original_frame = cam.read()
    original_frame = cv.flip(original_frame, 1)
    depth = model.predict(np.array([original_frame]))
    depth = depth[0]
    original_frame = draw_target(original_frame)
    h, w = depth.shape[:2]
    y_target = h//2
    x_target = w//2
    cv.putText(original_frame, "Depth : " + str(depth[x_target, y_target]), (20,40), cv.FONT_HERSHEY_SIMPLEX, 1.2,(255, 0, 0), 2)
    depth = to_img(depth)
    depth = cv.applyColorMap(depth, cv.COLORMAP_MAGMA)
    cv.imshow("Video original", original_frame)
    cv.imshow("Prediction", cv.medianBlur(depth,5))
    if cv.waitKey(100) & 0xFF == ord('q'):
        break
cam.release()
cv.destroyAllWindows()