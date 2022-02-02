# Importamos el conjunto de datos
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
# Mostramos el f1_score resultante de la clasificación
from sklearn.metrics import f1_score

class ImageRecognition:

    def recon(self):

        # Añadimos as_frame=False para forzar la devolución de un array
        mnist = fetch_openml('mnist_784', as_frame=False)

        plt.figure(figsize=(20, 4))

        for index, img in zip(range(1, 9), mnist.data[:8]):
            plt.subplot(1, 8, index)
            plt.imshow(np.reshape(img, (28, 28)), cmap=plt.cm.gray)
            plt.title('Ejemplo: {}'.format(index))
        plt.show()

        # Visualización de datos
        df = pd.DataFrame(mnist.data)
        df

        # Division del subconjunto
        X_train, X_test, Y_train, Y_test = train_test_split(mnist.data, mnist.target, test_size=0.25)

        # Entrenamiento del algoritmo
        clf = Perceptron(max_iter=2000, random_state=40, n_jobs=-1)
        clf.fit(X_train, Y_train)

        # Predicción del conjunto de pruebas
        y_prend = clf.predict(X_test)
        y_prend

        f1_score(Y_test, y_prend, average="weighted")

        # Mostrando imagenes mal clasificadas
        index = 0
        index_errors = []

        for etiqueta, predict in zip(Y_test, y_prend):
            if etiqueta != predict:
                index_errors.append(index)
            index += 1

        plt.figure(figsize=(20, 4))

        for index, img_index in zip(range(1, 9), index_errors[8:16]):
            plt.subplot(1, 8, index)
            plt.imshow(np.reshape(X_test[img_index], (28, 28)), cmap=plt.cm.gray)
            plt.title('Orig:' + str(Y_test[img_index]) + ' Pred:' + str(y_prend[img_index]))
        plt.show()