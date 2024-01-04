import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, num_classes, labels, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.labels = labels
        self.confusion_matrix = self.add_weight(
            name="confusion_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros",
            dtype=tf.int32,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, tf.shape(y_true))

        new_confusion_matrix = tf.math.confusion_matrix(
            y_true, y_pred, self.num_classes, weights=sample_weight, dtype=tf.int32
        )
        self.confusion_matrix.assign_add(tf.cast(new_confusion_matrix, tf.int32))

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def result(self):
        return self.confusion_matrix

    def summary(self):
        # Calculate precision, sensitivity, specificity
        cm = self.confusion_matrix.numpy()
        data = []

        for i in range(self.num_classes):
            TP = cm[i, i]
            FP = np.sum(cm[i, :]) - TP
            FN = np.sum(cm[:, i]) - TP
            TN = np.sum(cm) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.0
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.0
            F1 = round(2 * (Precision * Recall) / (Precision + Recall), 3) if TP + FN != 0 else 0.0
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.0
            data.append([self.labels[i], Precision, Recall, Specificity,F1])

        df = pd.DataFrame(data, columns=["Label", "Precision", "Recall", "Specificity", "F1_Score"])
        print(df)

    def plot(self):
        cm = self.confusion_matrix.numpy()
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        plt.yticks(range(self.num_classes), self.labels)
        plt.colorbar()
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("Confusion matrix")

        thresh = cm.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info = int(cm[y, x])
                plt.text(
                    x,
                    y,
                    info,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="white" if info > thresh else "black",
                )
        plt.tight_layout()
        plt.show()
