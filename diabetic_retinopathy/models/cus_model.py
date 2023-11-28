import gin

from keras import layers, models

@gin.configurable
class SimpleCNN(models.Model):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        # Convolutional Layer
        self.conv1 = layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape)
        self.pool1 = layers.MaxPooling2D((2, 2))
        # Flatten Layer
        self.flatten = layers.Flatten()
        # Dense Layer with Dropout
        self.dense1 = layers.Dense(32, activation='relu')
        self.dropout = layers.Dropout(0.5)
        # Output Layer
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def __call__(self, x):
        x = self.pool1(self.conv1(x))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)

# Example Usage
input_shape = (256, 256, 3)  # Adapt to your image dimensions
num_classes = 2  # Adapt to the number of classes in IDRID dataset

model = SimpleCNN(input_shape=input_shape, num_classes=num_classes)
model.build((None, *input_shape))
model.summary()