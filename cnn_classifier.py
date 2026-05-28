import numpy as np
import pickle
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Loading the data from the pickle file.
with open('dogs_v_roads', 'rb') as f:
    data_dict = pickle.load(f)
X = data_dict['data']
Y = data_dict['labels']

# Converting text labels into a Binary Classification: 1 if Dog, 0 if Road.
y_bin = np.array([1 if label == 'dog' else 0 for label in Y])

# Reshaping the data back into 32x32x3 pixels. We also normalize the values to allow the model to learn faster (from 0-255 to 0-1).
X_squares = X.reshape(-1, 32, 32, 3)
X_squares = X_squares / 255.0

# Splitting the data into 80% for training, 20% for testing.
x_train, x_test, y_train, y_test = train_test_split(X_squares, y_bin, test_size=0.2)

# Building the model. Relu was used for speed and prioritizing positive weights.
model = models.Sequential([
    # Rotating the data so the model learns rather than memorizing the dataset.
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),

    # 2 layers of 64 and 128 filters were used for feature extraction. With MaxPooling, the feature map becomes smaller,
    # reducing computation size and focusing on the important features.
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'), layers.MaxPooling2D((2, 2)),

    # Flattens the final feature map into a vector, which inturn becomes a layer of neurons.
    layers.Flatten(),

    # 64 neurons summarize the data found in the previous layers. Used to make to finalize the result.
    layers.Dense(64, activation='relu'),

    # 1 neuron for the final result. Uses sigmoid, causing the result to be between 0 and 1.
    layers.Dense(1, activation='sigmoid')])


# Compiling and training the model.
model.compile(
    optimizer='adam', # Adaptive optimizer
    loss='binary_crossentropy', # Loss function used for binary cases.
    metrics=['accuracy'])

# Training with 30 epochs with a batch size of 16.
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(x_test, y_test) # testing
)

# Evaluating and Visualizing results
def plot_history(history):
    """
    Generates graph of training and validation accuracy.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_history(history)

# Using a .5 threshold. 0.5> is a dog, and 0.5< is a road.
y_prediction = (model.predict(x_test) > 0.5).astype("int32")

# Creating a confusion matrix.
cm = confusion_matrix(y_test, y_prediction)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()