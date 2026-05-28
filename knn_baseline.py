import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data(file_path):
    """
    Loads the processed image data from a pickle file.
    The file is in the format of 32x32x3 pixel arrays stored in a  dictionary.
    """
    with open(file_path, 'rb') as f: # rb means 'read binary', which is necessary for pickle files.
        data_dict = pickle.load(f, encoding='latin1')
        X = data_dict['data'] # X is the images' dictionary.
        Y = data_dict['labels'] # Y is the labels dictionary (dog or road).
        y_bin = np.array([1 if label == 'dog' else 0 for label in Y]) # Binary Classification: 1 for Dog, 0 for Road
        return X, y_bin

X, Y = load_data('dogs_v_roads')

# Splitting the data: 80% for training, 20% for testing.
# The random_state=42 ensures that the shuffling is consistent for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# n_neighbors=5: The model decides the result based on the 5 nearest neighbors in the feature space. When K=57, there is a slight boost
# in accuracy (+1.5%), but due to the risk of underfitting and high time complexity, k=5 is chosen as our K.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train) # Training the data and remembering the results.
y_prediction = knn.predict(X_test) # Model being tested on unseen data.

#Showing the data.
accuracy = accuracy_score(y_test, y_prediction)
print(f"KNN Baseline Accuracy: {accuracy * 100:.2f}%")
cm = confusion_matrix(y_test, y_prediction)
print("Confusion Matrix:")
print(cm)