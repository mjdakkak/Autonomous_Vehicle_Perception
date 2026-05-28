**-Computer Vision for Autonomous Safety-**



This project implements two distinct machine learning architectures to solve a critical safety problem: distinguishing between a clear road and a potential hazard (a dog). It demonstrates a complete Computer Vision (CV) pipeline, from raw image processing to deep learning optimization.

Key Takeaway: Successfully evolved from a 79 to 81% K-Nearest Neighbors (KNN) baseline to a 98% accuracy Convolutional Neural Network (CNN), essentially minimizing high risk false positives.





**The Models**



1. K-Nearest Neighbors (Baseline Classifier)

The goal was to establish a baseline using classical "Lazy Learning" algorithm. The accuracy obtained was \~79 to 81%. The model relies on raw pixel distance. While computational simple, it suffered from high spatial complexity and struggled with different textures, mixing up tree-lined roads with dogs due to similar color distributions.



2\. Convolutional Neural Network

The goal was to implement a deep learning algorithm capable of feature extraction. The average accuracy reached was 98%, with the best run being 99.5% accurate. This was achieved using 30 epochs. By utilizing 64 and 128 filter convolutional layers, the model picked up on geometric shapes (like ears and tails) rather than pixel colors. This transition drastically removed hallucinated hazards.





**Engineering Pipeline**



To reach a stable 98% performance, I developed a robust image and training pipeline.

1. Normalization: Scaled RGB pixel values from \[0,255] to \[0,1] to ensure faster gradient descent convergence.
2. Data Augmentation: Implemented RandomFlip and RandomRotation(0,1) layers. This forced the model to learn the features, preventing the memorization of the training data.
3. Feature Extraction: Utilized two Conv2D layers paired with MaxPooling2D to reduce spatial dimensions while identifying high-level features.
4. Batch Optimization: Tuned Batch Size to 16. This introduced enough random noise to help the Adam optimizer escape local minimum and find a superior global optimum.
5. Sigmoid Activation: Employed a single neuron output with a 0.5 probability threshold for high precision binary classification.





**Evaluation**



Model performance was validated using Learning Curves and Confusion Matrices.



* Generalization: The Accuracy/Loss plots showed the training and validation lines being close together, confirming the model is not overfitted and generalizes well to unseen data.
* Safety Priority: In the 99.5% run, the model achieved a near perfect recall for the "Dog" class. For an autonomous driving application, minimizing False Negatives (missing a dog) was prioritized over minimizing False Positives.
* Bias Mitigation: Analysis of the Confusion Matrix showed that increasing the filter count from 64 to 128 successfully resolved early biases where wooded road backgrounds were triggering false alarms.





**Lessons Learned and Limitations**



* Resolution Constraints: At 32x32 pixels, some images are naturally ambiguous. This creates a "theoretical ceiling" where even a human would struggle to achieve 100% accuracy.
* Environmental Noise: Early iterations showed a bias toward "Roads with Trees." Using Data Augmentation helped the model distinguish between organic tree textures and dog fur.
* Convergence: I found that 30 Epochs provided the ideal balance. Extending to 40+ epochs showed signs of diminishing returns and a slight rise in validation loss.





**Resources Used**



Language: Python

Libraries: tensorflow/keras, numpy, matplotlib, scikit-learn, seaborn, pickle

Architecture: CNN (Sequential)

Optimizer: Adam (Adaptive Moment Estimation)




