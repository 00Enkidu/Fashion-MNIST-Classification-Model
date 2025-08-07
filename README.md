# Fashion-MNIST Classification Model

A deep learning project for classifying Fashion-MNIST images, featuring a structured dataset pipeline, a robust CNN model, a Streamlit web application for real-time prediction, and Dockerized deployment for easy reproducibility.

---

## 1. Dataset

Fashion-MNIST is a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes (such as T-shirt/top, Trouser, Pullover, etc).

**Key Points:**
- Images normalized to [0, 1] range.
- Images reshaped to (28, 28, 1) for use with CNNs.

```python
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
```

---

## 2. Model Training

### Model Architecture

A Convolutional Neural Network (CNN) is used:

```python
model = models.Sequential()
# Extracts low-level features (edges, textures)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
# Learns more complex patterns
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Captures higher-level features
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Flattens for dense layers
model.add(layers.Flatten())
# Fully connected layer for classification
model.add(layers.Dense(64, activation='relu'))
# Output layer for 10 classes
model.add(layers.Dense(10))
```

**Compilation and Training:**

```python
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)
history = model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)
```

#### Training Log

```
Epoch 1/10: accuracy: 0.7440 - loss: 0.7052 - val_accuracy: 0.8683 - val_loss: 0.3684
Epoch 2/10: accuracy: 0.8760 - loss: 0.3434 - val_accuracy: 0.8846 - val_loss: 0.3185
Epoch 3/10: accuracy: 0.8945 - loss: 0.2862 - val_accuracy: 0.8922 - val_loss: 0.3009
Epoch 4/10: accuracy: 0.9072 - loss: 0.2524 - val_accuracy: 0.8950 - val_loss: 0.2917
Epoch 5/10: accuracy: 0.9163 - loss: 0.2260 - val_accuracy: 0.8976 - val_loss: 0.2880
Epoch 6/10: accuracy: 0.9255 - loss: 0.2019 - val_accuracy: 0.8969 - val_loss: 0.3031
Epoch 7/10: accuracy: 0.9322 - loss: 0.1835 - val_accuracy: 0.8974 - val_loss: 0.3218
Epoch 8/10: accuracy: 0.9391 - loss: 0.1675 - val_accuracy: 0.8924 - val_loss: 0.3538
Epoch 9/10: accuracy: 0.9423 - loss: 0.1552 - val_accuracy: 0.8964 - val_loss: 0.3515
Epoch 10/10: accuracy: 0.9493 - loss: 0.1372 - val_accuracy: 0.8948 - val_loss: 0.3701
Test accuracy: 0.8948
```

#### Training Result Analysis

- **Accuracy**: The model quickly achieves high accuracy, reaching above 0.9 on the training set by epoch 4, and stabilizes around 0.95 by epoch 10. Validation accuracy reaches approximately 0.89-0.90 and remains stable throughout.
- **Loss**: Training loss consistently decreases, indicating effective learning. Validation loss decreases at first but slightly increases after epoch 5, suggesting mild overfitting.
- **Generalization**: The gap between training and validation accuracy is small, demonstrating good generalization for this model and dataset.
- **Test Performance**: Final test accuracy is 0.8948, showing robust performance on unseen data.

#### Training Curves

<img width="990" height="682" alt="image" src="https://github.com/user-attachments/assets/617080a0-1a3e-47cb-8954-3fa4c2d5b0eb" />

---

## 3. Streamlit Web Application

The project includes a Streamlit-based web frontend for user-friendly, real-time predictions.

**Features:**
- Upload a local image for prediction.
- Real-time inference using the trained model.
- Displays predicted class and probability.

**How to run locally:**

```bash
streamlit run app/main.py
```

---

## 4. Docker Deployment

A Dockerfile is provided for containerized deployment, ensuring reproducibility and easy sharing.

**Build and Run:**

```bash
docker build -t fashion-mnist-app .
docker run -p 8501:8501 fashion-mnist-app
```

- The app will be available at `http://localhost:8501/`

---


## 5. How to Use

1. **Train the model** (optional): See `model_notebook/`.
2. **Run the Streamlit app**:
   - Either locally or via Docker.
3. **Upload an image** and get instant predictions through the web UI.

---

## 6. References

- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)

---

> **All model code, logs, and result plots are based on the original notebook and project files.  
> For any questions or suggestions, please open an issue.**
