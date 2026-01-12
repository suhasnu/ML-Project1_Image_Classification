import sys
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# We use RandomOverSampler to increase the count of minority classes
from imblearn.over_sampling import RandomOverSampler 

def load_data(filename):
    """
    Loads data and performs Random Oversampling to balance classes.
    """
    data = np.load(filename)
    images, labels = data['images'], data['labels']
    
    # 1. Flatten images (N, 28, 28) -> (N, 784)
    # The sampler requires 2D input (rows, cols), not 3D images
    N, h, w = images.shape
    images_flat = images.reshape(N, h * w)
    
    # 2. Split Data (Train vs Test)
    # IMPORTANT: Split BEFORE oversampling to prevent data leakage!
    X_train_flat, X_test_flat, y_train, y_test = train_test_split(
        images_flat, labels, test_size=0.2, random_state=42
    )
    
    # 3. Apply Oversampling
    print(f"Original Class 5 count: {sum(y_train == 5)}")
    
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_flat, y_train)
    
    print(f"New Class 5 count:      {sum(y_train_res == 5)}")
    
    # 4. Reshape back to 3D Images and Normalize (0-1)
    # Shape becomes (N, 28, 28, 1)
    X_train = X_train_res.reshape(-1, h, w, 1).astype('float32') / 255.
    X_test = X_test_flat.reshape(-1, h, w, 1).astype('float32') / 255.
    
    return X_train, X_test, y_train_res, y_test


def build_model():
    """
    Defines the Convolutional Neural Network Architecture.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28,1)),
        
        # Feature Extraction (Convolution + Pooling)
        tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((2,2)),
        
        tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D((2,2)),
        
        tf.keras.layers.Flatten(),
        
        # Classification (Dense Layers)
        tf.keras.layers.Dropout(0.5), # Prevents overfitting
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax") # Output: 10 probabilities
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=["accuracy"])
    return model


def main():
    # 1. Argument Check
    if len(sys.argv) < 3:
        sys.exit("Usage: python CNN.py <data.npz> <train/test>")
    
    npz_file = sys.argv[1]
    mode = sys.argv[2]
    
    # 2. Load and Balance Data
    X_train, X_test, y_train, y_test = load_data(npz_file)
    
    # 3. Train Mode
    if mode == "train":
        model = build_model()
        
        # No 'class_weight' needed because we physically balanced the data!
        model.fit(X_train, y_train, 
                  epochs=15, 
                  batch_size=32, 
                  validation_data=(X_test, y_test))
        
        model.save("my_model.keras")
        print("Training complete. Model saved!")
        
    # 4. Test Mode
    else:
        model = tf.keras.models.load_model("my_model.keras")
        
        # Generate Predictions
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Print Text Report (Check Recall!)
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred))
        
        # Generate Heatmap
        # normalize='true' turns counts into percentages (0.00 - 1.00)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
        plt.title("Confusion Matrix (Accuracy %)")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.show()


if __name__ == "__main__":
    main()