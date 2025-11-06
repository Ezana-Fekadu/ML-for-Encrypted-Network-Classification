import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the network security dataset
    """
    df = pd.read_csv(file_path)

    numerical_features = [
        'packet_size', 'inter_arrival_time',
        'packet_count_5s', 'spectral_entropy', 'frequency_band_energy'
    ]

    boolean_features = [
        'protocol_type_TCP', 'protocol_type_UDP',
        'tcp_flags_FIN', 'tcp_flags_SYN', 'tcp_flags_SYN-ACK'
    ]

    target = 'label'

    X_numerical = df[numerical_features].values
    X_boolean = df[boolean_features].values.astype(np.float32)
    y = df[target].values

    print(f"Dataset shape: {X_numerical.shape}")
    print(f"Boolean features shape: {X_boolean.shape}")
    print(f"Labels distribution:\n{pd.Series(y).value_counts()}")

    return X_numerical, X_boolean, y, numerical_features, boolean_features


def create_sequences_with_proper_splitting(file_path, sequence_length=10, test_size=0.15, val_size=0.15):
    """
    Create sequences AFTER splitting to avoid data leakage
    """
    # Load data first
    X_numerical, X_boolean, y, numerical_features, boolean_features = load_and_preprocess_data(file_path)

    # First split the data temporally (respect time order)
    total_samples = len(X_numerical)
    train_end = int(total_samples * (1 - test_size - val_size))
    val_end = int(total_samples * (1 - test_size))

    # Split indices
    train_indices = range(0, train_end)
    val_indices = range(train_end, val_end)
    test_indices = range(val_end, total_samples)

    # Create scaler ONLY on training data
    scaler = StandardScaler()
    X_numerical_train_scaled = scaler.fit_transform(X_numerical[train_indices])
    X_numerical_val_scaled = scaler.transform(X_numerical[val_indices])
    X_numerical_test_scaled = scaler.transform(X_numerical[test_indices])

    # Combine features for each split
    def combine_features(num_data, bool_data):
        return np.concatenate([num_data, bool_data], axis=1)

    X_train_combined = combine_features(X_numerical_train_scaled, X_boolean[train_indices])
    X_val_combined = combine_features(X_numerical_val_scaled, X_boolean[val_indices])
    X_test_combined = combine_features(X_numerical_test_scaled, X_boolean[test_indices])

    y_train = y[train_indices]
    y_val = y[val_indices]
    y_test = y[test_indices]

    # Create sequences for each split
    def create_sequences_from_data(X_data, y_data, seq_length):
        X_sequences = []
        y_sequences = []
        for i in range(len(X_data) - seq_length):
            X_sequences.append(X_data[i:i + seq_length])
            y_sequences.append(y_data[i + seq_length])
        return np.array(X_sequences), np.array(y_sequences)

    X_train_seq, y_train_seq = create_sequences_from_data(X_train_combined, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences_from_data(X_val_combined, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences_from_data(X_test_combined, y_test, sequence_length)

    print(f"Training sequences: {X_train_seq.shape}")
    print(f"Validation sequences: {X_val_seq.shape}")
    print(f"Test sequences: {X_test_seq.shape}")

    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq), scaler


def create_simpler_rnn_model(input_shape, units=64, dropout_rate=0.3):
    """
    Create a simpler RNN model to avoid overfitting
    """
    model = tf.keras.Sequential()

    # Input layer
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Single LSTM layer with regularization
    model.add(tf.keras.layers.LSTM(
        units,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate * 0.5,
        return_sequences=False
    ))

    # Add batch normalization
    model.add(tf.keras.layers.BatchNormalization())

    # Dense layers with regularization
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate * 0.5))

    # Output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return model


def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced datasets
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    print(f"Class weights: {class_weight_dict}")

    return class_weight_dict


def train_rnn_model(model, X_train, y_train, X_val, y_val, class_weights, epochs=50, batch_size=32):
    """
    Train the RNN model with proper validation
    """
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'Analysis/RecurrentNeuralNetworkHistory/best_rnn_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    return history


def evaluate_rnn_model(model, X_test, y_test):
    """
    Evaluate the RNN model
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n=== RNN MODEL EVALUATION ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Classification report
    print(f"\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Malicious'], zero_division=0))

    # Confusion matrix
    print(f"\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return accuracy, precision, recall, f1, y_pred


def plot_training_history(history):
    """
    Plot training history
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()

    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()

    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('rnn_training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved as 'rnn_training_history.png'")

    plt.close('all')


def main():
    """
    Main function to run RNN training and evaluation
    """
    # Simpler configuration
    file_path = "/home/unbreakaskull/Downloads/embedded_system_network_security_dataset.csv"
    sequence_length = 10
    rnn_units = 64
    dropout_rate = 0.3
    batch_size = 32
    epochs = 150

    print("=== Fixed RNN for Network Intrusion Detection ===")

    try:
        # Prepare datasets with proper splitting
        print("\n1. Preparing RNN datasets with temporal splitting...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = create_sequences_with_proper_splitting(
            file_path, sequence_length
        )

        # Calculate class weights
        print("\n2. Calculating class weights...")
        class_weights = calculate_class_weights(y_train)

        # Create simpler model
        print("\n3. Creating RNN model...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_simpler_rnn_model(
            input_shape=input_shape,
            units=rnn_units,
            dropout_rate=dropout_rate
        )

        print(f"\nModel Summary:")
        model.summary()
        print(f"Total Parameters: {model.count_params():,}")

        # Train model
        print(f"\n4. Training RNN model for {epochs} epochs...")
        history = train_rnn_model(
            model, X_train, y_train, X_val, y_val, class_weights, epochs, batch_size
        )

        # Evaluate model
        print("\n5. Evaluating RNN model...")
        accuracy, precision, recall, f1, y_pred = evaluate_rnn_model(model, X_test, y_test)

        # Plot training history
        print("\n6. Plotting training history...")
        plot_training_history(history)

        # Save final model
        model.save('final_rnn_model.keras')
        print("\n7. Model saved as 'final_rnn_model.keras'")

        # Save preprocessing objects
        joblib.dump(scaler, 'rnn_scaler.pkl')
        print("Scaler saved as 'rnn_scaler.pkl'")

        # Final summary
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"Best Validation Loss: {min(history.history['val_loss']):.4f}")
        print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"Final Test F1-Score: {f1:.4f}")

        # Test 10 values from test set
        print("\n8. Testing model on 10 sequences from test set...")
        test_model_on_sequences(model, X_test, y_test, num_sequences=10)

        # Demonstrate prediction function with new data
        print("\n9. Demonstrating prediction function with new sequences...")
        demonstrate_prediction_function(model, scaler, file_path, sequence_length)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_model_on_sequences(model, X_test, y_test, num_sequences=10):
    """
    Test the model on specific sequences from the test set
    """
    print(f"Testing on {num_sequences} sequences from test set:")
    print("-" * 60)

    correct_predictions = 0
    total_predictions = min(num_sequences, len(X_test))

    for i in range(total_predictions):
        # Get one sequence from the test set
        X_demo = X_test[i:i + 1]
        y_actual = y_test[i]

        # Make prediction
        prediction_proba = model.predict(X_demo, verbose=0)[0][0]
        predicted_label = 1 if prediction_proba > 0.5 else 0
        confidence = prediction_proba if predicted_label == 1 else 1 - prediction_proba

        is_correct = y_actual == predicted_label
        if is_correct:
            correct_predictions += 1

        print(f"Sequence {i + 1}:")
        print(f"  Actual: {y_actual} | Predicted: {predicted_label}")
        print(f"  Confidence: {confidence:.4f} | Raw output: {prediction_proba:.4f}")
        print(f"  Correct: {'✓' if is_correct else '✗'}")
        print()

    accuracy = correct_predictions / total_predictions
    print(
        f"Test Accuracy on {total_predictions} sequences: {accuracy:.4f} ({correct_predictions}/{total_predictions} correct)")


def demonstrate_prediction_function(model, scaler, file_path, sequence_length):
    """
    Demonstrate the predict_with_rnn function with proper sequences
    """
    # Load the full dataset for prediction demo
    test_df = pd.read_csv(file_path)

    # Test on 10 different sequences from the dataset
    print("Testing predict_with_rnn function on 10 sequences:")
    print("-" * 50)

    correct_predictions = 0
    total_predictions = 10

    for i in range(total_predictions):
        # Get a random sequence from the dataset (avoid data leakage by using later data)
        start_idx = len(test_df) - sequence_length - (i * 5)  # Space out the sequences
        if start_idx < sequence_length:
            start_idx = sequence_length

        demo_data = test_df.iloc[start_idx - sequence_length:start_idx]

        prediction = predict_with_rnn_loaded(model, scaler, demo_data, sequence_length)

        if prediction is not None:
            # The prediction corresponds to the last row in the sequence
            actual_label = demo_data.iloc[-1]['label']
            predicted_label = 1 if prediction > 0.5 else 0
            confidence = prediction if predicted_label == 1 else 1 - prediction

            is_correct = actual_label == predicted_label
            if is_correct:
                correct_predictions += 1

            print(f"Sequence {i + 1}:")
            print(f"  Actual: {actual_label} | Predicted: {predicted_label}")
            print(f"  Confidence: {confidence:.4f} | Raw output: {prediction:.4f}")
            print(f"  Correct: {'✓' if is_correct else '✗'}")
            print()

    accuracy = correct_predictions / total_predictions
    print(f"Prediction Function Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions} correct)")


def predict_with_rnn(model_path, scaler_path, new_data, sequence_length=10):
    """
    Make predictions using the trained RNN model (loads model from file)
    """
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return predict_with_rnn_loaded(model, scaler, new_data, sequence_length)


def predict_with_rnn_loaded(model, scaler, new_data, sequence_length=10):
    """
    Make predictions using already loaded model and scaler
    """
    numerical_features = [
        'packet_size', 'inter_arrival_time',
        'packet_count_5s', 'spectral_entropy', 'frequency_band_energy'
    ]

    boolean_features = [
        'protocol_type_TCP', 'protocol_type_UDP',
        'tcp_flags_FIN', 'tcp_flags_SYN', 'tcp_flags_SYN-ACK'
    ]

    X_numerical = new_data[numerical_features].values
    X_boolean = new_data[boolean_features].values.astype(np.float32)
    X_numerical_scaled = scaler.transform(X_numerical)
    X_combined = np.concatenate([X_numerical_scaled, X_boolean], axis=1)

    if len(X_combined) >= sequence_length:
        X_sequence = X_combined[-sequence_length:].reshape(1, sequence_length, -1)
        prediction = model.predict(X_sequence, verbose=0)
        return prediction[0][0]
    else:
        print(f"Need at least {sequence_length} samples for prediction")
        return None


if __name__ == "__main__":
    main()