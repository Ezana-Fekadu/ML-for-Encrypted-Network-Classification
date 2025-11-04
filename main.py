import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



########################
#K-CROSS-VALIDATION
########################

def kfold_cross_validation(file_path, n_splits=5, epochs=15, batch_size=32, hidden_layers=[128, 64, 32], dropout_rate=0.3):
    """
    Perform k-fold cross-validation on the entire dataset
    """
    # Load and prepare the full dataset
    df = load_and_preprocess_data(file_path)
    X_numerical, X_boolean, y, numerical_features, boolean_features = prepare_features_and_labels(df)

    # Scale numerical features once for the entire dataset
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)

    # Initialize k-fold
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_no = 1
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    histories = []

    # Get input dimensions for model creation
    numerical_dim = X_numerical.shape[1]
    boolean_dim = X_boolean.shape[1]

    for train_idx, val_idx in kfold.split(X_numerical_scaled, y):
        print(f'\n{"=" * 50}')
        print(f'Training Fold {fold_no}/{n_splits}')
        print(f'{"=" * 50}')

        # Split data for this fold
        X_num_train, X_num_val = X_numerical_scaled[train_idx], X_numerical_scaled[val_idx]
        X_bool_train, X_bool_val = X_boolean[train_idx], X_boolean[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create TensorFlow datasets for this fold
        def create_dataset(numerical_data, boolean_data, labels, shuffle=False):
            dataset = tf.data.Dataset.from_tensor_slices((
                {'numerical_features': numerical_data, 'boolean_features': boolean_data},
                labels
            ))
            if shuffle:
                dataset = dataset.shuffle(buffer_size=len(numerical_data))
            return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_dataset = create_dataset(X_num_train, X_bool_train, y_train, shuffle=True)
        val_dataset = create_dataset(X_num_val, X_bool_val, y_val)

        # Calculate class weights for this fold
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        # Create preprocessing results for model creation
        fold_preprocessing_results = {
            'input_dims': {
                'numerical_dim': numerical_dim,
                'boolean_dim': boolean_dim,
                'total_features': numerical_dim + boolean_dim
            }
        }

        # Create and train model for this fold
        model = create_perceptron(fold_preprocessing_results, hidden_layers=hidden_layers, dropout_rate=dropout_rate)

        print(f'Training samples: {len(X_num_train)}, Validation samples: {len(X_num_val)}')

        # Train the model
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            class_weight=class_weight_dict,
            verbose=1
        )
        histories.append(history.history)

        # Evaluate on validation set
        val_predictions = model.predict(val_dataset)
        val_pred_classes = (val_predictions > 0.5).astype(int).flatten()

        # Calculate metrics
        accuracy = accuracy_score(y_val, val_pred_classes)
        precision = precision_score(y_val, val_pred_classes, zero_division=0)
        recall = recall_score(y_val, val_pred_classes, zero_division=0)
        f1 = f1_score(y_val, val_pred_classes, zero_division=0)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        print(f'Fold {fold_no} Results:')
        print(f'  Accuracy:  {accuracy:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall:    {recall:.4f}')
        print(f'  F1-Score:  {f1:.4f}')

        fold_no += 1

    # Print overall cross-validation results
    print(f'\n{"=" * 60}')
    print(f'{n_splits}-FOLD CROSS-VALIDATION RESULTS')
    print(f'{"=" * 60}')
    print(f'Average Accuracy:  {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})')
    print(f'Average Precision: {np.mean(precision_scores):.4f} (+/- {np.std(precision_scores):.4f})')
    print(f'Average Recall:    {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})')
    print(f'Average F1-Score:  {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})')



    return {
        'accuracy_scores': accuracy_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'f1_scores': f1_scores,
        'histories': histories,
        # Add these mean values for easy access:
        'mean_accuracy': np.mean(accuracy_scores),
        'mean_precision': np.mean(precision_scores),
        'mean_recall': np.mean(recall_scores),
        'mean_f1': np.mean(f1_scores),
        # Standard deviations for confidence intervals:
        'std_accuracy': np.std(accuracy_scores),
        'std_precision': np.std(precision_scores),
        'std_recall': np.std(recall_scores),
        'std_f1': np.std(f1_scores)
    }


def train_final_model(preprocessing_results, epochs=15, hidden_layers=[128, 64, 32], dropout_rate=0.3):
    """
    Train a final model on the entire training+validation data using the original split
    """
    print(f'\n{"=" * 50}')
    print('TRAINING FINAL MODEL ON FULL TRAINING DATA')
    print(f'{"=" * 50}')

    # Combine training and validation datasets
    full_train_dataset = preprocessing_results['train_dataset'].concatenate(
        preprocessing_results['val_dataset']
    )

    # Create and train final model
    model = create_perceptron(preprocessing_results, hidden_layers=hidden_layers, dropout_rate=dropout_rate)

    history = model.fit(
        full_train_dataset,
        epochs=epochs,
        class_weight=preprocessing_results['class_weights'],
        verbose=1
    )

    # Evaluate on test set
    print(f'\n{"=" * 50}')
    print('FINAL MODEL TEST EVALUATION')
    print(f'{"=" * 50}')

    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        preprocessing_results['test_dataset'],
        verbose=1
    )

    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

    print(f'\n=== FINAL TEST RESULTS ===')
    print(f'Loss:     {test_loss:.4f}')
    print(f'Accuracy: {test_accuracy:.4f}')
    print(f'Precision: {test_precision:.4f}')
    print(f'Recall:    {test_recall:.4f}')
    print(f'F1-Score:  {test_f1:.4f}')

    return model, history





########################
#TRADITIONAL TRAIN/VALIDATE/TEST
########################
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the network security dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # print(f"Dataset shape: {df.shape}")
    # print(f"Label distribution:\n{df['label'].value_counts()}")

    return df


def prepare_features_and_labels(df):
    """
    Separate features and labels, and handle feature types
    """
    # Identify feature columns (REMOVED src_port, dst_port, mean_packet_size)
    numerical_features = [
        'packet_size', 'inter_arrival_time',
        'packet_count_5s', 'spectral_entropy', 'frequency_band_energy'
        # Removed: 'src_port', 'dst_port', 'mean_packet_size'
    ]

    # Boolean features (REMOVED IP ADDRESS FEATURES)
    boolean_features = [
        'protocol_type_TCP', 'protocol_type_UDP',
        'tcp_flags_FIN', 'tcp_flags_SYN', 'tcp_flags_SYN-ACK'
        # Removed: 'src_ip_192.168.1.2', 'src_ip_192.168.1.3',
        #          'dst_ip_192.168.1.5', 'dst_ip_192.168.1.6'
    ]

    # Target variable
    target = 'label'

    # Extract features and labels
    X_numerical = df[numerical_features].values
    X_boolean = df[boolean_features].values.astype(np.float32)
    y = df[target].values

    # print(f"Numerical features shape: {X_numerical.shape}")
    # print(f"Boolean features shape: {X_boolean.shape}")
    # print(f"Labels shape: {y.shape}")

    return X_numerical, X_boolean, y, numerical_features, boolean_features


def create_tf_datasets(X_numerical, X_boolean, y, batch_size=32, validation_split=0.2, test_split=0.15):
    """
    Create TensorFlow datasets for training, validation, and testing
    """
    # First split: separate test set
    X_num_temp, X_num_test, X_bool_temp, X_bool_test, y_temp, y_test = train_test_split(
        X_numerical, X_boolean, y, test_size=test_split, random_state=42, stratify=y
    )

    # Second split: separate validation from training
    val_size = validation_split / (1 - test_split)
    X_num_train, X_num_val, X_bool_train, X_bool_val, y_train, y_val = train_test_split(
        X_num_temp, X_bool_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)
    X_num_test_scaled = scaler.transform(X_num_test)

    # print(f"Training set: {X_num_train_scaled.shape[0]} samples")
    # print(f"Validation set: {X_num_val_scaled.shape[0]} samples")
    # print(f"Test set: {X_num_test_scaled.shape[0]} samples")

    # Create TensorFlow datasets
    def create_dataset(numerical_data, boolean_data, labels, batch_size, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'numerical_features': numerical_data,
                'boolean_features': boolean_data
            },
            labels
        ))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(numerical_data))

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    train_dataset = create_dataset(X_num_train_scaled, X_bool_train, y_train, batch_size, shuffle=True)
    val_dataset = create_dataset(X_num_val_scaled, X_bool_val, y_val, batch_size)
    test_dataset = create_dataset(X_num_test_scaled, X_bool_test, y_test, batch_size)

    return train_dataset, val_dataset, test_dataset, scaler


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

    # print(f"Class weights: {class_weight_dict}")
    return class_weight_dict


def create_feature_columns(numerical_features, boolean_features):
    """
    Create feature columns for TensorFlow models (useful for some model types)
    """
    feature_columns = []

    # Numerical features
    for feature_name in numerical_features:
        feature_columns.append(
            tf.feature_column.numeric_column(feature_name)
        )

    # Boolean features
    for feature_name in boolean_features:
        feature_columns.append(
            tf.feature_column.numeric_column(feature_name)
        )

    return feature_columns


def main_preprocessing_pipeline(file_path, batch_size=32):
    """
    Complete preprocessing pipeline
    """
    # print("=== Network Security Data Preprocessing ===")

    # 1. Load data
    df = load_and_preprocess_data(file_path)

    # 2. Prepare features and labels
    X_numerical, X_boolean, y, numerical_features, boolean_features = prepare_features_and_labels(df)

    # 3. Calculate class weights (for handling imbalanced data)
    class_weights = calculate_class_weights(y)

    # 4. Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset, scaler = create_tf_datasets(
        X_numerical, X_boolean, y, batch_size
    )

    # 5. Create feature columns (for certain models)
    feature_columns = create_feature_columns(numerical_features, boolean_features)

    # 6. Get input dimensions for neural networks
    numerical_dim = X_numerical.shape[1]
    boolean_dim = X_boolean.shape[1]
    total_features = numerical_dim + boolean_dim

    # print(f"\n=== Preprocessing Summary ===")
    # print(f"Numerical features dimension: {numerical_dim}")
    # print(f"Boolean features dimension: {boolean_dim}")
    # print(f"Total features: {total_features}")
    # print(f"Batch size: {batch_size}")

    # Sample batch for inspection
    # for batch_features, batch_labels in train_dataset.take(1):
        # print(f"Batch features - numerical: {batch_features['numerical_features'].shape}")
        # print(f"Batch features - boolean: {batch_features['boolean_features'].shape}")
        # print(f"Batch labels: {batch_labels.shape}")

    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'scaler': scaler,
        'class_weights': class_weights,
        'feature_columns': feature_columns,
        'input_dims': {
            'numerical_dim': numerical_dim,
            'boolean_dim': boolean_dim,
            'total_features': total_features
        },
        'feature_names': {
            'numerical': numerical_features,
            'boolean': boolean_features
        }
    }

def create_model_inputs(numerical_dim, boolean_dim):
    """
    Create input layers for different model architectures
    """
    # For simple models that combine all features
    numerical_input = tf.keras.layers.Input(shape=(numerical_dim,), name='numerical_features')
    boolean_input = tf.keras.layers.Input(shape=(boolean_dim,), name='boolean_features')

    # For models that treat features separately
    combined_input = tf.keras.layers.Concatenate()([numerical_input, boolean_input])

    return numerical_input, boolean_input, combined_input

def create_simple_preprocessor(numerical_dim, boolean_dim):
    """
    Create a simple preprocessing model that can be used with different architectures
    """
    numerical_input = tf.keras.layers.Input(shape=(numerical_dim,), name='numerical_features')
    boolean_input = tf.keras.layers.Input(shape=(boolean_dim,), name='boolean_features')

    # Normalize numerical features (additional normalization)
    normalized_numerical = tf.keras.layers.BatchNormalization()(numerical_input)

    # Combine features
    combined = tf.keras.layers.Concatenate()([normalized_numerical, boolean_input])

    preprocessor = tf.keras.Model(
        inputs=[numerical_input, boolean_input],
        outputs=combined,
        name='feature_preprocessor'
    )

    return preprocessor


# Example of how to use the preprocessor with different models
def example_model_usage(preprocessing_results):
    """
    Example showing how to use the preprocessed data with different models
    """
    input_dims = preprocessing_results['input_dims']

    # Create preprocessor
    preprocessor = create_simple_preprocessor(
        input_dims['numerical_dim'],
        input_dims['boolean_dim']
    )

    # Example: Simple Perceptron
    perceptron_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Create full model with preprocessor
    numerical_input = tf.keras.layers.Input(shape=(input_dims['numerical_dim'],), name='numerical_features')
    boolean_input = tf.keras.layers.Input(shape=(input_dims['boolean_dim'],), name='boolean_features')

    preprocessed_features = preprocessor([numerical_input, boolean_input])
    output = perceptron_model(preprocessed_features)

    full_model = tf.keras.Model(
        inputs=[numerical_input, boolean_input],
        outputs=output,
        name='intrusion_detection_model'
    )

    full_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return full_model


def create_perceptron(preprocessing_results, hidden_layers=[128, 64, 32], dropout_rate=0.3):
    """
    Create a customizable perceptron model
    """
    input_dims = preprocessing_results['input_dims']

    # Create preprocessor
    preprocessor = create_simple_preprocessor(
        input_dims['numerical_dim'],
        input_dims['boolean_dim']
    )

    # Create customizable perceptron
    perceptron_layers = []
    for units in hidden_layers:
        perceptron_layers.extend([
            tf.keras.layers.Dense(units, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate)
        ])

    # Remove dropout from last layer if present
    if perceptron_layers and isinstance(perceptron_layers[-1], tf.keras.layers.Dropout):
        perceptron_layers.pop()

    # Add output layer
    perceptron_layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

    perceptron_model = tf.keras.Sequential(perceptron_layers)

    # Create full model with preprocessor
    numerical_input = tf.keras.layers.Input(shape=(input_dims['numerical_dim'],), name='numerical_features')
    boolean_input = tf.keras.layers.Input(shape=(input_dims['boolean_dim'],), name='boolean_features')

    preprocessed_features = preprocessor([numerical_input, boolean_input])
    output = perceptron_model(preprocessed_features)

    full_model = tf.keras.Model(
        inputs=[numerical_input, boolean_input],
        outputs=output,
        name='intrusion_detection_model'
    )

    full_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )

    return full_model




if __name__ == "__main__":
    file_path = "/home/unbreakaskull/Downloads/embedded_system_network_security_dataset.csv"

    kcv_avg_f1_score = 0
    kcv_avg_precision = 0
    kcv_avg_recall = 0
    kcv_avg_accuracy = 0

    avg_f1 = 0
    avg_precision = 0
    avg_recall = 0
    avg_accuracy = 0


    for i in range (0,5):
        try:
            print("=== NETWORK SECURITY INTRUSION DETECTION ===")

            # Option 1: Perform k-fold cross-validation
            print("\n1. Performing K-Fold Cross-Validation...")
            cv_results = kfold_cross_validation(
                file_path,
                n_splits=5,
                epochs=15,
                batch_size=32,
                hidden_layers=[64, 32]
            )

            # Option 2: Traditional train/val/test split
            print("\n2. Traditional Train/Validation/Test Split...")
            preprocessing_results = main_preprocessing_pipeline(file_path, batch_size=64)

            # Train final model on full training data and evaluate on test set
            final_model, final_history = train_final_model(
                preprocessing_results,
                epochs=15,
                hidden_layers=[64, 32]
            )

            # Compare results
            print(f'\n{"=" * 60}')
            print('RESULTS SUMMARY')
            print(f'{"=" * 60}')
            print(f'Cross-Validation F1-Score: {cv_results["mean_f1"]:.4f}')


            # Calculate final test F1 for comparison
            test_loss, test_accuracy, test_precision, test_recall = final_model.evaluate(
                preprocessing_results['test_dataset'], verbose=0
            )
            test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
            print(f'Final Test F1-Score:      {test_f1:.4f}')

            #Add to averages for this iteration
            kcv_avg_f1_score += cv_results["mean_f1"]
            kcv_avg_precision += cv_results["mean_precision"]
            kcv_avg_recall += cv_results["mean_recall"]
            kcv_avg_accuracy += cv_results["mean_accuracy"]

            avg_f1 += test_f1
            avg_precision += test_precision
            avg_recall += test_recall
            avg_accuracy += test_accuracy



            # Save the final model (optional)
            final_model.save('train_validate_test_perceptron_128_64_32.h5')
            print("\nFinal model saved as 'train_validate_test_perceptron.h5'")

        except FileNotFoundError:
            print(f"File {file_path} not found. Please check the file path.")
        except Exception as e:
            print(f"Error during processing: {e}")
            import traceback

            traceback.print_exc()

    print(f'\n{"=" * 60}')
    print('After Five Loops:')
    print(f'{"=" * 60}')

    kcv_avg_f1_score /= 5
    kcv_avg_precision /= 5
    kcv_avg_recall /= 5
    kcv_avg_accuracy /= 5

    print(f'Cross-validation F1-Score: {kcv_avg_f1_score:.4f}')
    print(f'Cross-validation precision: {kcv_avg_precision:.4f}')
    print(f'Cross-validation recall: {kcv_avg_recall:.4f}')
    print(f'Cross-validation accuracy: {kcv_avg_accuracy:.4f}')

    avg_f1 /= 5
    avg_precision /= 5
    avg_recall /= 5
    avg_accuracy /= 5

    print(f'Final Test F1-Score: {avg_f1:.4f}')
    print(f'Final Test Precision: {avg_precision:.4f}')
    print(f'Final Test Recall: {avg_recall:.4f}')
    print(f'Final Test Accuracy: {avg_accuracy:.4f}')






