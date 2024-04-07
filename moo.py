import numpy as np
from MLKNN import MLKNN
from evaluation import hammingLoss

def feature_selection_objectives(solution, Data):
    
    X_train = Data[0]
    Y_train = Data[1]
    X_test = Data[2]
    Y_test = Data[3]
    
    # Convert solution to binary to select features
    selected_features = np.where(solution >= 0.5)[0]
    
    # Check if no feature is selected, return worst case
    if len(selected_features) == 0:
        return 1.0, len(solution)  # Max Hamming loss, max number of features

    # Select features based on the solution
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Apply MLKNN
    mlknn = MLKNN(k=10)
    Ph1, Ph0, Peh1, Peh0 = mlknn.fit(X_train_selected, Y_train)
    _, predictions = mlknn.predict(X_train_selected, Y_test, X_test_selected, Y_train, Ph1, Ph0, Peh1, Peh0)

    # Calculate objectives: Hamming Loss and Number of Features
    h_loss = hammingLoss(Y_test, predictions)  # Use the hammingLoss function from evaluation.py
    num_features = len(selected_features)

    return np.array([h_loss, num_features])
