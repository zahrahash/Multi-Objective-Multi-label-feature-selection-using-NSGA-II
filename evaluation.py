import numpy as np



def relevantIndexes(matrix, row):
       
    relevant = []
    for j in range(matrix.shape[1]):
        if matrix[row,j] == 1:
            relevant.append(int(j))
    
    return relevant

def irrelevantIndexes(matrix, row):
   
    irrelevant = []
    for j in range(matrix.shape[1]):
        if matrix[row,j] == 0:
            irrelevant.append(int(j))
    
    return irrelevant

def accuracy(y_test, predictions):
    accuracy = 0.0
    for i in range(y_test.shape[0]):
        intersection = 0.0
        union = 0.0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j]) == 1 or int(predictions[i,j]) == 1:
                union += 1
            if int(y_test[i,j]) == 1 and int(predictions[i,j]) == 1:
                intersection += 1
            
        if union != 0:
            accuracy = accuracy + float(intersection/union)

    accuracy = float(accuracy/y_test.shape[0])

    return accuracy

def hammingLoss(y_test, predictions):
    hammingloss = 0.0
    for i in range(y_test.shape[0]):
        aux = 0.0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j]) != int(predictions[i,j]):
                aux = aux+1.0
        aux = aux/y_test.shape[1]
        hammingloss = hammingloss + aux
    
    return hammingloss/y_test.shape[0]

def oneError(y_true, y_pred_prob):
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    if y_true.shape != y_pred_prob.shape:
        raise ValueError("Input shapes of y_true and y_pred_prob must be the same.")

    n_samples, n_labels = y_true.shape


    y_pred = np.argmax(y_pred_prob, axis=1)

    
    one_error = np.mean([1 if y_true[i, y_pred[i]] == 0 else 0 for i in range(n_samples)])

    return one_error

def coverage(y_true, y_pred_prob):

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    if y_true.shape != y_pred_prob.shape:
        raise ValueError("Input shapes of y_true and y_pred_prob must be the same.")

    n_samples, n_labels = y_true.shape


    sorted_indices = np.argsort(-y_pred_prob, axis=1)
    coverage_errors = []

    for i in range(n_samples):
        true_labels = np.where(y_true[i] == 1)[0]
        sorted_labels = sorted_indices[i]
        num_labels_needed = 0
        labels_covered = set()

        for label in sorted_labels:
            labels_covered.add(label)
            num_labels_needed += 1

            if len(labels_covered.intersection(true_labels)) == len(true_labels):
                coverage_errors.append(num_labels_needed)
                break

    coverage_error = np.mean(coverage_errors)

    return coverage_error


def average_precision(y_true, y_pred_prob):
   
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    if y_true.shape != y_pred_prob.shape:
        raise ValueError("Input shapes of y_true and y_pred_prob must be the same.")

    n_samples, n_labels = y_true.shape

    average_precisions = []
    
    for label in range(n_labels):
        true_labels = y_true[:, label]
        pred_probs = y_pred_prob[:, label]
    
        sorted_indices = np.argsort(-pred_probs)
        sorted_labels = true_labels[sorted_indices]

        tp_cumsum = np.cumsum(sorted_labels)

        precision = tp_cumsum / np.arange(1, n_samples + 1)

        average_precision_label = np.sum(precision * sorted_labels) / np.sum(sorted_labels)
        average_precisions.append(average_precision_label)

    avg_precision = np.mean(average_precisions)

    return avg_precision

def rankingLoss(y_test, probabilities):
    rankingloss = 0.0

    for i in range(y_test.shape[0]):
        relevantVector = relevantIndexes(y_test, i)
        irrelevantVector = irrelevantIndexes(y_test, i)
        loss = 0.0

        for j in range(y_test.shape[1]):
            if y_test[i,j] == 1:
                for k in range(y_test.shape[1]):
                    if y_test[i,k] == 0:
                        if float(probabilities[i,j]) <= float(probabilities[i,k]):
                            loss += 1.0
        if len(relevantVector) != 0 and len(irrelevantVector) != 0:
            rankingloss += loss/float(len(relevantVector)*len(irrelevantVector))
    
    rankingloss /= y_test.shape[0]

    return rankingloss


def macro_f1(predicted, actual):
    num_labels = len(predicted[0])
    macro_f1_sum = 0
    
    for label_idx in range(num_labels):
        label_pred = [pred[label_idx] for pred in predicted]
        label_actual = [label[label_idx] for label in actual]
        
        label_tp = sum([1 for pred, actual in zip(label_pred, label_actual) if pred == 1 and actual == 1])
        label_fp = sum([1 for pred, actual in zip(label_pred, label_actual) if pred == 1 and actual == 0])
        label_fn = sum([1 for pred, actual in zip(label_pred, label_actual) if pred == 0 and actual == 1])
        
        label_f1 = (2 * label_tp) / (2 * label_tp + label_fp + label_fn) if 2 * label_tp + label_fp + label_fn > 0 else 0
        
        macro_f1_sum += label_f1
    
    macro_f1_score = macro_f1_sum / num_labels
    
    return macro_f1_score

def micro_f1(predicted, actual):
    micro_tp = sum([1 for pred_row, actual_row in zip(predicted, actual) for pred, actual in zip(pred_row, actual_row) if pred == 1 and actual == 1])
    micro_fp = sum([1 for pred_row, actual_row in zip(predicted, actual) for pred, actual in zip(pred_row, actual_row) if pred == 1 and actual == 0])
    micro_fn = sum([1 for pred_row, actual_row in zip(predicted, actual) for pred, actual in zip(pred_row, actual_row) if pred == 0 and actual == 1])
    
    micro_f1_score = (2 * micro_tp) / (2 * micro_tp + micro_fp + micro_fn) if 2 * micro_tp + micro_fp + micro_fn > 0 else 0
    
    return micro_f1_score

