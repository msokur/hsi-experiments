from tensorflow.keras import backend as K
import config 

def recall_m(y_true, y_pred):   #sensitivity
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def specificity_m(y_true, y_pred):
    #true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    #predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    #specificity = true_positives / (predicted_positives + K.epsilon())
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    false_positives = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)))
    false_negatives = K.sum(K.round(K.clip(y_true - y_true, 0, 1)))
    true_negatives = (config.BATCH_SIZE * K.int_shape(y_true)[1]) - true_positives - false_positives - false_negatives
    
    print(f'tp {true_positives}, fp {false_positives}, tn {true_negatives}, fn {false_negatives}')
    specificity = true_negatives / (true_negatives + false_positives + K.epsilon())
    
    #check_binary(K.eval(y_true))    # must check that input values are 0 or 1
    #check_binary(K.eval(y_pred))    # 

    #TN = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 0)
    #FP = np.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 1)

    # as Keras Tensors
    #TN = K.sum(K.variable(TN))
    #FP = K.sum(K.variable(FP))

    #specificity = TN / (TN + FP + K.epsilon())
    return specificity

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))