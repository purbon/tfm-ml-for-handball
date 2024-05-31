from sklearn.metrics import confusion_matrix


def get_score_rates(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = cm[0][0]
    fp = cm[0][1] #if cm.shape[1] > 1 else -1
    fn = cm[1][0] #if cm.shape[0] > 1 else -1
    tp = cm[1][1] #if cm.shape[1] > 1 else -1

    fnr = fn / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    tpr = tp / (tp + fn)
    bacc = (tpr + tnr) / 2

    return fnr, fpr, tnr, tpr, bacc
