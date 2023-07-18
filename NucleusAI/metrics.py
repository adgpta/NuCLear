import numpy as np
from mask_overlay_func import overlay
from stardist.matching import matching, matching_dataset
from tqdm import tqdm

#======================-------3D_METRICS-----==========================

def prec_rec_3d(y_true, y_pred):
    stats = matching_dataset(y_true, y_pred, thresh=0.5, show_progress=False)
    prec, recall = stats[5:7]
    
    return prec, recall

def calculate_iou_3d(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    iou = intersection / union

    return iou

def calculate_dice_3d(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(np.logical_not(y_true), y_pred).sum()
    fn = np.logical_and(y_true, np.logical_not(y_pred)).sum()

    dice = (2 * tp) / (2 * tp + fp + fn)

    return dice

def calculate_auc_3d(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    tn = np.sum((1 - y_true) * (1 - y_pred))
    thresholds = np.unique(y_pred)
    sensitivities = []
    specificities = []
    for threshold in thresholds:
        tp_threshold = np.sum((y_pred >= threshold) * y_true)
        fn_threshold = np.sum((y_pred < threshold) * y_true)
        tn_threshold = np.sum((y_pred < threshold) * (1 - y_true))
        fp_threshold = np.sum((y_pred >= threshold) * (1 - y_true))
        sensitivity = tp_threshold / (tp_threshold + fn_threshold)
        specificity = tn_threshold / (tn_threshold + fp_threshold)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    auc_score = np.trapz(sensitivities, x=specificities)

    return auc_score

 #======================-------2D_METRICS-----==========================


def read_image_mask(true_mask, pred_mask):
    true = overlay.imread(true_mask)
    pred = overlay.imread(pred_mask)

    return true, pred

def cal_iou(true_mask, pred_mask):
    A = np.squeeze(true_mask)
    B = np.squeeze(pred_mask)

    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    
    return iou

def IOU(true_mask, pred_mask):

    intersection = np.logical_and(np.asarray(true_mask, dtype=float), np.asarray(pred_mask, dtype=float)).astype("uint8")
    union = np.logical_or(np.asarray(true_mask, dtype=float),np.asarray(pred_mask, dtype=float)).astype("uint8")

    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
    
def get_accuracy(true_mask, pred_mask):
    true_positive, true_negative, false_positive, false_negative = get_pixel_tp_tn_fp_fn(true_mask, pred_mask)
    #PRECISION
    if (true_positive + false_positive) != 0 : prec = true_positive / (true_positive+false_positive)
    else : prec = 0.0
    #RECALL
    if (true_positive + false_negative) != 0 : recall = true_positive / (true_positive+false_negative)
    else : recall = 0.0
    #ACCURACY
    acc = (true_positive + true_negative) / (true_negative + true_positive + false_negative + false_positive)
    #DICE
    if (2*true_positive + false_negative + false_positive) != 0 : dice = 2*true_positive / (2*true_positive + false_negative + false_positive)
    else : dice = 0.0
    #AUC
    if (false_positive+true_negative) != 0 : x = false_positive/(false_positive+true_negative)
    else : x = 0.0
    if (false_negative+true_positive) != 0 : y = false_negative/(false_negative+true_positive)
    else : y = 0.0
    auc = 1 - (1/2)*(x + y)
    

    return prec, recall, acc, dice, auc

def get_3D_accuracy(true_mask, pred_mask):
    true_positive, true_negative, false_positive, false_negative = get_pixel_tp_tn_fp_fn(true_mask, pred_mask)
    
    #ACCURACY
    acc = (true_positive + true_negative) / (true_negative + true_positive + false_negative + false_positive)

    return acc

def get_pixel_tp_tn_fp_fn(true_mask, predicted_mask):
    true_positive = np.logical_and(true_mask, predicted_mask).sum()

    not_gt = np.logical_not(true_mask)
    not_pd = np.logical_not(predicted_mask)
    np.logical_and(not_pd, not_gt).sum()
    
    true_negative = np.logical_and(not_pd, not_gt).sum()
    false_positive = np.logical_and(predicted_mask, not_gt).sum()
    false_negative = np.logical_and(not_pd, true_mask).sum()

    return true_positive, true_negative, false_positive, false_negative
