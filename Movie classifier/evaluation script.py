import sys
from sklearn.metrics import precision_score, recall_score
import numpy as np
import csv

if len(sys.argv) != 3:
    print("Usage: python task2_eval_script.py <predictions_file> <validation_file>")
    sys.exit(1)
    
pred_f_name = sys.argv[1]
gold_f_name = sys.argv[2]

def process_data(data):
    """sort by word pairs ids and then remove ids"""
    data = sorted(data, key=lambda i: i[0])
    data = [row[1:] for row in data]
    data = [[int(item) for item in row] for row in data]
    return data

def read_ans_file(file_name):
    """Read csv file as a list and sort it by ids"""
    with open(file_name) as f:
        reader = csv.reader(f)
        data = list(reader)
    data = process_data(data)
    return data

def read_val_file(file_name):
    """Read csv file as a list and sort it by ids"""
    with open(file_name) as f:
        reader = csv.reader(f)
        data = [item[:1]+item[3:] for item in list(reader)][1:]
    data = process_data(data)
    return data

predictions = read_ans_file(pred_f_name)
gold_standards = read_val_file(gold_f_name)

# Class level precision socres and recall scores 
# Give a precision and a recall for each class
class_level_precision = precision_score(gold_standards, predictions, average=None, zero_division=0)
class_level_recall = recall_score(gold_standards, predictions, average=None, zero_division=0)

# Movie level precision scores and recall scores
# Average over all testing samples
gold_standard_array = np.array(gold_standards, dtype=float)
prediction_array = np.array(predictions, dtype=float)
out_array = np.subtract(gold_standard_array, prediction_array)

# False positive
FP = np.count_nonzero(out_array == -1, axis=1)
# True Positive + False Positive
TP_FP = np.count_nonzero(prediction_array == 1, axis=1)
# False Negtive
FN = np.count_nonzero(out_array == 1, axis=1)
# False Negtive + True Positve
FN_TP = np.count_nonzero(gold_standard_array == 1, axis=1)
movie_level_precision = 1 - np.divide(FP, TP_FP, out=np.ones_like(FP, dtype=float), where=TP_FP!=0)
movie_level_recall = 1 - np.divide(FN, FN_TP, out=np.ones_like(FN, dtype=float), where=FN_TP!=0)


if __name__ == "__main__":
    print("Class level: ")
    for i in range(9):
        try:
            class_precision = class_level_precision[i]
            class_recall = class_level_recall[i]

            if class_precision == 0 or class_recall == 0:
                f1_score = 0
            else:
                f1_score = (2 * class_precision * class_recall) / (class_precision + class_recall)

            print("Class {0:2d} F1 score: {1:.4f}".format(i+1, f1_score))
        except Exception as e:
            print("Class {0:2d} F1 score: 0".format(i+1))
    print("----------------------------")
    print("Movie (document) level: ")
    print("Precision: {0:.4f}".format(np.mean(movie_level_precision)))
    print("Recall: {0:.4f}".format(np.mean(movie_level_recall)))
    
    
    
