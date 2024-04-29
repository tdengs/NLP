import sys
import csv
from itertools import groupby
import itertools
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python task1_eval_script.py <predictions_file> <validation_file>")
    sys.exit(1)
    
pred_f_name = sys.argv[1]
gold_f_name = sys.argv[2]

def process_data(data):
    """sort by word pairs ids and then remove ids"""
    data = sorted(data, key=lambda i: int(i[0]))
    data = [row[1:] for row in data]
    data = [row[:-1]+[float(row[-1])] for row in data]
    return data

def read_file(file_name):
    """Read a csv file as a list and sort it"""
    with open(file_name) as f:
        reader = csv.reader(f)
        data = list(reader)
    data = process_data(data)
    return data

gold_standards = read_file(gold_f_name)
predictions = read_file(pred_f_name)
# Attach text word pairs to prediction similarity scores
predictions = [[gold_standards[i][0],gold_standards[i][1], predictions[i][0]] for i in range(len(gold_standards))]

# Group words pairs that have common words
gold_std_grouped = [list(it) for k, it in groupby(gold_standards, lambda p:p[0])]  
pred_grouped = [list(it) for k, it in groupby(predictions, lambda p:p[0])]

print('The following simalarity scores may need checking:')
total = 0
wrong_hits = 0
for idx in range(len(gold_std_grouped)):
    preds = pred_grouped[idx]
    gold_std = gold_std_grouped[idx]
    # Get all possible permutations between word pairs that have common words
    pred_pairs_ = list(itertools.combinations(preds, 2))
    gold_pairs_ = list(itertools.combinations(gold_std, 2))
    # Get the relative order of word pairs' similarity scores in each permutation
    pred_pairs = map(lambda x:1 if x[0][2]-x[1][2] > 0 else 0, pred_pairs_)
    gold_pairs = map(lambda x:1 if x[0][2]-x[1][2] > 0 else 0, gold_pairs_)
    # Check if your prediction gives the same relative order as the gold standard
    out_arr = np.subtract(np.array(list(pred_pairs)), np.array(list(gold_pairs)))
    total+=len(out_arr)
    incorr = np.nonzero(out_arr)
    wrong_hits += len(incorr[0])
    for item in incorr[0]:
        print("({},{}) similarity score: {}, gold ranking: {}".format(pred_pairs_[item][0][0], pred_pairs_[item][0][1], pred_pairs_[item][0][2], gold_pairs_[item][0][2]))
        print("({},{}) similarity score: {}, gold ranking: {}".format(pred_pairs_[item][1][0], pred_pairs_[item][1][1], pred_pairs_[item][1][2], gold_pairs_[item][1][2]))
        print("----------------------------")

# Accuracy = hit/total number of permutations 
# We consider a hit as the predicted similarity scores' relative order between two pairs is the same as gold ranking
print("Accuracy: {}".format(1-wrong_hits/total))
        