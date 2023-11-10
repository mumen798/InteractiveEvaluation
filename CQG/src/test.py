import evaluate
import csv
import numpy as np
import torch
import math
from tqdm import tqdm
import pandas as pd

rouge = evaluate.load('rouge')

true_label = []
distil = []
probs_distil = []
gpt2 = []
probs_gpt2 = []
bart = []
probs_bart = []
t5 = []
probs_t5 = []
human_rel = []
human_nar = []
chat_rel = []
chat_nar = []

with open('/home/qinpeixin/InteractiveEvaluation/CQG/data/cqg.csv', 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        question = row
        true_label.append(question[0])

with open('/home/qinpeixin/InteractiveEvaluation/CQG/data/distil.csv', 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        question = row
        distil.append(question[0])
        probs_distil.append(question[1])

with open('/home/qinpeixin/InteractiveEvaluation/CQG/data/gpt2_entropy.csv', 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        question = row
        gpt2.append(question[0])
        probs_gpt2.append(float(question[1]))

with open('/home/qinpeixin/InteractiveEvaluation/CQG/data/bartbase_entropy.csv', 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        question = row
        bart.append(question[0])
        probs_bart.append(float(question[1]))

with open('/home/qinpeixin/InteractiveEvaluation/CQG/data/t5base_entropy.csv', 'r', errors='ignore') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        question = row
        t5.append(question[0])
        probs_t5.append(float(question[1]))

with open('/home/qinpeixin/InteractiveEvaluation/CQG/data/bartEval.csv', 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        question = row
        human_rel.append(float(question[0]))
        human_nar.append(float(question[1]))

with open('/home/qinpeixin/InteractiveEvaluation/CQG/data/bartChatgpt.csv', 'r') as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        question = row
        chat_rel.append(float(question[0]))
        chat_nar.append(float(question[1]))


# for i in range(len(probs)):
#     probs[i] = (probs[i] - p_min) / (p_max - p_min)

true_loss = []
weights = []
rouge1_list = []
rouge2_list = []
plabel = []

for i in tqdm(range(len(true_label))):
    # rouge1 = rouge.compute(predictions=[bart[i]], references=[true_label[i]])
    # rouge2 = rouge.compute(predictions=[bart[i]], references=[gpt2[i]])
    # rouge1_list.append(r_bart['rougeL'])
    # rouge2_list.append(r_t5['rougeL'])
    # rouge3 = rouge.compute(predictions=[bart[i]], references=[gpt2[i]])
    plabel.append(chat_nar[i])
    # rouge4 = rouge.compute(predictions=[bart[i]], references=[t5[i]])
    true_loss.append(human_nar[i])
    weight = probs_bart[i]
    # weight = 5 - chat_rel[i] * probs_bart[i]
    # weight = 1 - rouge1['rougeL']
    weights.append(weight)

# print(np.mean(true_loss))
# print()
# num_samples_list = [5, 10, 15, 20, 25, 30]
# for n in num_samples_list:
#     hmce_list = pd.Series(weights).sort_values(ascending=True).index[:n]
#     label2 = []
#     for i in range(len(hmce_list)):
#         label2.append(true_loss[hmce_list[i]])
#     # for idx in hmce_list:
#     #     label2[idx] = true_loss[idx]
#
#     print(np.mean(label2))
# exit(0)
# print(np.mean(rouge1_list))
# print(np.mean(rouge2_list))

sum = 0
for weight in weights:
    sum += weight

for i in range(len(weights)):
    weights[i] = weights[i] / sum
    if weights[i] < 0.2 / len(weights):
        # print(i)
        weights[i] = 0

# w_min = np.min(weights)
# for i in range(len(weights)):
#     weights[i] = weights[i] - w_min
#
# sum = 0
# for weight in weights:
#     sum += weight
#
# for i in range(len(weights)):
#     weights[i] = weights[i] / sum
#     if weights[i] < 0.2 / len(weights):
#         print(i)
#         weights[i] = 0.2 / len(weights)


num_samples_list = [5, 10, 15, 20, 25, 30]
seeds = [42, 912, 323, 2023, 1914, 1024, 10086, 406, 401, 318, 1650, 411, 5515, 114, 514, 2, 10, 5,
         200, 1797, 21, 1637, 10124, 3912, 321, 8914, 8003, 2083, 165, 184, 708, 1499, 5523, 8551,
         8927, 6807, 98004, 988, 1708, 928, 81, 17647, 2809, 87, 29, 1411, 74, 14174, 14117, 90543,
         27231, 8480, 1825, 2001, 4769, 2377, 9784, 1107, 1456, 6348, 1838, 3285, 53, 4931, 6027]
# np.random.seed(42)
# seeds = np.random.randint(0, 100000, size=650).tolist()

true_rouge = np.mean(true_loss)

with open('/home/qinpeixin/InteractiveEvaluation/CQG/out/result.csv', 'w') as f:
    write_head = ['method', 'num_samples', 'acc_mean', 'acc_var', 'mse']
    writer = csv.writer(f)
    writer.writerow([true_rouge])
    print([true_rouge])
    writer.writerow(write_head)
    print(write_head)
    for num_samples in num_samples_list:
        random_list = []
        active_list = []
        active_error = []
        random_error = []
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            randomSampler = torch.utils.data.RandomSampler(range(len(true_label)), num_samples=num_samples)
            random_sample_index = [j for j in randomSampler]

            random_rouge = 0
            for i in random_sample_index:
                random_rouge += true_loss[i]
            random_rouge = random_rouge / len(random_sample_index)
            random_list.append(random_rouge)
            random_error.append((random_rouge - true_rouge) ** 2)

            weights = torch.tensor(weights, device='cuda')
            ActiveSampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=num_samples,
                                                                   replacement=False)
            weights = weights.tolist()
            active_sample_index = [j for j in randomSampler]

            # for i in range(len(active_sample_index)):
            #     for j in range(i + 1, len(active_sample_index)):
            #         if weights[i] < weights[j]:
            #             active_sample_index[i], active_sample_index[j] = active_sample_index[j], \
            #                                                              active_sample_index[i]

            N = len(true_label)
            M = num_samples

            active_rouge = 0
            for i in range(len(active_sample_index)):
                j = active_sample_index[i]
                m = i + 1
                if weights[j] == 0:
                    v = 1
                else:
                    v = 1 + (1 / ((N - m + 1) * (weights[j])) - 1) * (N - M) / (N - m)
                # v = 1

                active_rouge += v * (5 - true_loss[j])
            active_rouge = active_rouge / len(active_sample_index)
            active_rouge = 5 - active_rouge
            active_list.append(active_rouge)
            active_error.append((active_rouge - true_rouge) ** 2)

        random_mean = np.mean(random_list)
        random_var = np.var(random_list)
        active_mean = np.mean(active_list)
        active_var = np.var(active_list)
        random_err = np.mean(random_error)
        active_err = np.mean(active_error)
        write_data1 = ['active', num_samples, active_mean, active_var, active_err]
        # write_data1 = ['active', num_samples] + active_list
        write_data2 = ['random', num_samples, random_mean, random_var, random_err]
        # write_data2 = ['random', num_samples] + random_list
        writer.writerow(write_data1)
        # writer.writerow(write_data2)
        print(write_data1)
        print(write_data2)

