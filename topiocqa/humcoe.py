import csv
import pandas as pd
import numpy as np
import torch

keys = ['true_em', 'true_f1', 'psedo_em', 'psedo_f1', 'g_confidence', 'f_confidence']
clip = 0.2
num_samples_list = [5, 10, 15, 20, 25, 30]
seeds = [42, 912, 323, 2023, 1914, 1024, 10086, 406, 401, 318, 1650, 411, 5515, 114, 514, 2, 10, 5,
         200, 1797, 21, 1637, 10124, 3912, 321, 8914, 8003, 2083, 165, 184, 708, 1499, 5523, 8551,
         8927, 6807, 98004, 988, 1708, 928, 81, 17647, 2809, 87, 29, 1411, 74, 14174, 14117, 90543,
         27231, 8480, 1825, 2001, 4769, 2377, 9784, 1107, 1456, 6348, 1838, 3285, 53, 4931, 6027]


def read_file(result_file):
    result_dict = {}
    for key in keys:
        result_dict[key] = []

    with open(result_file, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            row = row[0].split('\t')
            for i in range(len(keys)):
                result_dict[keys[i]].append(float(row[i]))

    return result_dict


def save_file(filename, result_list):
    with open(filename, 'w') as f:
        write_head = ['method', 'num_samples', 'acc_mean', 'acc_var', 'mse']
        writer = csv.writer(f)
        writer.writerow(write_head)
        for item in result_list:
            writer.writerow(item)


def active_testing(result_dict, metric):
    if metric == 'em':
        metric_keys = ['true_em', 'psedo_em']
    else:
        metric_keys = ['true_f1', 'psedo_f1']

    sample_weight = []

    for i in range(len(result_dict[metric_keys[0]])):
        sample_weight.append(result_dict['g_confidence'][i] * (1 - result_dict[metric_keys[1]][i]))

    sum = np.sum(sample_weight)

    for i in range(len(sample_weight)):
        sample_weight[i] = sample_weight[i] / sum
        if sample_weight[i] < clip / len(sample_weight):
            sample_weight[i] = 0

    true_metric = np.mean(result_dict[metric_keys[0]])
    true_metric_list = result_dict[metric_keys[0]]

    print(true_metric)

    result_list = []

    for num_samples in num_samples_list:
        random_list = []
        active_list = []
        active_error = []
        random_error = []
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            randomSampler = torch.utils.data.RandomSampler(range(len(sample_weight)), num_samples=num_samples)
            random_sample_index = [j for j in randomSampler]

            random_metric = 0
            for i in random_sample_index:
                random_metric += true_metric_list[i]
            random_metric = random_metric / len(random_sample_index)
            random_list.append(random_metric)
            random_error.append((random_metric - true_metric) ** 2)

            weights = torch.tensor(sample_weight, device='cuda')
            ActiveSampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=num_samples,
                                                                   replacement=False)
            weights = weights.tolist()
            active_sample_index = [j for j in ActiveSampler]

            N = len(sample_weight)
            M = num_samples

            active_metric = 0
            for i in range(len(active_sample_index)):
                j = active_sample_index[i]
                m = i + 1
                if weights[j] == 0:
                    print(weights[j])
                v = 1 + (1 / ((N - m + 1) * (weights[j])) - 1) * (N - M) / (N - m)

                active_metric += v * (1 - true_metric_list[j])
            active_metric = active_metric / len(active_sample_index)
            active_metric = 1 - active_metric
            active_list.append(active_metric)
            active_error.append((active_metric - true_metric) ** 2)

        random_mean = np.mean(random_list)
        random_var = np.var(random_list)
        active_mean = np.mean(active_list)
        active_var = np.var(active_list)
        random_err = np.mean(random_error)
        active_err = np.mean(active_error)

        active_data = ['active', num_samples, active_mean, active_var, active_err]
        random_data = ['random', num_samples, random_mean, random_var, random_err]

        result_list.append(active_data)
        result_list.append(random_data)

    return result_list


def hmceval(result_dict, metric):
    if metric == 'em':
        metric_keys = ['true_em', 'psedo_em']
    else:
        metric_keys = ['true_f1', 'psedo_f1']

    uncertainty = []
    for item in result_dict['f_confidence']:
        uncertainty.append(1 - item)
    plabel = result_dict[metric_keys[1]]
    true_metric = result_dict[metric_keys[0]]

    result_list = []

    for n in num_samples_list:
        hmce_list = pd.Series(uncertainty).sort_values(ascending=False).index[:n]
        hmce_metric = []
        for i in range(len(plabel)):
            hmce_metric.append(plabel[i])
        for idx in hmce_list:
            hmce_metric[idx] = true_metric[idx]

        result_list.append(['HMCE', n, np.mean(hmce_metric)])

    return result_list


def active_top_k(result_dict, metric):
    if metric == 'em':
        metric_keys = ['true_em', 'psedo_em']
    else:
        metric_keys = ['true_f1', 'psedo_f1']

    uncertainty = []
    for item in result_dict['f_confidence']:
        uncertainty.append(1 - item)

    true_metric = result_dict[metric_keys[0]]

    result_list = []

    for n in num_samples_list:
        topk_list = pd.Series(uncertainty).sort_values(ascending=False).index[:n]
        topk_metric = []
        for i in range(n):
            topk_metric.append(true_metric[topk_list[i]])

        result_list.append(['top-K', n, np.mean(topk_metric)])

    return result_list


def mcm(result_dict, metric):
    if metric == 'em':
        metric_keys = ['true_em', 'psedo_em']
    else:
        metric_keys = ['true_f1', 'psedo_f1']

    uncertainty = []
    for item in result_dict['f_confidence']:
        uncertainty.append(1 - item)

    true_metric = result_dict[metric_keys[0]]

    result_list = []

    for n in num_samples_list:
        mcm_list = pd.Series(uncertainty).sort_values(ascending=True).index[:n]
        mcm_metric = []
        for i in range(n):
            mcm_metric.append(true_metric[mcm_list[i]])

        result_list.append(['MCM', n, np.mean(mcm_metric)])

    return result_list


def main(result_file, metric, save_file_name):
    result_dict = read_file(result_file)
    # result_list = active_testing(result_dict, metric)
    result_list = hmceval(result_dict, metric)
    # result_list = active_top_k(result_dict, metric)
    # result_list = mcm(result_dict, metric)
    for item in result_list:
        print(item)
    save_file(save_file_name, result_list)


if __name__ == "__main__":
    result_file = '/home/qinpeixin/InteractiveEvaluation/topiocqa/results/activeTest/dpr.csv'
    save_file_name = '/home/qinpeixin/InteractiveEvaluation/topiocqa/results/activeTest/tmp.csv'
    main(result_file, 'f1', save_file_name)
