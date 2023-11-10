import csv

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import torch
import json
import evaluate
import numpy as np
from tqdm import tqdm
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

base_tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq",
                                               cache_dir='/data/qinpeixin/huggingface')
base_model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq",
                                                   cache_dir='/data/qinpeixin/huggingface')
device = torch.device('cuda')

large_tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq",
                                                cache_dir='/data/qinpeixin/huggingface')
large_model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq",
                                                    cache_dir='/data/qinpeixin/huggingface')

softmax = torch.nn.Softmax(dim=1)

# generator = pipeline(model="WillHeld/roberta-base-coqa", cache_dir='/data/qinpeixin/huggingface')


def generate(instruction, knowledge, dialog, model, tokenizer, do_sample):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=1, top_p=0.9, do_sample=do_sample,
                             return_dict_in_generate=True, output_scores=True)
    output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    output_ids = outputs.sequences[0].tolist()
    prob = 1
    total_entropy = 0
    for i in range(len(output_ids) - 1):
        logits = softmax(outputs.scores[i])
        prob = prob * logits[0][output_ids[i + 1]]

        top_k_logits = torch.topk(outputs.scores[i], 5)[0]
        logits = softmax(top_k_logits)
        prob_cpu = logits.cpu().numpy()
        log_probs = np.log(prob_cpu)
        if np.isnan((-1 * np.sum(prob_cpu * log_probs)) / np.log(5)):
            total_entropy += 1
        else:
            total_entropy += (-1 * np.sum(prob_cpu * log_probs)) / np.log(5)
    total_entropy = 1 - (total_entropy / (len(output_ids) - 2))

    return output, prob.item(), total_entropy


# Instruction for a chitchat task
instruction = f'Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.'

with open('coqa-dev-v1.0.json', 'r', encoding='utf-8') as fp:
    dev_data = json.load(fp)

dev_data = dev_data['data']
knowledge_list = []
question_list = []
dialog_list = []
label_list = []
context_list = []
for i in range(len(dev_data)):
    for j in range(len(dev_data[i]['questions'])):
        knowledge_list.append(dev_data[i]['story'])
        single_dialog = []
        context = dev_data[i]['story']
        for k in range(j):
            single_dialog.append(dev_data[i]['questions'][k]['input_text'])
            single_dialog.append(dev_data[i]['answers'][k]['input_text'])
            context = context + '[SEP]' + dev_data[i]['questions'][k]['input_text'] + '[SEP]' + dev_data[i]['answers'][k]['input_text']
            context_list.append(context)
        single_dialog.append(dev_data[i]['questions'][j]['input_text'])
        question_list.append(dev_data[i]['questions'][j]['input_text'])
        dialog_list.append(single_dialog)
        label_list.append([dev_data[i]['answers'][j]['input_text']])

pop_list = [888, 889, 2178, 1811]
for p in pop_list:
    knowledge_list.pop(p)
    dialog_list.pop(p)
    label_list.pop(p)
    question_list.pop(p)

rouge = evaluate.load('rouge')
# bleu = evaluate.load('bleu')

# knowledge_list = knowledge_list[0:1000]
# dialog_list = dialog_list[0:1000]
# label_list = label_list[0:1000]

true_loss = []
weights = []
prob2_list = []
weight_result_list = []
uncertainty = []
plabel = []
for i in tqdm(range(len(knowledge_list))):
    res, prob, entropy1 = generate(instruction, knowledge_list[i], dialog_list[i], base_model, base_tokenizer, False)
    # result = generator(question=question_list[i], context=knowledge_list[i], min_length=1)
    # res = result['answer'].lower()
    res2, prob2, entropy2 = generate(instruction, knowledge_list[i], dialog_list[i], large_model, large_tokenizer, False)
    # res3, prob3 = generate(instruction, knowledge_list[i], dialog_list[i], large_model, large_tokenizer, True)
    # res4, prob4 = generate(instruction, knowledge_list[i], dialog_list[i], large_model, large_tokenizer, True)
    # res5, prob5 = generate(instruction, knowledge_list[i], dialog_list[i], large_model, large_tokenizer, True)
    res = res.lower()
    res2 = res2.lower()
    # res3 = res3.lower()
    # res4 = res4.lower()
    # res5 = res5.lower()
    results = rouge.compute(predictions=[res2], references=label_list[i])
    true_loss.append(results['rougeL'])
    weight_result = rouge.compute(predictions=[res2], references=[res])
    uncertainty.append(entropy2 * (1 - weight_result['rougeL']))
    plabel.append(weight_result['rougeL'])
    # weight_result2 = rouge.compute(predict/ions=[res], references=[res3])
    # weight_result3 = rouge.compute(predictions=[res], references=[res4])
    # weight_result4 = rouge.compute(predictions=[res], references=[res5])
    # weight = (prob2 * (1 - weight_result['rougeL']) + prob3 * (1 - weight_result2['rougeL']) + prob4 * (1 - weight_result3['rougeL']) + prob5 * (1-weight_result4['rougeL'])) / 4
    # weight = 1 - weight_result['rougeL'] * prob
    # weight = (1 - weight_result['rougeL']) * prob2
    # weights.append(weight)

num_samples_list = [5, 10, 15, 20, 25, 30]
for n in num_samples_list:
    hmce_list = pd.Series(uncertainty).sort_values(ascending=False).index[:n]
    label2 = []
    for i in range(len(plabel)):
        label2.append(plabel[i])
    for idx in hmce_list:
        label2[idx] = true_loss[idx]

    print(np.mean(label2))
exit(0)

# with open('/home/qinpeixin/InteractiveEvaluation/godel/tmp/weights_godel_inner.csv', 'r') as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         weights = row
#     weights = [float(x) for x in weights]
#
# with open('/home/qinpeixin/InteractiveEvaluation/godel/tmp/true_rouge_godel_inner.csv', 'r') as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         true_loss = row
#     true_loss = [float(x) for x in true_loss]

# with open('/home/qinpeixin/InteractiveEvaluation/godel/tmp/prob2_.csv', 'r') as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         prob2_list = row
#     prob2_list = [float(x) for x in prob2_list]
#
# with open('/home/qinpeixin/InteractiveEvaluation/godel/tmp/weight_result.csv', 'r') as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         weight_result_list = row
#     weight_result_list = [float(x) if 0.3 < float(x) < 0.7 else 1 for x in weight_result_list]
#
# with open('/home/qinpeixin/InteractiveEvaluation/godel/tmp/weights_.csv', 'r') as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         weights = row
#     weights = [float(x) for x in weights]
#
# with open('/home/qinpeixin/InteractiveEvaluation/godel/tmp/true_loss_.csv', 'r') as f:
#     f_csv = csv.reader(f)
#     for row in f_csv:
#         true_loss = row
#     true_loss = [float(x) for x in true_loss]
#
# for i in range(len(weights)):
#     weights[i] = prob2_list[i] * (1 - weight_result_list[i])

sum = 0
for weight in weights:
    sum += weight

# for i in range(len(weights)):
#     weights[i] = weights[i] / sum + 0.2 / len(weights)
# if weights[i] == 0:
#     weights[i] = 0.3 / len(weights)
# if weights[i] < 0.2 / len(weights):
#     # weights[i] = 0.2 / len(weights)
#     weights[i] = 0

# sum = np.sum(weights)

for i in range(len(weights)):
    weights[i] = weights[i] / sum
    # if weights[i] < 0.2 / len(weights):
    #     weights[i] = 0

num_samples_list = [5, 10, 15, 20, 25, 30]
# num_samples_list = [10, 20, 30, 50, 100, 150]
seeds = [42, 912, 323, 2023, 1914, 1024, 10086, 406, 401, 318, 1650, 411, 5515, 114, 514, 2, 10, 5,
         200, 1797, 21, 1637, 10124, 3912, 321, 8914, 8003, 2083, 165, 184, 708, 1499, 5523, 8551,
         8927, 6807, 98004, 988, 1708, 928, 81, 17647, 2809, 87, 29, 1411, 74, 14174, 14117, 90543,
         27231, 8480, 1825, 2001, 4769, 2377, 9784, 1107, 1456, 6348, 1838, 3285, 53, 4931, 6027]

true_rouge = np.mean(true_loss)

with open('/home/qinpeixin/InteractiveEvaluation/godel/tmp/tmp.csv', 'w') as f:
    write_head = ['method', 'num_samples', 'acc_mean', 'acc_var', 'mse']
    writer = csv.writer(f)
    writer.writerow([true_rouge])
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

            randomSampler = torch.utils.data.RandomSampler(range(len(label_list)), num_samples=num_samples)
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
            active_sample_index = [j for j in ActiveSampler]

            # for i in range(len(active_sample_index)):
            #     for j in range(i + 1, len(active_sample_index)):
            #         if weights[i] < weights[j]:
            #             active_sample_index[i], active_sample_index[j] = active_sample_index[j], \
            #                                                              active_sample_index[i]

            N = len(label_list)
            M = num_samples

            active_rouge = 0
            for i in range(len(active_sample_index)):
                j = active_sample_index[i]
                m = i + 1
                if weights[j] == 0:
                    print(weights[j])
                v = 1 + (1 / ((N - m + 1) * (weights[j])) - 1) * (N - M) / (N - m)
                v = 1

                active_rouge += v * (1 - true_loss[j])
            active_rouge = active_rouge / len(active_sample_index)
            active_rouge = 1 - active_rouge
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
