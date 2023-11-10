import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
from transformers import TrainingArguments, Trainer, AutoConfig
import numpy as np
import evaluate
import os
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["WANDB_DISABLED"] = "true"

ActiveTesting = True
num_samples = 512
seed = 912
metric = 'loss'
saveFile = False
clip = 0.2

bertTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ernieTokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en")
debertaTokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large", cache_dir='/data/qinpeixin/huggingface')


def bert_preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples["premise"], examples["hypothesis"])
    )
    result = bertTokenizer(*args, padding="max_length", max_length=128, truncation=True)

    return result


def ernie_preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples["premise"], examples["hypothesis"])
    )
    result = ernieTokenizer(*args, padding="max_length", max_length=128, truncation=True)

    return result


def deberta_preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples["premise"], examples["hypothesis"])
    )
    result = debertaTokenizer(*args, padding="max_length", max_length=128, truncation=True)

    return result


metrics = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)


def main(metric, num_samples_list, seed_list, clip):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    bertConfig = AutoConfig.from_pretrained("bert-base-uncased", num_labels=3)
    ernieConfig = AutoConfig.from_pretrained("nghuyong/ernie-2.0-base-en", num_labels=3)
    debertaConfig = AutoConfig.from_pretrained("microsoft/deberta-large", num_labels=3)

    bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=bertConfig,
                                                              ignore_mismatched_sizes=True, revision='main')
    bert.load_state_dict(torch.load(
        "/home/qinpeixin/InteractiveEvaluation/Classification/output/bert/random_eval/checkpoint-1000/pytorch_model.bin"))

    ernie = AutoModelForSequenceClassification.from_pretrained("nghuyong/ernie-2.0-base-en", config=ernieConfig,
                                                               ignore_mismatched_sizes=True, revision='main')
    ernie.load_state_dict(torch.load(
        "/home/qinpeixin/InteractiveEvaluation/Classification/output/ernie/random_eval/checkpoint-1152/pytorch_model.bin"))

    deberta = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large", config=debertaConfig,
                                                                 ignore_mismatched_sizes=True, revision='main',
                                                                 cache_dir='/data/qinpeixin/huggingface')
    deberta.load_state_dict(torch.load(
        "/data/qinpeixin/models/deberta/large/mnli/pytorch_model.bin"))

    raw_datasets = datasets.load_dataset('glue', 'mnli')

    bert_datasets = raw_datasets.map(bert_preprocess_function, batched=True)
    ernie_datasets = raw_datasets.map(ernie_preprocess_function, batched=True)
    deberta_datasets = raw_datasets.map(deberta_preprocess_function, batched=True)

    # random_sampler = torch.utils.data.sampler.RandomSampler(bert_datasets['validation_matched'], num_samples=num_samples, replacement=False)

    bert_datasets = bert_datasets.remove_columns(["premise", 'hypothesis', 'idx'])
    bert_datasets = bert_datasets.rename_column("label", "labels")
    bert_datasets.set_format("torch")

    deberta_datasets = deberta_datasets.remove_columns(["premise", 'hypothesis', 'idx'])
    deberta_datasets = deberta_datasets.rename_column("label", "labels")
    deberta_datasets.set_format("torch")

    eval_dataloader = torch.utils.data.DataLoader(bert_datasets['validation_matched'], batch_size=1, shuffle=False)
    deberta_eval_dataloader = torch.utils.data.DataLoader(deberta_datasets['validation_matched'], batch_size=1,
                                                          shuffle=False)

    device = torch.device('cuda')
    bert = bert.to(device)
    ernie = ernie.to(device)
    deberta = deberta.to(device)

    bert.eval()
    ernie.eval()
    deberta.eval()
    weights = []
    true_loss = []
    for i, data in enumerate(zip(eval_dataloader, deberta_eval_dataloader)):
        batch = data[0]
        batch2 = data[1]

        batch = {k: v.to(device) for k, v in batch.items()}
        batch2 = {k: v.to(device) for k, v in batch2.items()}
        with torch.no_grad():
            outputs = bert(**batch)
            # outputs2 = deberta(**batch2)
            outputs2 = ernie(**batch)

        logits = outputs.logits
        softmax = torch.nn.Softmax(dim=1)
        predictions = softmax(outputs2.logits)

        if metric == 'loss':
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), predictions.view(-1, 3)).item()
            t_loss = outputs.loss.item()
        elif metric == 'accuracy':
            arg = torch.argmax(logits, dim=1).item()
            loss = (1 - predictions[0][arg].item()) * predictions[0][arg].item()
            arg = torch.argmax(outputs.logits, dim=1).item()
            if arg == batch['labels'].item():
                t_loss = 0
            else:
                t_loss = 1

        weights.append(loss)
        true_loss.append(t_loss)

    true_loss_num = np.mean(true_loss)

    save_weights = weights

    sum = 0
    for weight in weights:
        sum += weight

    for i in range(len(weights)):
        weights[i] = weights[i] / sum

    # weights = torch.tensor(weights)
    # softmax = torch.nn.Softmax(dim=0)
    # weights = softmax(weights)
    # weights = weights.tolist()

    if clip is not None:
        for i in range(len(weights)):
            if weights[i] < clip / 9815:
                # weights[i] = clip / 9815
                weights[i] = 0

    with open('/home/qinpeixin/InteractiveEvaluation/Classification/result_1145.csv', 'w') as f:
        write_head = ['method', 'num_samples', 'acc_mean', 'acc_var', 'mse']
        writer = csv.writer(f)
        writer.writerow([true_loss_num])
        writer.writerow(write_head)
        for num_samples in num_samples_list:
            random_list = []
            active_list = []
            active_error = []
            random_error = []
            for seed in seed_list:
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                randomSampler = torch.utils.data.RandomSampler(range(len(true_loss)), num_samples=num_samples)
                random_sample_index = [j for j in randomSampler]

                random_loss = 0
                for i in random_sample_index:
                    random_loss += true_loss[i]
                random_loss = random_loss / len(random_sample_index)
                if metric == 'loss':
                    random_list.append(random_loss)
                elif metric == 'accuracy':
                    random_list.append(1 - random_loss)
                random_error.append((random_loss - true_loss_num) ** 2)

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
                N = len(true_loss)
                M = num_samples

                active_loss = 0
                for i in range(len(active_sample_index)):
                    j = active_sample_index[i]
                    m = i + 1
                    v = 1 + (1 / ((N - m + 1) * (weights[j])) - 1) * (N - M) / (N - m)

                    active_loss += v * true_loss[j]

                active_loss = active_loss / len(active_sample_index)
                if metric == 'loss':
                    active_list.append(active_loss)
                elif metric == 'accuracy':
                    active_list.append(1 - active_loss)
                active_error.append((active_loss - true_loss_num) ** 2)
            random_mean = np.mean(random_list)
            random_var = np.var(random_list)
            active_mean = np.mean(active_list)
            active_var = np.var(active_list)
            random_err = np.mean(random_error)
            active_err = np.mean(active_error)
            print(active_var)
            print(active_err)
            write_data1 = ['active', num_samples, active_mean, active_var, active_err]
            write_data2 = ['random', num_samples, random_mean, random_var, random_err]
            writer.writerow(write_data1)
            writer.writerow(write_data2)


if __name__ == '__main__':
    num_samples_list = [256]

    seeds = [42, 912, 323, 2023, 1914, 1024, 10086, 406, 401, 318, 1650, 411, 5515, 114, 514, 2, 10, 5,
             200, 1797, 21, 1637, 10124, 3912, 321, 8914, 8003, 2083, 165, 184, 708, 1499, 5523, 8551,
             8927, 6807, 98004, 988, 1708, 928, 81, 17647, 2809, 87, 29, 1411, 74, 14174, 14117, 90543,
             27231, 8480, 1825, 2001, 4769, 2377, 9784, 1107, 1456, 6348, 1838, 3285, 53, 4931, 6027]

    # seeds = [42, 912, 323, 2023, 1914, 1024, 10086, 406, 401, 318]

    main('loss', num_samples_list, seeds, 0.4)
