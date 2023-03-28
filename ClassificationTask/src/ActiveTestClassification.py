import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
from transformers import TrainingArguments, Trainer, AutoConfig
import numpy as np
import evaluate
import os
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ActiveTesting = True
num_samples = 512
seed = 912
metric = 'loss'
saveFile = False
clip = 0.2

bertTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
ernieTokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-base-en")


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


metrics = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=predictions, references=labels)


def main(metric, num_samples, seed, clip, saveFile, sorted):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    bertConfig = AutoConfig.from_pretrained("bert-base-uncased", num_labels=3)
    ernieConfig = AutoConfig.from_pretrained("nghuyong/ernie-2.0-base-en", num_labels=3)

    bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=bertConfig,
                                                              ignore_mismatched_sizes=True, revision='main')
    bert.load_state_dict(torch.load(
        "/home/qinpeixin/InteractiveEvaluation/Classification/output/bert/random_eval/checkpoint-1000/pytorch_model.bin"))

    ernie = AutoModelForSequenceClassification.from_pretrained("nghuyong/ernie-2.0-base-en", config=ernieConfig,
                                                               ignore_mismatched_sizes=True, revision='main')
    ernie.load_state_dict(torch.load(
        "/home/qinpeixin/InteractiveEvaluation/Classification/output/ernie/random_eval/checkpoint-1152/pytorch_model.bin"))

    raw_datasets = datasets.load_dataset('glue', 'mnli')

    bert_datasets = raw_datasets.map(bert_preprocess_function, batched=True)
    ernie_datasets = raw_datasets.map(ernie_preprocess_function, batched=True)

    # random_sampler = torch.utils.data.sampler.RandomSampler(bert_datasets['validation_matched'], num_samples=num_samples, replacement=False)

    random_eval_dataset = bert_datasets['validation_matched'].shuffle(seed=seed).select(range(num_samples))

    training_args = TrainingArguments(
        output_dir="/home/qinpeixin/InteractiveEvaluation/Classification/output/ernie/random_eval",
        evaluation_strategy="epoch",
        per_device_eval_batch_size=256,
        per_device_train_batch_size=256,
        fp16=True,
        save_strategy='epoch'
    )

    bert_datasets = bert_datasets.remove_columns(["premise", 'hypothesis', 'idx'])
    bert_datasets = bert_datasets.rename_column("label", "labels")
    bert_datasets.set_format("torch")
    eval_dataloader = torch.utils.data.DataLoader(bert_datasets['validation_matched'], batch_size=1, shuffle=False)
    device = torch.device('cuda')
    bert = bert.to(device)
    ernie = ernie.to(device)

    bert.eval()
    ernie.eval()
    weights = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = bert(**batch)
            outputs2 = ernie(**batch)

        logits = outputs.logits
        softmax = torch.nn.Softmax(dim=1)
        predictions = softmax(outputs2.logits)

        if metric == 'loss':
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), predictions.view(-1, 3)).item()
        elif metric == 'accuracy':
            arg = torch.argmax(logits, dim=1).item()
            loss = 1 - predictions[0][arg].item()

        weights.append(loss)

    sum = 0
    for weight in weights:
        sum += weight

    for i in range(len(weights)):
        weights[i] = weights[i] / sum

    if clip is not None:
        for i in range(len(weights)):
            if weights[i] < clip / 9815:
                weights[i] = clip / 9815

    weights = torch.tensor(weights, device='cuda')
    ActiveSampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=False)
    sample_index = [j for j in ActiveSampler]
    if sorted:
        for i in range(len(sample_index)):
            for j in range(i+1, len(sample_index)):
                if weights[i] < weights[j]:
                    sample_index[i], sample_index[j] = sample_index[j], sample_index[i]
    active_datasets = bert_datasets['validation_matched'].select(sample_index)
    active_eval_dataloader = torch.utils.data.DataLoader(active_datasets, batch_size=1, shuffle=False)

    i = 0
    N = 9815
    M = num_samples

    weights = weights.tolist()
    total_loss = 0
    mean_loss = 0
    with open('/home/qinpeixin/InteractiveEvaluation/Classification/weights_accuracy_clip.csv', 'w') as f:
        for batch in active_eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = bert(**batch)

            if metric == 'loss':
                loss = outputs.loss.item()
            elif metric == 'accuracy':
                arg = torch.argmax(outputs.logits, dim=1).item()
                if arg == batch['labels'].item():
                    loss = 0
                else:
                    loss = 1

            m = i + 1
            if m == 9815:
                v = 1
            else:
                v = 1 + (1 / ((N - m + 1) * (weights[sample_index[i]])) - 1) * (N - M) / (N - m)
            total_loss += v * loss
            mean_loss += loss

            if saveFile:
                write_data = [weights[sample_index[i]], loss, v]
                writer = csv.writer(f)
                writer.writerow(write_data)

            i += 1

    unbiased_loss = total_loss / M
    mean_loss = mean_loss / M

    print("unbiased_loss:" + str(unbiased_loss))
    print("mean_loss:" + str(mean_loss))

    trainer = Trainer(
        model=bert,
        args=training_args,
        train_dataset=None,
        eval_dataset=random_eval_dataset,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    return unbiased_loss, metrics


if __name__ == '__main__':
    num_samples_list = [64, 128, 256, 512, 1024]
    seeds = [42, 912, 323, 2023, 1914, 1024, 10086, 406, 401, 318]
    # clips = [None, 0.1, 0.2, 0.3, 0.4, 0.5]
    clips = [0.2]
    with open('/home/qinpeixin/InteractiveEvaluation/Classification/result2.csv', 'w') as f:
        write_head = ['method', 'num_samples', 'clip', 'loss_mean', 'loss_var', 'acc_mean', 'acc_var']
        writer = csv.writer(f)
        writer.writerow(write_head)
        for num_samples in num_samples_list:
            for clip in clips:
                loss_list = []
                acc_list = []
                for seed in seeds:
                    unbiased_loss, metrics_ = main('loss', num_samples, seed, clip, False, True)
                    accuracy_, metric_ = main('accuracy', num_samples, seed, clip, False, True)
                    accuracy_ = 1 - accuracy_
                    # write_data1 = ['active', num_samples, clip, seed, unbiased_loss, accuracy_]
                    # write_data2 = ['random', num_samples, clip, seed, metric_['eval_loss'], metric_['eval_accuracy']]
                    # writer.writerow(write_data1)
                    # writer.writerow(write_data2)
                    loss_list.append(unbiased_loss)
                    acc_list.append(accuracy_)
                loss_mean = np.mean(loss_list)
                loss_var = np.var(loss_list)
                acc_mean = np.mean(acc_list)
                acc_var = np.var(acc_list)
                write_data = ['active', num_samples, clip, loss_mean, loss_var, acc_mean, acc_var]
                writer.writerow(write_data)
