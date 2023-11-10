import csv

import pandas as pd
from transformer_rankers.trainers import transformer_trainer
from transformer_rankers.datasets import dataset
# from transformer_rankers.negative_samplers import negative_sampling
from transformer_rankers.eval import results_analyses_tools
from transformers.data.data_collator import DefaultDataCollator
from transformers.data.processors.utils import InputFeatures
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForSequenceClassification, ErnieForSequenceClassification, \
    ElectraTokenizerFast, ElectraForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, \
    RobertaForSequenceClassification

import logging
import os
import sys
import torch
import random
import numpy as np


class RandomNegativeSampler():
    """
    Randomly sample candidates from a list of candidates.

    Args:
        candidates: list of str containing the candidates
        num_candidates_samples: int containing the number of negative samples for each query.
    """

    def __init__(self, candidates, num_candidates_samples, seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.name = "RandomNS"

    def sample(self, query_str, relevant_docs):
        """
        Samples from a list of candidates randomly.

        If the samples match the relevant doc,
        then removes it and re-samples.

        Args:
            query_str: the str of the query. Not used here for random sampling.
            relevant_docs: list with the str of the relevant documents, to avoid sampling them as negative sample.

        Returns:
             sampled_documents: list with the str of the sampled documents
             scores: list with the size of sampled_documents containing their respective scores
             was_relevant_sampled: boolean indicating if one of the relevant documents would be sampled (we remove the relevant docs from sampled_documents)
             relevant_rank: -1 if was_relevant_sampled=False and the position of the relevant otherwise.
                    This does not work well if there are multiple relevants, i.e. only the last position is returned
             relevant_docs_scores: list of float with the negative sampling model scores for the list of relevant_docs
        """
        # Since random is not a model per se, it makes sense to return 1 for the scores of the relevant docs.
        relevant_docs_scores = [1 for _ in range(len(relevant_docs))]
        sampled_docs_scores = [0 for _ in range(self.num_candidates_samples)]
        sampled_initial = random.sample(self.candidates, self.num_candidates_samples)
        was_relevant_sampled = False
        relevant_doc_rank = -1
        sampled = []
        for i, d in enumerate(sampled_initial):
            if d in relevant_docs:
                was_relevant_sampled = True
                relevant_doc_rank = i
            else:
                sampled.append(d)

        while len(sampled) != self.num_candidates_samples:
            sampled = [d for d in random.sample(self.candidates, self.num_candidates_samples) if d not in relevant_docs]
        return sampled, sampled_docs_scores, was_relevant_sampled, relevant_doc_rank, relevant_docs_scores


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

data_path = "/home/qinpeixin/InteractiveEvaluation/ClariQ/data/"

train = pd.read_csv(data_path + "train_original.tsv", sep="\t")
valid = pd.read_csv(data_path + "dev.tsv", sep="\t")

train = train[["initial_request", "question"]]
train.columns = ["query", "clarifying_question"]
train = train[~train["clarifying_question"].isnull()]

valid = valid[["initial_request", "question"]]
valid.columns = ["query", "clarifying_question"]
valid = valid[~valid["clarifying_question"].isnull()]

train.to_csv(data_path + "train.tsv", sep="\t", index=False)
valid.to_csv(data_path + "valid.tsv", sep="\t", index=False)

question_bank = pd.read_csv(data_path + "question_bank.tsv", sep="\t")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# The combination of query and question are not that big.
max_seq_len = 50

# Lets use an almost balanced amount of positive and negative samples during training.
average_relevant_per_query = train.groupby("query").count().mean().values[0]

# We could also use random sampling which does not require Anserini.
ns_train = RandomNegativeSampler(list(question_bank["question"].values[1:]),
                                 int(average_relevant_per_query))
ns_val = RandomNegativeSampler(list(question_bank["question"].values[1:]),
                               int(average_relevant_per_query))

# Create the loaders for the dataset, with the respective negative samplers
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataloader = dataset.QueryDocumentDataLoader(train_df=train,
                                             val_df=valid, test_df=valid,
                                             tokenizer=tokenizer, negative_sampler_train=ns_train,
                                             negative_sampler_val=ns_val, task_type='classification',
                                             train_batch_size=48, val_batch_size=1, max_seq_len=max_seq_len,
                                             sample_data=-1, cache_path=data_path)

train_loader, val_loader, test_loader = dataloader.get_pytorch_dataloaders()

# Use BERT (any model that has SequenceClassification class from HuggingFace would work here)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
ltokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
ernie = ErnieForSequenceClassification.from_pretrained('nghuyong/ernie-2.0-large-en', cache_dir='/data/qinpeixin/huggingface')
electra_tokenizer = AutoTokenizer.from_pretrained("google/electra-large-discriminator", cache_dir='/data/qinpeixin/huggingface')
electra = AutoModelForSequenceClassification.from_pretrained("google/electra-large-discriminator", cache_dir='/data/qinpeixin/huggingface')

# Instantiate trainer that handles fitting.
trainer = transformer_trainer.TransformerTrainer(model=model,
                                                 train_loader=train_loader,
                                                 val_loader=val_loader, test_loader=test_loader,
                                                 num_ns_eval=int(average_relevant_per_query),
                                                 task_type="classification", tokenizer=tokenizer,
                                                 validate_every_epochs=1, num_validation_batches=-1,
                                                 num_epochs=1, lr=5e-7, sacred_ex=None)

trainer2 = transformer_trainer.TransformerTrainer(model=ernie,
                                                  train_loader=train_loader,
                                                  val_loader=val_loader, test_loader=test_loader,
                                                  num_ns_eval=int(average_relevant_per_query),
                                                  task_type="classification", tokenizer=tokenizer,
                                                  validate_every_epochs=1, num_validation_batches=-1,
                                                  num_epochs=1, lr=5e-7, sacred_ex=None)

trainer3 = transformer_trainer.TransformerTrainer(model=electra,
                                                  train_loader=train_loader,
                                                  val_loader=val_loader, test_loader=test_loader,
                                                  num_ns_eval=int(average_relevant_per_query),
                                                  task_type="classification", tokenizer=electra_tokenizer,
                                                  validate_every_epochs=1, num_validation_batches=-1,
                                                  num_epochs=1, lr=5e-7, sacred_ex=None)

# Train (our validation eval uses the NS sampling procedure)
trainer.fit()
all_logits, all_labels, all_softmax_logits = trainer.predict(val_loader)

trainer3.fit()
all_logits2, all_labels2, all_softmax_logits2 = trainer2.predict(val_loader)

# trainer3.fit()
# all_logits3, all_labels3, all_softmax_logits3 = trainer3.predict(val_loader)

label_list = []
logits_list = []
logits_list2 = []
uncertainty = []
# logits_list3 = []
for i in range(len(all_labels)):
    label_list += all_labels[i]
    logits_list += all_softmax_logits[i]
    logits_list2 += all_softmax_logits2[i]

    # logits_list3 += all_softmax_logits3[i]

for i in range(len(logits_list2)):
    # uncertainty.append(-(logits_list2[i] * np.log(logits_list2[i]) + (1-logits_list2[i]) * np.log(1-logits_list2[i])) / np.log(2))
    if (logits_list2[i] > 0.5 and logits_list[i] > 0.5) or (logits_list2[i] <= 0 and logits_list[i] <= 0.5):
        uncertainty.append(0)
    else:
        uncertainty.append(-(logits_list2[i] * np.log(logits_list2[i]) + (1-logits_list2[i]) * np.log(1-logits_list2[i])) / np.log(2))

accuracy = 0
for i in range(len(label_list)):
    if (label_list[i] == 1 and logits_list[i] > 0.5) or (label_list[i] == 0 and logits_list[i] <= 0.5):
        accuracy += 1
print(accuracy / len(label_list))
# exit(0)
print()

# num_samples_list = [16, 32, 64, 128, 256, 512]
num_samples_list = [5, 10, 15, 20, 25, 30]
# a
# for n in num_samples_list:
#     hmce_list = pd.Series(uncertainty).sort_values(ascending=False).index[:n]
#     label2 = []
#     predict_label = []
#     for i in range(len(logits_list2)):
#         label2.append(logits_list2[i])
#     for idx in hmce_list:
#         label2[idx] = label_list[idx]
#
#     hmce_acc = 0
#     for i in range(len(label_list)):
#         if (logits_list[i] > 0.5 and label2[i] > 0.5) or (logits_list[i] <= 0.5 and label2[i] <= 0.5):
#             hmce_acc += 1
#     print(hmce_acc / len(label_list))
# exit(0)
# num_samples_list = [200]
seeds = [42, 912, 323, 2023, 1914, 1024, 10086, 406, 401, 318, 1650, 411, 5515, 114, 514, 2, 10, 5,
         200, 1797, 21, 1637, 10124, 3912, 321, 8914, 8003, 2083, 165, 184, 708, 1499, 5523, 8551,
         8927, 6807, 98004, 988, 1708, 928, 81, 17647, 2809, 87, 29, 1411, 74, 14174, 14117, 90543,
         27231, 8480, 1825, 2001, 4769, 2377, 9784, 1107, 1456, 6348, 1838, 3285, 53, 4931, 6027]
clips = [0.2]
stratify = False

with open('/home/qinpeixin/InteractiveEvaluation/ClariQ/tmp/tmp.csv', 'w') as f:
    write_head = ['method', 'num_samples', 'clip', 'acc_mean', 'acc_var', 'mse']
    writer = csv.writer(f)
    writer.writerow(write_head)
    print(write_head)
    for num_samples in num_samples_list:
        for clip in clips:
            random_list = []
            active_list = []
            active_error = []
            random_error = []
            true_acc = accuracy / len(label_list)
            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                randomSampler = torch.utils.data.RandomSampler(range(len(label_list)), num_samples=num_samples)
                random_sample_index = [j for j in randomSampler]

                random_accuracy = 0
                for i in random_sample_index:
                    if (label_list[i] == 1 and logits_list[i] > 0.5) or (label_list[i] == 0 and logits_list[i] <= 0.5):
                        random_accuracy += 1

                random_list.append(random_accuracy / num_samples)
                random_error.append(((random_accuracy / num_samples) - true_acc)**2)

                weights = []
                for i in range(len(label_list)):
                    if logits_list[i] > 0.5:
                        weights.append(1 - logits_list2[i])
                    else:
                        weights.append(logits_list2[i])

                sum = 0
                for weight in weights:
                    sum += weight

                for i in range(len(weights)):
                    weights[i] = weights[i] / sum

                if clip is not None and stratify is False:
                    for i in range(len(weights)):
                        if weights[i] < clip / len(label_list):
                            # weights[i] = clip / len(label_list)
                            weights[i] = 0

                weights = torch.tensor(weights, device='cuda')
                ActiveSampler = torch.utils.data.WeightedRandomSampler(weights=weights, num_samples=num_samples,
                                                                       replacement=False)
                active_sample_index = [j for j in randomSampler]

                for i in range(len(active_sample_index)):
                    for j in range(i + 1, len(active_sample_index)):
                        if weights[i] < weights[j]:
                            active_sample_index[i], active_sample_index[j] = active_sample_index[j], \
                                                                             active_sample_index[i]

                N = len(label_list)
                M = num_samples

                total_loss = 0
                weights = weights.tolist()
                for i in range(len(active_sample_index)):
                    j = active_sample_index[i]
                    m = i + 1
                    # m = M
                    if (label_list[j] == 1 and logits_list[j] > 0.5) or (label_list[j] == 0 and logits_list[j] <= 0.5):
                        loss = 0
                    else:
                        loss = 1
                    v = 1 + (1 / ((N - m + 1) * (weights[j])) - 1) * (N - M) / (N - m)
                    total_loss += v * loss

                total_loss = total_loss / M
                active_list.append(1 - total_loss)
                active_error.append(((1 - total_loss)-true_acc)**2)
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
            writer.writerow(write_data2)
            print(write_data1)
            print(write_data2)
