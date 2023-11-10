from transformers import pipeline
import evaluate
import json
from tqdm import tqdm
import numpy as np

generator = pipeline(model="WillHeld/roberta-base-coqa", cache_dir='/data/qinpeixin/huggingface')

question = ["What color was Cotton?"]
context = [
    "Once upon a time, in a barn near a farm house, there lived a little white kitten named Cotton. Cotton lived high up in a nice warm place above the barn where all of the farmer's horses slept. But Cotton wasn't alone in her little home above the barn, oh no. She shared her hay bed with her mommy and 5 other sisters. All of her sisters were cute and fluffy, like Cotton. But she was the only white one in the bunch. The rest of her sisters were all orange with beautiful white tiger stripes like Cotton's mommy. Being different made Cotton quite sad. She often wished she looked like the rest of her family. So one day, when Cotton found a can of the old farmer's orange paint, she used it to paint herself like them. When her mommy and sisters found her they started laughing. \n\n\"What are you doing, Cotton?!\" \n\n\"I only wanted to be more like you\". \n\nCotton's mommy rubbed her face on Cotton's and said \"Oh Cotton, but your fur is so pretty and special, like you. We would never want you to be any other way\". And with that, Cotton's mommy picked her up and dropped her into a big bucket of water. When Cotton came out she was herself again. Her sisters licked her face until Cotton's fur was all all dry. \n\n\"Don't ever do that again, Cotton!\" they all cried. \"Next time you might mess up that pretty white fur of yours and we wouldn't want that!\" \n\nThen Cotton thought, \"I change my mind. I like being special\"."]
result = generator(question=question, context=context, min_length=1)
print(result)

with open('coqa-dev-v1.0.json', 'r', encoding='utf-8') as fp:
    dev_data = json.load(fp)

dev_data = dev_data['data']
knowledge_list = []
question_list = []
dialog_list = []
label_list = []
for i in range(len(dev_data)):
    for j in range(len(dev_data[i]['questions'])):
        knowledge_list.append(dev_data[i]['story'])
        single_dialog = []
        for k in range(j):
            single_dialog.append(dev_data[i]['questions'][k]['input_text'])
            single_dialog.append(dev_data[i]['answers'][k]['input_text'])
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

knowledge_list = knowledge_list[0:500]
question_list = question_list[0:500]

true_loss = []
for i in tqdm(range(len(knowledge_list))):
    context = knowledge_list[i]
    question = question_list[i]
    result = generator(question=question, context=context, min_length=1)
    res = result['answer'].lower()
    results = rouge.compute(predictions=[res], references=label_list[i])
    true_loss.append(results['rougeL'])

print(np.mean(true_loss))
