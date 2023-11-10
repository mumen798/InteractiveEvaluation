import evaluate

bleu = evaluate.load("bleu")

predictions = ["white"]
references = ["white"]

results = bleu.compute(predictions=predictions, references=references, max_order=1)
print(results)