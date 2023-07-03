# Определяет по тексту отзыва на фильм положительный отзыв или отрицательный
import math
import numpy as np
import pandas as pd
import sys
import os
from collections import Counter

# Удаляет символы из строки
def clean_line(line, symbols):
    for symbol in symbols:
        line = line.replace(symbol, "")
    return line.replace("  ", " ")

# Разбивает строку массив, удаляет пустые элементы
def split_tokens(line):
    tokens = list(set(line.split(" ")))
    return tokens

# Формирует словарь
def get_vocab(tokens):
    vocab = set()
    for sent in tokens:
        for word in sent:
            vocab.add(word)
    return list(vocab)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

number_of_lines = 10000
number_of_test_lines = 5000
stop_symbols = ["<br", "/>", ".", ",", "?", "!"]

raw_reviews = []
raw_labels = []

data = pd.read_csv(os.environ['MRP_DATASET_PATH'])
raw_reviews = data['review'].head(number_of_lines)
raw_labels = data['sentiment'].head(number_of_lines)

for i, raw_review in enumerate(raw_reviews):
    raw_review = raw_review.lower()
    raw_reviews[i] = clean_line(raw_review, stop_symbols)

# Набор строк состоящих из массива слов
tokens = list(map(split_tokens, raw_reviews))
vocab = get_vocab(tokens[:number_of_lines-number_of_test_lines])

# Формируем асоциативный массив, слово - его позация в словаре
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i

# Так как ниже вместо dot мы используем sum для вычисления layer_1
# Eсли бы мы хотели использовать dot input_dataset было бы необходимо преобразовать к виду [0,1,0,1,0,0...],
# размер массива должен был бы равняться размеру словаря
input_dataset = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
        except:
            ""
    input_dataset.append(list(set(sent_indices)))
    
target_dataset = list()
for label in raw_labels:
    if label == 'positive':
        target_dataset.append(1)
    else:
        target_dataset.append(0)

alpha = 0.01
iterations = 2
hidden_size = 100

weights_0_1 = 0.2 * np.random.random((len(vocab), hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1

correct, total = (0, 0)
for iter in range(iterations):
    for i in range(len(input_dataset) - number_of_test_lines):
        x, y = (input_dataset[i], target_dataset[i])
        
        # Берем матрицу (100, 1) для каждого слова в предложении и сумираем их веса,
        # так как нету разницы взять массив из нулей и едениц и перемножить на веса
        # или же просто взять веса тех элементов которые присутствуют в предложении и сложить их
        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
        
        layer_2_delta = layer_2 - y
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)
        
        # Балансируем вес синопсов для каждого слова в предложении
        weights_0_1[x] -= layer_1_delta * alpha
        
        # Тоже что и (layer_1 * layer_2_delta[0] * alpha).reshape(weights_1_2.shape[0], -1), только без доп. действий по приведению размерности
        weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha
        
        if (np.abs(layer_2_delta) < 0.5):
            correct += 1
        total += 1
        
        # Выводить информацию после каждого 10 элемента
        if (i % 10 == 1):
            progress = str(i / float(len(input_dataset) - number_of_test_lines))
            sys.stdout.write('\rIter:' + str(iter) + ' Progress:' + progress[2:4] + '.' + progress[4:6] + '% Training Accurancy:' + str(correct / float(total)) + '%')
    print()

correct, total = (0, 0)
for i in range(len(input_dataset) - number_of_test_lines, len(input_dataset)):
    x = input_dataset[i]
    y = target_dataset[i]
    
    layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))
    
    if (np.abs(layer_2 - y) < 0.5):
        correct += 1
    total += 1

print("Test Accuracy:" + str(correct / float(total)))

# Побочный эфект
def analogy(positive=['terrible', 'good'], negative=['bad']):
    query_vect = np.zeros(len(weights_0_1[0]))
    for word in positive:
        query_vect += weights_0_1[word2index[word]]
    for word in negative:
        query_vect -= weights_0_1[word2index[word]]
        
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - query_vect
        squared_difference = raw_difference * raw_difference
        scores[word] -= math.sqrt(sum(squared_difference))
    
    return scores.most_common(10)[1:]

print(analogy(['elizabeth', 'he'], ['she']))
print(analogy(['terrible', 'good'], ['bad']))