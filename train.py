import json
import numpy



from nltik_utils import bag_of_words, stem, tokenize
with open('intent.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattren in intent['patterns']:
        token = tokenize(pattren)
        all_words.extend(token)
        xy.append((token, tag))

ignore_words = ['?', '.', ',', '!']
# all_words = [stem(w) for w in all_words if w  not in ignore_words]
all_words = [
    stem(w)
    for w in all_words
    if w not in ignore_words
]
# for printing all_words 
# print(all_words)
#this is used for sorted all_words and set is used for avoid duplicate words
# means that set is used for printing unique wwords in all_words and tags
# all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(all_words)
print(tags)


x_trains = []
y_trains = []

for (pattren_sentence, tag) in xy:
    bag = bag_of_words(pattren_sentence, all_words)
    x_trains.append(bag)

    label = tags.index(tag)
    y_trains.append(label)

x_trains = numpy.array(x_trains)
y_trains = numpy.array(y_trains)
