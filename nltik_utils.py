import nltk
# at first time you have to download it because it necessary
# because following is package that is necessary for tokenize
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    pass

# For Tokenize Function
a = "Hey How is going everything?"
print(a)
a = tokenize(a)
print(a)

# For Steming
# words = ["Organize", "organizer", "Organizing"]
# words = ["teach", "teacher", "teachers", "teaching"]
words = ["star", "stars"]
stemm_words = [stem(w) for w in words]
print(stemm_words)