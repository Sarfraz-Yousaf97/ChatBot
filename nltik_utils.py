import nltk
import numpy 
# at first time you have to download it because it necessary
# because following is package that is necessary for tokenize
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# For Tokenize Function
# a = "Hey How is going everything?"
# print(a)
# a = tokenize(a)
# print(a)



def stem(word):
    return stemmer.stem(word.lower())


# For Steming
# words = ["Organize", "organizer", "Organizing"]
# words = ["teach", "teacher", "teachers", "teaching"]
# words = ["star", "stars"]
# stemm_words = [stem(w) for w in words]
# print(stemm_words)




def bag_of_words(tokenized_sentence, all_words):
    """
        sentence = "Hello How are you" # tokenized suntence
        words = "hi hello I you bye thank cool we" all_words
        [0. 1. 0. 1. 0. 0. 0. 0.] #output
    """


    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = numpy.zeros(len(all_words), dtype= numpy.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag

# sentence = "Hello How are you"
# a = tokenize(sentence) # its tokenixe the above string
# words = "hi hello I you bye thank cool we"
# b = tokenize(words)
# # following we calling above function with its require arguments
# bog = bag_of_words(a, b)
# print(bog)


