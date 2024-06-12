import spacy as sp
from spacy.tokenizer import Tokenizer
from spacy.lang.en.stop_words import STOP_WORDS

myVocab = sp.load("en_core_web_sm")

yourMessage = input("hi user , please enter your sentence : ")

"""   Tokenize  """
myVocab.tokenize = Tokenizer(myVocab.vocab) # type: ignore
myDoc = myVocab(yourMessage)    
sents = list(myDoc.sents)

"""   StopWord's  """
nothaveSW = [token for token in myDoc if token.text not in STOP_WORDS]
SWinSents = [token for token in myDoc if token.text in STOP_WORDS]


"""   Lemmatization  """
lemmText =" ".join([tk.lemma_ for tk in myDoc])

"""   Part of Tagging  """
pos = [i.pos_ for i in myDoc]
#spacy.explain(token.tag)
print("pot: ",pos)

"""   Named Entity Recognize  """
for token in myDoc.ents:
    print("txt : ",token.text , "label:", token.label_ , "explin :",  sp.explain(token.label_), "\n") # type: ignore


#hi my name is mehrab . i am 23 years old