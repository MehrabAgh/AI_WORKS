import nltk
from nltk.tokenize import sent_tokenize , word_tokenize , regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer , PorterStemmer , WordNetLemmatizer

"""     Get Input       """
yourMessage = input("\nhi user , please enter your sentence : ")

"""     Download Model Tokenize     """
# nltk.download('punkt')

"""  Sentense Tokenize  """
newST = sent_tokenize(yourMessage)
print("\n sent_tokenize:",newST, "\n")

"""  Word Tokenize  """
yourMessage = yourMessage.lower()
newWT = yourMessage.split()
print("word tokenize with split : ",newWT, "\n")

yourMessage = yourMessage.lower()
newWT = word_tokenize(yourMessage)
print("word tokenize with word_tokenize : ",newWT , "\n")

newWTR = regexp_tokenize(yourMessage , r"[0-9]")
print("word tokenize with Regexp : " , newWT, "\n")

"""   StopWord's  """
# nltk.download("stopwords")
stop_words = set(stopwords.words('english'))
# print("StopWords: ",stop_words)
newWords = [word for word in newWT if word.lower() not in stop_words]
print("Remove StopWords: " , newWords)

"""   Stemming & Lemmatization  """
PS = PorterStemmer()
LS = LancasterStemmer()
check_Stem = [PS.stem(token) for token in newWords]
print("Stemmed : ", check_Stem)

"""   Part of Tagging  """

"""   Named Entity Recognize  """

# hi my Name is Mehrab. i am 23 years_old , mehrab is developer