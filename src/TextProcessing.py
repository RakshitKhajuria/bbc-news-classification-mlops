from keras.preprocessing.text import text_to_word_sequence
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from src.exception import CustomException
import sys
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocessDataset(train_text):
    try:
       
        #word tokenization using text-to-word-sequence
        train_text= str(train_text)
        tokenized_train_set = text_to_word_sequence(train_text,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
            
        #stop word removal
        stop_words = set(stopwords.words('english'))
        stopwordremove = [i for i in tokenized_train_set if not i in stop_words]
            
        
        #join words into sentence
        stopwordremove_text = ' '.join(stopwordremove)
            
            
        #remove numbers
        numberremove_text = ''.join(c for c in stopwordremove_text if not c.isdigit())
        
            
        #--Stemming--
        stemmer= PorterStemmer()

        stem_input=nltk.word_tokenize(numberremove_text)
        stem_text=' '.join([stemmer.stem(word) for word in stem_input])
            
            
        lemmatizer = WordNetLemmatizer()

        def get_wordnet_pos(word):
            """Map POS tag to first character lemmatize() accepts"""
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

            return tag_dict.get(tag, wordnet.NOUN)

        lem_input = nltk.word_tokenize(stem_text)
        lem_text= ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in lem_input])
            
        return lem_text
    except Exception as e:
        raise CustomException(e,sys)