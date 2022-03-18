import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import wordcloud
import matplotlib.pyplot as plt

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    special=r'[^a-zA-Z\s]'
    split=r'[a-z][A-Z]'
    text=re.sub(special,'',text)
    text=re.sub(split,' ',text)
    return text

#Stemming the text
def simple_stemmer(text):
    ps=PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokenizer=ToktokTokenizer()
    stopword_list=nltk.corpus.stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


#Build Dataset
data=pd.read_csv('./input/IMDB Dataset.csv')
data.review = data.review.apply(denoise_text)
data.review = data.review.apply(remove_special_characters)
data.review = data.review.apply(remove_stopwords)
data.review = data.review.apply(simple_stemmer)

# sort by sentiment
data = data.sort_values(by='sentiment')


#Tfidf vectorizer
vectorizer=TfidfVectorizer(ngram_range=(3,3))

#transformed reviews
tfidf=vectorizer.fit_transform(data.review)
tv_features = vectorizer.get_feature_names_out()

tfidf_neg = tfidf[:25000].sum(0).getA1()
tfidf_pos = tfidf[25000:].sum(0).getA1()

word_list_neg = sorted(list(zip(tv_features,tfidf_neg)),key=lambda x: x[1],reverse=True)
word_list_pos = sorted(list(zip(tv_features,tfidf_pos)),key=lambda x: x[1],reverse=True)

cloud_neg = wordcloud.WordCloud(background_color="white", max_words=20).generate_from_frequencies(dict(word_list_neg))
cloud_pos = wordcloud.WordCloud(background_color="white", max_words=20).generate_from_frequencies(dict(word_list_pos))

#top 10 tf-idf scores
print('top 10 tf-idf scores (negative): \n', word_list_neg[:10])
print('top 10 tf-idf scores (positive): \n', word_list_pos[:10])

# Display the generated image:
plt.imshow(cloud_neg, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.imshow(cloud_pos, interpolation='bilinear')
plt.axis("off")
plt.show()