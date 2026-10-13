# NLP Lab

## 1. Text Processing, Vocabulary and Set Operations

```python
import nltk
from nltk.tokenize import word_tokenize

text = "She sells sea shells by the sea shore."
words = [w.lower() for w in word_tokenize(text) if w.isalpha()]

print("Total:", len(words))
print("Distinct:", len(set(words)))
print("Vocabulary:", sorted(set(words)))

s1 = ['she', 'sells', 'sea']
s2 = ['sea', 'shore']

print("Union:", sorted(set(s1) | set(s2)))
print("Intersection:", sorted(set(s1) & set(s2)))
print("Difference:", sorted(set(s1) - set(s2)))

sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
print("Starts with sh:", [w for w in sent if w.startswith('sh')])
print("Length > 4:", [w for w in sent if len(w) > 4])
```

## 2. Conditional Frequency Distribution

```python
import nltk
from nltk.corpus import brown

nltk.download('brown')

genres = ['news', 'romance', 'humor']
cfd = nltk.ConditionalFreqDist(
    (g, w.lower())
    for g in genres
    for w in brown.words(categories=g)
    if w.isalpha()
)

cfd.tabulate(conditions=genres, samples=['love', 'man', 'woman'])
```

## 3. Morphological Analysis and Generation

```python
root = "play"
suffixes = ["s", "ed", "ing", "er"]

for s in suffixes:
    print(root + s)

word = "playing"
for s in suffixes:
    if word.endswith(s):
        print("Root:", word[:-len(s)])
        break
```

## 4. Add-Delete Table

```python
table = {
    "play": ["s", "ed", "ing"],
    "happy": ["ness", "ly"]
}

for root, suffixes in table.items():
    for s in suffixes:
        print(root + s)

word = "played"
suffix = "ed"

if word.endswith(suffix):
    print("Root:", word[:-len(suffix)])
```

## 5. Tokenization and Sentence Segmentation

```python
import nltk
import re

text = "Hello world! NLP is interesting. Let's learn it."

print("Sentences:", nltk.sent_tokenize(text))
print("Words:", re.findall(r"\b\w+\b", text))
```

## 6. Stemming and Lemmatization

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')

words = ['playing', 'played', 'cars', 'better']
ps = PorterStemmer()
lm = WordNetLemmatizer()

for w in words:
    print(w, ps.stem(w), lm.lemmatize(w))
```

## 7. Bigrams and Sentence Probability

```python
from nltk.util import bigrams
from collections import Counter

words = ['i', 'love', 'nlp']
bg = list(bigrams(words))
counts = Counter(bg)
uni = Counter(words)

p = (counts[('i', 'love')] / uni['i']) * \
    (counts[('love', 'nlp')] / uni['love'])

print("Bigrams:", bg)
print("Probability:", p)
```

## 8. N-Gram Language Model

```python
from nltk.util import ngrams
from collections import Counter

words = ['i', 'love', 'nlp', 'i', 'love', 'python']
N = 2

ng = list(ngrams(words, N))
counts = Counter(ng)
context = Counter(words[:-1])

print("N-grams:", ng)
print("P(love|i):", counts[('i', 'love')] / context['i'])
```

## 9. HMM Transition and Emission Probabilities

```python
import nltk
from nltk.probability import ConditionalFreqDist

nltk.download('brown')
nltk.download('universal_tagset')

data = nltk.corpus.brown.tagged_sents(
    categories='news', tagset='universal'
)[:100]

trans, emis = [], []

for sent in data:
    sent = [('<s>', 'START')] + sent
    for i in range(1, len(sent)):
        trans.append((sent[i-1][1], sent[i][1]))
        emis.append((sent[i][1], sent[i][0].lower()))

tc = ConditionalFreqDist(trans)
ec = ConditionalFreqDist(emis)

print("Transition:", tc['DET'].freq('NOUN'))
print("Emission:", ec['NOUN'].freq('time'))
```

## 10. POS Tagging using Viterbi

```python
import nltk
from nltk.tag import hmm

nltk.download('brown')
nltk.download('universal_tagset')

data = nltk.corpus.brown.tagged_sents(
    categories='news', tagset='universal'
)

trainer = hmm.HiddenMarkovModelTrainer()
model = trainer.train_supervised(data[:300])

sentence = "the market is good".split()
print(model.tag(sentence))
```

## 11. Context-Sensitive POS Tagger using Trigrams

```python
import nltk
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

nltk.download('brown')
nltk.download('universal_tagset')

data = nltk.corpus.brown.tagged_sents(
    categories='news', tagset='universal'
)[:500]

u = UnigramTagger(data)
b = BigramTagger(data, backoff=u)
t = TrigramTagger(data, backoff=b)

print(t.tag("the market is growing".split()))
```

## 12. Feature Extraction and Document Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

docs = [
    "team won the match", "player scored a goal",
    "new AI model released", "computer learning technology"
]
labels = ['sports', 'sports', 'tech', 'tech']

tfidf = TfidfVectorizer(ngram_range=(1, 2))
X = tfidf.fit_transform(docs)

model = MultinomialNB()
model.fit(X, labels)

test = tfidf.transform(["AI technology model"])
print(model.predict(test))
```

## 13. Feature Selection and Training Corpus Size

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB

docs = [
    "team won match", "player scored goal",
    "sports team", "football match",
    "AI model", "computer AI",
    "machine learning", "new technology"
]

labels = ['sports'] * 4 + ['tech'] * 4

for size in [4, 6, 8]:
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(docs[:size])

    selector = SelectKBest(chi2, k=min(3, X.shape[1]))
    X = selector.fit_transform(X, labels[:size])

    model = MultinomialNB()
    model.fit(X, labels[:size])

    print(size, model.predict(X[:1]))
```
