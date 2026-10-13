# NLP Lab Programs

## 1. Given a text document, perform the following tasks:
i. Count the number of words and the number of distinct words.

ii. Using list addition, set and sorted operations, compute the vocabulary of the sentences `sent1` to `sent8`.

iii. Define `sent` to be the list of words `['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']`. Write code to:

a. Print all words beginning with `sh`.

b. Print all words longer than four characters.

```python
import nltk
from nltk.tokenize import word_tokenize

text = "She sells sea shells by the sea shore."
words = [w.lower() for w in word_tokenize(text) if w.isalpha()]

print("Total words:", len(words))
print("Distinct words:", len(set(words)))

sent1 = ['she', 'sells', 'sea']
sent2 = ['sea', 'shore']
sent3 = ['shells', 'by']
sent4 = ['the', 'sea']
sent5 = ['shore', 'she']
sent6 = ['sells', 'shells']
sent7 = ['by', 'the']
sent8 = ['sea', 'shore']

vocab = sorted(set(sent1 + sent2 + sent3 + sent4 +
                   sent5 + sent6 + sent7 + sent8))
print("Vocabulary:", vocab)

sent = ['she', 'sells', 'sea', 'shells', 'by', 'the', 'sea', 'shore']
print("Starts with sh:", [w for w in sent if w.startswith('sh')])
print("Length > 4:", [w for w in sent if len(w) > 4])
```

## 2. Write a Python program to find the Conditional Frequency Distribution (CFD) of words in different genres of the Brown Corpus.

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

cfd.tabulate(
    conditions=genres,
    samples=['love', 'man', 'woman']
)
```

## 3. Write a Python program to perform morphological analysis and generation of words using root words and suffixes. Demonstrate the inverse process of recovering the root word from a generated word.

```python
root = "play"
suffixes = ["s", "ed", "ing", "er"]

print("Generated words:")
for s in suffixes:
    print(root + s)

word = "playing"
for s in suffixes:
    if word.endswith(s):
        print("Root:", word[:-len(s)])
        break
```

## 4. Write a Python program to demonstrate word morphology using an Add-Delete table for generating word forms and recovering the root word.

```python
table = {
    "play": ["s", "ed", "ing"],
    "happy": ["ness", "ly"]
}

print("Generated words:")
for root, suffixes in table.items():
    for s in suffixes:
        print(root + s)

word = "played"
suffix = "ed"

if word.endswith(suffix):
    print("Root:", word[:-len(suffix)])
```

## 5. Write a Python program to implement tokenization and sentence segmentation using regular expressions and NLTK.

```python
import nltk
import re

text = "Hello world! NLP is interesting. Let's learn it."

print("Sentences:", nltk.sent_tokenize(text))
print("Words:", re.findall(r"\b\w+\b", text))
```

## 6. Write a Python program to perform stemming and lemmatization on a given set of words using NLTK.

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')

words = ['playing', 'played', 'cars', 'better']
ps = PorterStemmer()
lm = WordNetLemmatizer()

for w in words:
    print(w, "Stem:", ps.stem(w),
          "Lemma:", lm.lemmatize(w))
```

## 7. Write a Python program to calculate bigrams from a given corpus and calculate the probability of a sentence using bigram probabilities.

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
print("Sentence probability:", p)
```

## 8. Write a Python program to implement an N-Gram Language Model and calculate the conditional probability of a word given its preceding context.

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

## 9. Write a Python program to calculate the emission and transition probability matrices required for Parts-of-Speech tagging using a Hidden Markov Model (HMM).

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

print("Transition P(NOUN|DET):", tc['DET'].freq('NOUN'))
print("Emission P(time|NOUN):", ec['NOUN'].freq('time'))
```

## 10. Write a Python program to find the Parts-of-Speech tags of words in a given sentence using Viterbi decoding with a Hidden Markov Model.

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

## 11. Develop a context-sensitive POS tagger using HMM and trigram features. Demonstrate how the size of the training corpus affects POS tagging.

```python
import nltk
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger

nltk.download('brown')
nltk.download('universal_tagset')

data = nltk.corpus.brown.tagged_sents(
    categories='news', tagset='universal'
)

for size in [100, 300, 500]:
    train = data[:size]

    u = UnigramTagger(train)
    b = BigramTagger(train, backoff=u)
    t = TrigramTagger(train, backoff=b)

    print("Training size:", size)
    print(t.tag("the market is growing".split()))
```

## 12. Write a Python program to implement a feature extractor and build a robust, context-sensitive document classification model using TF-IDF and Naive Bayes.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

docs = [
    "team won the match",
    "player scored a goal",
    "new AI model released",
    "computer learning technology"
]

labels = ['sports', 'sports', 'tech', 'tech']

tfidf = TfidfVectorizer(ngram_range=(1, 2))
X = tfidf.fit_transform(docs)

model = MultinomialNB()
model.fit(X, labels)

test = tfidf.transform(["AI technology model"])
print("Predicted class:", model.predict(test))
```

## 13. Write a Python program to implement feature selection using the chi-square test, train a document classification model on different training corpus sizes, and observe the effect of corpus size on prediction.

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

    print("Corpus size:", size)
    print("Prediction:", model.predict(X[:1]))
```
