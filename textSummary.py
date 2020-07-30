from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(file_name):
    file = open(file_name, "r",encoding="utf8")
    filedata = file.readlines()
    article = filedata[0].split(". ")  # Convert the paragraph into sentences
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))   # Tokenization
    sentences.pop()

    return sentences

def sentence_similarity(sent1, sent2, stopwords=None, query=[]):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # Evaluate the weighted occurrence frequency of the words

    for w in sent1:
        if w in stopwords:
            continue
        elif w in query:
            vector1[all_words.index(w)] += 10
        vector1[all_words.index(w)] += 1


    for w in sent2:
        if w in stopwords:
            continue
        elif w in query:
            vector1[all_words.index(w)] += 10
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)   # Calculate cosine similarity

def build_similarity_matrix(sentences, stop_words, query):

    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    # Build similarity matrix
    for index1 in range(len(sentences)):
        for index2 in range(len(sentences)):
            if index1 == index2:
                continue
            similarity_matrix[index1][index2] = sentence_similarity(sentences[index1], sentences[index2], stop_words, query )

    return similarity_matrix


def generate_summary(file_name, top_n=5, query=[]):
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences =  read_article(file_name)

    # Build sentence similarity matrix
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words, query)

    # Build sentence similarity graph 
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)

    # Calculate score of sentence using Pagerank Algorithm
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1])) # Join top ranked sentences

    print("Summarize Text: \n", ". ".join(summarize_text))
    output=". ".join(summarize_text)
    f = open('summary-output.txt', 'a+') # Output file
    f.write(output)
    f.close()

# Enter location of input file
data = input("Enter location and name of input file : ")
# Input query
query = input("Enter Query : ").split(" ")

generate_summary( data, 5, query)
