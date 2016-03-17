from xml.dom.minidom import parse
import xml.dom.minidom as minidom
import numpy as np

from phrase import Phrase
from paragraph import Paragraph
from document import Document

def load_phrases(phrase_file_path):
    DOMTree = minidom.parse(phrase_file_path)
    collection = DOMTree.documentElement

    if collection.hasAttribute("phrases"):
        print "Root element: %s" % collection.getAttribute("phrases")

    nounPhrases = {}
    verbPhrases = {}

    phraseElements = collection.getElementsByTagName("phrase")

    for phraseElement in phraseElements:
        id = int(phraseElement.getAttribute("id"))
        isNP = phraseElement.getAttribute("type") == "NP"
        parentId = int(phraseElement.getAttribute("parentId"))
        sentenceLength = int(phraseElement.getAttribute("sentenceLength"))
        word_length = int(phraseElement.getAttribute("length"))
        content = phraseElement.getElementsByTagName("content")[0].childNodes[0].data

        conceptElement = phraseElement.getElementsByTagName("concepts")[0]
        if len(conceptElement.childNodes) == 0:
            concepts = []
        else:
            concepts = conceptElement.childNodes[0].data.split(':')

        phrase = Phrase(id, isNP, parentId, sentenceLength, word_length, content, concepts)

        if isNP:
            nounPhrases[id] = phrase
        else:
            verbPhrases[id] = phrase

    return nounPhrases, verbPhrases

def load_corefs(corefs_file_path):
    corefs = {}
    f = open(corefs_file_path, "r")
    for line in f:
        data = line.strip().split(":")
        key = data[0]
        value = data[1]
        refs = value.split("|")
        corefs[key] = refs

    return corefs

def load_docs(docs_file_path):
    docs = []

    DOMTree = minidom.parse(docs_file_path)
    collection = DOMTree.documentElement

    docElements = collection.getElementsByTagName("doc")

    for docEl in docElements:
        paragraphs = []
        paragraphElements = docEl.getElementsByTagName("p")

        for pEl in paragraphElements:

            concepts = {}
            conceptElements = pEl.getElementsByTagName("concept")
            for cEl in conceptElements:
                name = cEl.getAttribute("name")
                freq = int(cEl.getAttribute("freq"))
                concepts[name] = freq

            p = Paragraph(concepts)
            paragraphs.append(p)
        doc = Document(paragraphs)
        docs.append(doc)

    return docs

def load_indicator_matrix(indicator_matrix_file_path, max_np_id, max_vp_id):
    indicator_matrix = np.zeros((max_np_id+1, max_vp_id+1))
    f = open(indicator_matrix_file_path, "r")

    for line in f:
        data = line.split(":")
        np_id = int(data[0].split("_")[1])
        vp_id = int(data[1].split("_")[1])
        value = int(data[2])

        indicator_matrix[np_id, vp_id] = value

    return indicator_matrix

def load_model(folder):
    nounPhrases, verbPhrases = load_phrases("%s/phrases.xml" % folder)
    corefs = load_corefs("%s/corefs.txt" % folder)
    docs = load_docs("%s/docs.xml" % folder)
    matrix = load_indicator_matrix("%s/indicator_matrix.txt" % folder, max(nounPhrases.keys()), max(verbPhrases.keys()))

    return nounPhrases, verbPhrases, corefs, docs, matrix
