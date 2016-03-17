import math

class Scorer:
    B = 6.0
    RHO = 0.5
    def __init__(self, docs):
        self.docs = docs

    def weight(self, paragraph_index):
        if paragraph_index < (-math.log(self.B) / math.log(self.RHO)):
            return math.pow(self.RHO, paragraph_index) * self.B
        else:
            return 1.0

    def calculate_score(self, phrase):
        score = 0.0
        for concept in phrase.concepts:
            for doc in self.docs:
                for index, p in enumerate(doc.paragraphs):
                    score += p.count_frequency(concept) * self.weight(index)

        return score