class Phrase:
    def __init__(self, id, is_np, parentId, sentence_length, word_length, content, concepts):
        self.id = id
        self.is_np = is_np
        self.content = content
        self.concepts = concepts
        self.parentId = parentId
        self.sentence_length = sentence_length
        self.word_length = word_length
        self._score = 0

    def __str__(self):
        return "id: %d, isNP: %s, parentId: %d, content: %s" % (self.id, self.is_np, self.parentId, self.content)

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self.score = value