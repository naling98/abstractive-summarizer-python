class Paragraph:
    def __init__(self, concept_frequency):
        self.concept_frequency = concept_frequency

    def count_frequency(self, concept):
        if self.concept_frequency.has_key(concept):
            return self.concept_frequency[concept]
        else:
            return 0.0

