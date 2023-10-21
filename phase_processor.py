import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.word2vec_processor import Word2VecProcessor

class PhrasesProcessor:
    def __init__(self, phrases_path, vectors_path):
        self.phrases_data = pd.read_csv(phrases_path)
        self.word2vec_processor = Word2VecProcessor(vectors_path)
        self.embeddings = None

    def preprocess_phrases(self):
        self.phrases_data['tokens'] = self.phrases_data['phrase'].apply(lambda x: x.lower().split())

    def calculate_embeddings(self):
        self.embeddings = []
        for tokens in self.phrases_data['tokens']:
            phrase_embedding = np.mean([self.word2vec_processor.calculate_word_embedding(word) for word in tokens], axis=0)
            self.embeddings.append(phrase_embedding)
        self.embeddings = np.array(self.embeddings)

    def calculate_batch_similarity(self):
        similarities = cosine_similarity(self.embeddings, self.embeddings)
        return similarities

    def find_closest_match(self, user_input):
        user_input_tokens = user_input.lower().split()
        user_input_embedding = np.mean([self.word2vec_processor.calculate_word_embedding(word) for word in user_input_tokens], axis=0)

        if user_input_embedding is None:
            return None, None

        similarities = cosine_similarity([user_input_embedding], self.embeddings)
        closest_match_index = similarities.argmax()

        closest_match = self.phrases_data.loc[closest_match_index, 'phrase']
        similarity_score = similarities[0][closest_match_index]

        return closest_match, similarity_score

    def calculate_similarity_matrix(self):
        self.preprocess_phrases()
        self.calculate_embeddings()
        similarities = self.calculate_batch_similarity()
        return similarities

# Example usage:
if __name__ == "__main__":
    phrases_path = 'data/phrases.csv'  # Replace with the actual file path to your phrases data
    vectors_path = 'vectors.csv'  # Replace with the actual file path to your Word2Vec vectors
    user_input = "Your user input phrase here"

    phrases_processor = PhrasesProcessor(phrases_path, vectors_path)

    # Calculate similarity matrix for all phrases
    similarity_matrix = phrases_processor.calculate_similarity_matrix()
    print(similarity_matrix)

    # Find the closest match to the user input phrase
    closest_match, similarity_score = phrases_processor.find_closest_match(user_input)
    print(f"Closest Match: {closest_match}, Similarity Score: {similarity_score}")
