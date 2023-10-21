from .phrases_processor import PhrasesProcessor

def main():
    phrases_path = 'data/phrases.csv'  # Replace with the actual file path to your phrases data
    vectors_path = 'vectors.txt'  # Replace with the actual file path to your Word2Vec vectors
    user_input = "Your user input phrase here"

    phrases_processor = PhrasesProcessor(phrases_path, vectors_path)

    # Calculate similarity matrix for all phrases
    similarity_matrix = phrases_processor.calculate_similarity_matrix()
    print(similarity_matrix)

    # Find the closest match to the user input phrase
    closest_match, similarity_score = phrases_processor.find_closest_match(user_input)
    print(f"Closest Match: {closest_match}, Similarity Score: {similarity_score}")

if __name__ == "__main__":
    main()
