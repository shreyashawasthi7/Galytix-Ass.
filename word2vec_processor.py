import gensim
import gdown

class Word2VecProcessor:
    def __init__(self, download_url, binary_file_path, flat_file_path):
        self.download_url = download_url
        self.binary_file_path = binary_file_path
        self.flat_file_path = flat_file_path
        self.word2vec_model = None

    def download_pretrained_vectors(self):
        try:
            gdown.download(self.download_url, self.binary_file_path, quiet=False)
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def load_and_save_vectors(self, limit=1000000):
        try:
            # Load Word2Vec vectors from binary format
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.binary_file_path, binary=True, limit=limit)
            # Save them as a flat file
            self.word2vec_model.save_word2vec_format(self.flat_file_path, binary=False)
            return True
        except Exception as e:
            print(f"Load and save vectors failed: {e}")
            return False
# Instantiate Word2VecProcessor with download URL and file paths
processor = Word2VecProcessor(
    download_url='https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM',
    binary_file_path='word2vec.bin',
    flat_file_path='vectors.txt'
)

# Download and save the pretrained vectors
downloaded = processor.download_pretrained_vectors()
if downloaded:
    loaded_and_saved = processor.load_and_save_vectors()
    if loaded_and_saved:
        print("Pretrained vectors loaded and saved successfully.")
    else:
        print("Failed to load and save pretrained vectors.")
else:
    print("Download of pretrained vectors failed.")
