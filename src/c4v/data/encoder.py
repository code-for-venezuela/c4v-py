import tensorflow_datasets as tfds

# TODO: import Bert / add it to the toml file
# from official.nlp import bert


class Encoder:

    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        pass

    def load(self):
        encoder = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.vocab_path)

        ids = encoder.encode("hello world")
        text = encoder.decode([1, 2, 3, 4])
        print(f'example: ${ids}\n result: ${text}')
        return encoder

    # def create_tokenizer(self, vocab_path):
    #     self.tokenizer = bert.tokenization.FullTokenizer(
    #         vocab_file=vocab_path,
    #         do_lower_case=True)
    #
    #     print("Vocab size:", len(self.tokenizer.vocab))
    #
    #     #tokenize "Hello TensorFlow!"
    #     tokens = self.tokenizer.tokenize("Hello TensorFlow!")
    #     print(tokens)
    #     ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #     print(ids)


if __name__ == "__main__":
    # path = os.path.join(PROCESSED_DATA_FOLDER, 'spanish_vocabulary')
    path = "/Users/marianela/Documents/localrepo/c4v-py/tests/data/vocab_500.subwords"
    enco = Encoder(path)
