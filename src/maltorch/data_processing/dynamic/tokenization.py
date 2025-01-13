"""
Trizna, D. and Demetrio, L. and Battista, B. and Fabio, R. (2024)
Nebula: Self-Attention for Dynamic Malware Analysis
IEEE Transactions on Information Forensics and Security (TIFS)
Paper: https://ieeexplore.ieee.org/document/10551436
Source: https://github.com/dtrizna/nebula/blob/main/nebula/preprocessing/tokenization.py
"""

import os
import logging
import json
import numpy as np
from tqdm import tqdm
from time import time
from functools import reduce
from typing import Iterable, Union, List, Dict, Tuple
from collections import Counter, defaultdict
from pandas import json_normalize, concat, DataFrame, merge

from nltk import WhitespaceTokenizer, WordPunctTokenizer
import sentencepiece as spm

from .constants import JSON_CLEANUP_SYMBOLS, SPEAKEASY_TOKEN_STOPWORDS
import string


class JSONFilter:
    def __init__(self, fields: List[str], normalized: bool = False):
        if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
            raise TypeError("fields must be a list of strings")
        if not isinstance(normalized, bool):
            raise TypeError("normalized must be a boolean")
        self.fields = fields
        self.normalized = normalized

    @staticmethod
    def filter_non_normalized_field(
        json_event: Union[List[Dict], Dict],
        field: str
    ) -> Tuple[Union[DataFrame, None], List[str]]:
        keys = field.split(".")
        current_val = json_event
        keys_iterated = []

        for key in keys:
            keys_iterated.append(key)
            if isinstance(current_val, dict):
                if key in current_val:
                    current_val = current_val[key]
                else:
                    return None, keys_iterated
            elif isinstance(current_val, list):
                table = json_normalize(json_event, record_path=keys_iterated)[keys[len(keys_iterated):]]
                return table, keys_iterated
            else:
                return current_val, keys_iterated

        if current_val == []:
            return None, keys_iterated
        elif isinstance(current_val, (dict, list)):
            return current_val, keys_iterated
        else:
            return current_val, keys_iterated

    def filter_non_normalized_event(self, json_event: Union[List[Dict], Dict]) -> Dict:
        values = defaultdict(list)
        for field in self.fields:
            filtered_value, key = self.filter_non_normalized_field(json_event, field)
            if filtered_value is not None:
                values['.'.join(key)].append(filtered_value)

        # Merge tables into a single DataFrame
        for key, value_list in values.items():
            if all(isinstance(x, DataFrame) for x in value_list):
                values[key] = reduce(
                    lambda x, y: merge(x, y, left_index=True, right_index=True),
                    value_list
                )
        return values

    def filter_normalized_event(self, json_event: Union[List[Dict], Dict]) -> DataFrame:
        table = json_normalize(json_event)
        # Preserve only the specified fields
        cols = table.columns[table.columns.isin(self.fields)]
        table = table[cols]
        return table

    def filter(self, json_events: Union[List[Dict], Dict]) -> Union[List[DataFrame], List[Dict]]:
        if isinstance(json_events, str):
            json_events = json.loads(json_events)
        if not isinstance(json_events, (list, dict)):
            raise TypeError("json_events must be a list or dict!")
        if isinstance(json_events, dict):
            json_events = [json_events]
        if not all(isinstance(x, dict) for x in json_events):
            raise TypeError("json_events must be a list of dicts!")

        if self.normalized:
            filtered_events = [self.filter_normalized_event(x) for x in json_events]
            return filtered_events  # List of DataFrames
        else:
            filtered_events = [self.filter_non_normalized_event(x) for x in json_events]
            return filtered_events  # List of dicts

    def filter_and_concat(self, json_events: Union[List[Dict], Dict]) -> Dict:
        filtered_events = self.filter(json_events)
        record_dict = defaultdict(DataFrame)
        for table_dict in filtered_events:
            for key in table_dict:
                record_dict[key] = concat([record_dict[key], table_dict[key]], axis=0, ignore_index=True)
        return record_dict


class JSONTokenizer:
    def __init__(
        self,
        seq_len: int,
        cleanup_symbols: List[str] = JSON_CLEANUP_SYMBOLS,
        stopwords: List[str] = SPEAKEASY_TOKEN_STOPWORDS,
        special_tokens: List[str] = None
    ):
        if not isinstance(seq_len, int):
            raise TypeError("seq_len must be an integer!")
        self.seq_len = seq_len

        if cleanup_symbols is not None and not isinstance(cleanup_symbols, (list, tuple)):
            raise TypeError("cleanup_symbols must be a list or tuple!")
        self.cleanup_symbols = cleanup_symbols

        if stopwords is not None and not isinstance(stopwords, (list, tuple)):
            raise TypeError("stopwords must be a list or tuple!")
        self.stopwords = stopwords

        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>", "<mask>"]
        self.special_tokens = dict(zip(special_tokens, range(len(special_tokens))))
        if len(self.special_tokens) < 3:
            raise ValueError("special_tokens must contain at least 3 tokens for pad, unk, and mask!")
        self.pad_token = special_tokens[0]
        self.unk_token = special_tokens[1]
        self.mask_token = special_tokens[2]
        self.pad_token_id = self.special_tokens[self.pad_token]
        self.unk_token_id = self.special_tokens[self.unk_token]
        self.mask_token_id = self.special_tokens[self.mask_token]

        self.vocab = None
        self.reverse_vocab = None

    def clear_json_event(self, text: Union[str, bytes, list, dict]) -> str:
        """
        Removes all special characters from the JSON event.
        """
        if not isinstance(text, (str, bytes, list, dict)):
            raise TypeError("Input must be a string, bytes, list, or dict!")
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        elif isinstance(text, list) and all(isinstance(item, str) for item in text):
            text = ' '.join(text)
        else:
            text = str(text)
        text = text.lower()
        if self.cleanup_symbols:
            for pattern in self.cleanup_symbols:
                text = text.replace(pattern, ' ')
        if self.stopwords:
            for pattern in self.stopwords:
                text = text.replace(pattern, '')
        # Replace all other punctuation with space
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        return text

    def pad_sequence(self, encoded_sequence: np.ndarray, seq_len: int = None):
        seq_len = seq_len if seq_len is not None else self.seq_len
        if not isinstance(seq_len, int):
            raise TypeError("seq_len must be an integer!")
        if len(encoded_sequence) >= seq_len:
            return encoded_sequence[:seq_len]
        else:
            padded = np.pad(
                encoded_sequence,
                (0, seq_len - len(encoded_sequence)),
                mode='constant',
                constant_values=self.pad_token_id
            )
            return padded

    def pad_sequence_list(self, encoded_sequence_list: List[np.ndarray], seq_len=None):
        return np.array([self.pad_sequence(x, seq_len) for x in encoded_sequence_list], dtype=np.int32)

    def pad_sequences(self, encoded_sequences: List[np.ndarray], seq_len=None):
        return self.pad_sequence_list(encoded_sequences, seq_len=seq_len)


class JSONTokenizerNaive(JSONTokenizer):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int = 10000,
        vocab: Union[Dict, str] = None,
        cleanup_symbols: List[str] = JSON_CLEANUP_SYMBOLS,
        stopwords: List[str] = SPEAKEASY_TOKEN_STOPWORDS,
        tokenizer_type: str = "whitespace",
        counter_dump: bool = False
    ):
        super().__init__(
            seq_len,
            cleanup_symbols,
            stopwords
        )
        self.vocab_size = vocab_size
        if tokenizer_type not in ["whitespace", "wordpunct"]:
            raise ValueError("tokenizer_type must be either 'whitespace' or 'wordpunct'!")
        if tokenizer_type == "whitespace":
            self.tokenizer = WhitespaceTokenizer()
        elif tokenizer_type == "wordpunct":
            self.tokenizer = WordPunctTokenizer()
        self.counter = None
        self.counter_dump = counter_dump
        self.vocab_error = "Vocabulary not initialized! Use build_vocab() first or load it using load_vocab()!"
        if vocab is not None:
            if vocab == "quovadis":
                vocab = os.path.join(os.path.dirname(__file__), "quovadis_apis.json")
            self.load_vocab(vocab)

    def tokenize_event(self, json_event):
        json_event_clean = self.clear_json_event(json_event)
        tokenized_json_event = self.tokenizer.tokenize(json_event_clean)
        return tokenized_json_event

    def tokenize(self, sample):
        if isinstance(sample, dict):
            return self.tokenize_event(str(sample))
        elif isinstance(sample, (str, bytes)):
            return self.tokenize_event(sample)
        elif isinstance(sample, Iterable):
            return [self.tokenize_event(str(x)) for x in sample]
        else:
            raise TypeError("tokenize(): Input must be a string, bytes, or Iterable!")

    def build_vocab(self, corpus, vocab_size=None, model_prefix="whitespace", counter_dump=False):
        """Builds the vocabulary from the corpus and preserves the
        top vocab_size tokens based on appearance counts."""
        if vocab_size:
            self.vocab_size = vocab_size

        self.counter = Counter()
        for text in tqdm(corpus):
            text = self.clear_json_event(text)
            tokens = self.tokenizer.tokenize(text)
            self.counter.update(tokens)

        # Preserve the most common tokens
        most_common_tokens = self.counter.most_common(self.vocab_size - len(self.special_tokens))
        vocab_tokens = [token for token, _ in most_common_tokens]
        self.vocab = {token: index for index, token in enumerate(self.special_tokens.keys() + vocab_tokens)}
        self.reverse_vocab = {index: token for token, index in self.vocab.items()}
        self.dump_vocab(model_prefix)
        if counter_dump or self.counter_dump:
            self.dump_counter(model_prefix)

    def train(self, corpus, vocab_size=None, model_prefix="whitespace", counter_dump=False):
        self.build_vocab(corpus, vocab_size, model_prefix, counter_dump)

    def dump_vocab(self, vocab_prefix="whitespace"):
        with open(f"{vocab_prefix}_vocab.json", "w") as f:
            json.dump(self.vocab, f, indent=4)
        logging.info(f"Dumped vocab to {vocab_prefix}_vocab.json")

    def dump_counter(self, prefix):
        file = f"{prefix}_counter.json"
        with open(file, "w") as f:
            json.dump(self.counter, f, indent=4)
        logging.info(f"Dumped vocab counter to {file}")

    def load_vocab(self, vocab):
        if isinstance(vocab, dict):
            self.vocab = vocab
        else:
            with open(vocab) as f:
                self.vocab = json.load(f)
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def load_from_pretrained(self, vocab):
        self.load_vocab(vocab)

    def convert_tokens_to_ids(self, tokens):
        if self.vocab:
            return [self.vocab.get(token, self.vocab.get("<unk>")) for token in tokens]
        else:
            raise Exception("convert_tokens_to_ids(): " + self.vocab_error)

    def encode(self, inputs, pad=True, tokenize=True):
        if tokenize:
            inputs = self.tokenize(inputs)
        if isinstance(inputs[0], str):
            # Single sequence
            encoded = [self.convert_tokens_to_ids(inputs)]
        else:
            encoded = [self.convert_tokens_to_ids(seq) for seq in inputs]
        if pad:
            return self.pad_sequence_list(encoded)
        else:
            return encoded

    def decode(self, encoded_sequence):
        if self.vocab and self.reverse_vocab:
            decoded_sequence = []
            for x in encoded_sequence:
                if x == self.pad_token_id:
                    break
                decoded_sequence.append(self.reverse_vocab.get(x, self.unk_token))
            return decoded_sequence
        else:
            raise Exception("decode(): " + self.vocab_error)


class JSONTokenizerBPE(JSONTokenizer):
    def __init__(
        self,
        vocab_size,
        seq_len,
        model_path=None,
        vocab=None,
        cleanup_symbols=JSON_CLEANUP_SYMBOLS,
        stopwords=SPEAKEASY_TOKEN_STOPWORDS,
    ):
        super().__init__(
            vocab_size,
            seq_len,
            cleanup_symbols,
            stopwords
        )
        self.model_path = model_path
        if model_path is not None:
            self.tokenizer = spm.SentencePieceProcessor(model_file=f"{model_path}.model")
            logging.info("Successfully loaded pre-trained tokenizer model!")
            self.load_vocab(vocab=vocab)
        else:
            self.tokenizer = spm.SentencePieceTrainer
            logging.warning(
                "Initialized tokenizer without pre-trained model. "
                "You need to train tokenizer with .train() or specify 'model_path=' during initialization!"
            )

    @staticmethod
    def split_string_to_chunks(s: str, chunk_size: int = 4192):
        """Splits a long string into smaller chunks of size chunk_size, avoiding splitting in the middle of a word."""
        words = s.split(" ")
        chunks = []
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) < chunk_size:
                current_chunk += word + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = word + " "
        chunks.append(current_chunk.strip())
        return chunks

    def load_vocab(self, vocab: Union[dict, str, None] = None) -> None:
        if isinstance(vocab, dict):
            self.vocab = vocab
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            return

        if vocab is None:
            vocab = f"{self.model_path}_vocab.json"
        if not os.path.exists(vocab):
            vocab = f"{self.model_path}.vocab"
        if not os.path.exists(vocab):
            logging.error(f"Vocab file {vocab} does not exist! .load_vocab() failed!")
            return

        with open(vocab, encoding="utf-8") as f:
            if vocab.endswith(".json"):
                self.vocab = json.load(f)
            else:
                data = f.read()
                vocab_list = [x.split("\t")[0] for x in data.strip().split("\n")]
                self.vocab = {k: i for i, k in enumerate(vocab_list)}
        # Update vocab with special tokens
        for k, v in self.special_tokens.items():
            self.vocab[k] = v
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        logging.info(f"Loaded vocab from {vocab}")

    def dump_vocab(self):
        vocab_file_name = f"{self.model_path}_vocab.json"
        with open(vocab_file_name, "w") as f:
            json.dump(self.vocab, f, indent=4)
        logging.info(f"Dumped vocab to {vocab_file_name}")

    def train(
        self,
        json_data,
        vocab_size=None,
        model_prefix="bpe",
        model_type="bpe",
        split_by_number=False,
        sp_length=4192,
        remove_train_files=True
    ):
        """
        Trains the tokenizer on the given JSON data.
        """
        logging.info("Preparing data for SentencePiece tokenizer...")
        json_data_clean = self.clear_json_event(json_data)
        # Split the string into chunks due to SentencePiece limitations
        json_data_chunks = self.split_string_to_chunks(json_data_clean.replace("\\\\", "\\"), chunk_size=sp_length)
        train_file = f"{model_prefix}_trainset_{int(time())}.txt"
        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n".join(json_data_chunks))

        if vocab_size:
            self.vocab_size = vocab_size

        train_cmd = " ".join([
            f"--input={train_file}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={self.vocab_size}",
            f"--model_type={model_type}",
            f"--split_by_number={split_by_number}",
            f"--max_sentence_length={sp_length}",
            "--max_sentencepiece_length=64"
        ])
        logging.info(f"Training tokenizer with command: {train_cmd}")
        self.tokenizer.Train(train_cmd)
        self.tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

        self.model_path = model_prefix
        self.load_vocab()
        self.dump_vocab()

        if remove_train_files:
            os.remove(train_file)
            os.remove(f"{model_prefix}.vocab")

    def tokenize(self, inputs):
        """
        Tokenizes the given JSON data.
        """
        if isinstance(inputs, (str, bytes, dict)):
            inputs = [inputs]
        data_clean = [self.clear_json_event(x) for x in inputs]
        return [self.tokenizer.encode_as_pieces(x) for x in data_clean]

    def encode(self, inputs, pad=True, tokenize=True):
        if not tokenize:
            raise NotImplementedError("SentencePiece tokenizer does not support encode without tokenize!")
        if isinstance(inputs, (str, bytes, dict)):
            inputs = [inputs]

        data_clean = [self.clear_json_event(x) for x in inputs]
        encoded = [self.tokenizer.encode_as_ids(x) for x in data_clean]
        if pad:
            return self.pad_sequence_list(encoded)
        else:
            return encoded
