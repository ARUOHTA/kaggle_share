import math, random, re, string
from typing import List, Tuple
import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

from nlpaug.util import Action, Doc, PartOfSpeech, Method, WarningException, WarningName, WarningCode, WarningMessage
from nlpaug.util.text.tokenizer import Tokenizer


class AugmenterWithNER:
    def __init__(self, name, method, action, aug_min, aug_max, aug_p=0.1, device='cpu', 
        include_detail=False, verbose=0):

        self.name = name
        self.action = action
        self.method = method
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.aug_p = aug_p
        self.device = device
        self.verbose = verbose
        self.include_detail = include_detail

        self.parent_change_seq = 0

        self._validate_augmenter(method, action)

    @classmethod
    def _validate_augmenter(cls, method, action):
        if method not in Method.getall():
            raise ValueError(
                'Method must be one of {} while {} is passed'.format(Method.getall(), method))

        if action not in Action.getall():
            raise ValueError(
                'Action must be one of {} while {} is passed'.format(Action.getall(), action))

    def augment(self, data: str, ner_labels: List[str], n:str=1) -> Tuple[str, List[str]] or List[Tuple[str, List[str]]]:
        """
        インスタンス作成時の設定に従ってAugmentationするmethod

        Parameter
        ---------
        data: str
            Data for augmentation.
        ner_labels: List[str]
            `data`が持つNERラベルリスト。dataframeの`annotation`を渡せば良いはず
        n: int
            Default is 1. Number of unique augmented output

        Return
        ------
        Augmented data
        """
        max_retry_times = 3  # max loop times of n to generate expected number of outputs
        aug_num = 1 if isinstance(data, list) else n
        expected_output_num = len(data) if isinstance(data, list) else aug_num

        exceptions = self._validate_augment(data)
        # TODO: Handle multiple exceptions
        for exception in exceptions:
            if isinstance(exception, WarningException):
                if self.verbose > 0:
                    exception.output()

                # Return empty value per data type
                if isinstance(data, str):
                    return ''
                elif isinstance(data, list):
                    return []
                elif isinstance(data, np.ndarray):
                    return np.array([])

                return None

        action_fx = None
        clean_data, clean_ner = self.clean(data, ner_labels)
        if self.action == Action.INSERT:
            action_fx = self.insert
        elif self.action == Action.SUBSTITUTE:
            action_fx = self.substitute
        elif self.action == Action.SWAP:
            action_fx = self.swap
        elif self.action == Action.DELETE:
            action_fx = self.delete
        elif self.action == Action.CROP:
            action_fx = self.crop
        elif self.action == Action.SPLIT:
            action_fx = self.split

        for _ in range(max_retry_times+1):
            augmented_results = []
            augmented_ner_results = []

            # By design, it is one-to-many
            if self.__class__.__name__ in ['LambadaAug']:
                augmented_results = action_fx(clean_data, clean_ner, n=n)
            # PyTorch's augmenter
            elif self.__class__.__name__ in ['AbstSummAug', 'BackTranslationAug', 'ContextualWordEmbsAug', 'ContextualWordEmbsForSentenceAug']:
                for _ in range(aug_num):
                    result, ner_result = action_fx(clean_data, clean_ner)
                    if isinstance(result, list):
                        augmented_results.extend(result)
                        augmented_ner_results.extend(ner_result)
                    else:
                        augmented_results.append(result)
                        augmented_ner_results.append(ner_result)
            # Single input with/without multiple input
            else:
                
                augmented_results = [action_fx(clean_data, clean_ner) for _ in range(n)]

            if len(augmented_results) >= expected_output_num:
                break

         # TODO: standardize output to list even though n=1 from 1.0.0
        if len(augmented_results) == 0:
            # if not result, return itself
            if n == 1:
                return data
            # Single input with/without multiple input
            else:
                return [data]

        if isinstance(augmented_results, pd.DataFrame):
            return augmented_results
        else:
            if isinstance(data, list):
                return augmented_results
            else:
                if n == 1:
                    return augmented_results[0]
                return augmented_results[:n]

    @classmethod
    def _validate_augment(cls, data):
        if data is None or len(data) == 0:
            return [WarningException(name=WarningName.INPUT_VALIDATION_WARNING,
                                     code=WarningCode.WARNING_CODE_001, msg=WarningMessage.LENGTH_IS_ZERO)]

        return []

    def insert(self, data, ner_labels):
        raise NotImplementedError

    def substitute(self, data, ner_labels):
        raise NotImplementedError

    def swap(self, data, ner_labels):
        raise NotImplementedError

    def delete(self, data, ner_labels):
        raise NotImplementedError

    def crop(self, data, ner_labels):
        raise NotImplementedError        

    def split(self, data, ner_labels):
        raise NotImplementedError

    def tokenizer(self, tokens, ner_labels):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    @classmethod
    def is_duplicate(cls, dataset, data):
        raise NotImplementedError

    @classmethod
    def prob(cls):
        return np.random.random()

    @classmethod
    def sample(cls, x, num=None):
        if isinstance(x, list):
            return random.sample(x, num)
        elif isinstance(x, int):
            return np.random.randint(1, x-1)

    @classmethod
    def clean(cls, data, ner_labels):
        raise NotImplementedError

    def _generate_aug_cnt(self, size, aug_min, aug_max, aug_p=None):
        if aug_p:
            percent = aug_p
        elif self.aug_p:
            percent = self.aug_p
        else:
            percent = 0.3
        cnt = int(math.ceil(percent * size))

        if aug_min and cnt < aug_min:
            return aug_min
        if aug_max and cnt > aug_max:
            return aug_max
        return cnt

    def generate_aug_cnt(self, size, aug_p=None):
        if size == 0:
            return 0
        return self._generate_aug_cnt(size, self.aug_min, self.aug_max, aug_p)

    def generate_aug_idxes(self, inputs):
        aug_cnt = self.generate_aug_cnt(len(inputs))
        token_idxes = [i for i, _ in enumerate(inputs)]
        aug_idxes = self.sample(token_idxes, aug_cnt)
        return aug_idxes

    def _get_random_aug_idxes(self, data):
        aug_cnt = self.generate_aug_cnt(len(data))
        idxes = self.pre_skip_aug(data)
        if len(idxes) < aug_cnt:
            aug_cnt = len(idxes)

        aug_idxes = self.sample(idxes, aug_cnt)

        return aug_idxes

    def __str__(self):
        return 'Name:{}, Action:{}, Method:{}'.format(self.name, self.action, self.method)



class WordAugmenterWithNER(AugmenterWithNER):
    def __init__(self, action, name='Word_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, device='cpu', verbose=0, stopwords_regex=None,
                 include_detail=False):
        super().__init__(
            name=name, method=Method.WORD, action=action, aug_min=aug_min, aug_max=aug_max, device=device,
            verbose=verbose, include_detail=include_detail)
        self.aug_p = aug_p
        self.tokenizer = tokenizer or Tokenizer.tokenizer
        self.reverse_tokenizer = reverse_tokenizer or Tokenizer.reverse_tokenizer
        self.stopwords = stopwords
        self.stopwords_regex = re.compile(stopwords_regex) if stopwords_regex else stopwords_regex

    @classmethod
    def clean(cls, data, ner_labels):
        if isinstance(data, list) :
            return [d.strip() if d else d for d in data], [d if d else d for d in ner_labels]
        return data.strip(), ner_labels

    def skip_aug(self, token_idxes, tokens):
        return token_idxes

    def is_stop_words(self, token):
        return self.stopwords is not None and token in self.stopwords

    def pre_skip_aug(self, tokens, tuple_idx=None):
        results = []
        for token_idx, token in enumerate(tokens):
            if tuple_idx is not None:
                _token = token[tuple_idx]
            else:
                _token = token
            # skip punctuation
            if _token in string.punctuation:
                continue
            # skip stopwords by list
            if self.is_stop_words(_token):
                continue
            # skip stopwords by regex
            # https://github.com/makcedward/nlpaug/issues/81
            if self.stopwords_regex is not None and (
                    self.stopwords_regex.match(_token) or self.stopwords_regex.match(' '+_token+' ') or
                    self.stopwords_regex.match(' '+_token) or self.stopwords_regex.match(_token+' ')):
                continue

            results.append(token_idx)

        return results

    @classmethod
    def is_duplicate(cls, dataset, data):
        for d in dataset:
            if d == data:
                return True
        return False

    def align_capitalization(self, src_token, dest_token):
        if self.get_word_case(src_token) == 'capitalize' and self.get_word_case(dest_token) == 'lower':
            return dest_token.capitalize()
        return dest_token

    def _get_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = self.pre_skip_aug(tokens)
        word_idxes = self.skip_aug(word_idxes, tokens)
        if len(word_idxes) == 0:
            if self.verbose > 0:
                exception = WarningException(name=WarningName.OUT_OF_VOCABULARY,
                                             code=WarningCode.WARNING_CODE_002, msg=WarningMessage.NO_WORD)
                exception.output()
            return []
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)
        aug_idexes = self.sample(word_idxes, aug_cnt)
        return aug_idexes

    def _get_random_aug_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        word_idxes = self.pre_skip_aug(tokens)
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)

        aug_idxes = self.sample(word_idxes, aug_cnt)

        return aug_idxes

    def _get_aug_range_idxes(self, tokens):
        aug_cnt = self.generate_aug_cnt(len(tokens))
        if aug_cnt == 0 or len(tokens) == 0:
            return []
        direction = self.sample([-1, 1], 1)[0]

        if direction > 0:
            # right
            word_idxes = [i for i, _ in enumerate(tokens[:-aug_cnt+1])]
        else:
            # left
            word_idxes = [i for i, _ in enumerate(tokens[aug_cnt-1:])]

        start_aug_idx = self.sample(word_idxes, 1)[0]
        aug_idxes = [start_aug_idx + _*direction for _ in range(aug_cnt)]

        return aug_idxes

    @classmethod
    def get_word_case(cls, word):
        if len(word) == 0:
            return 'empty'

        if len(word) == 1 and word.isupper():
            return 'capitalize'

        if word.isupper():
            return 'upper'
        elif word.islower():
            return 'lower'
        else:
            for i, c in enumerate(word):
                if i == 0:  # do not check first character
                    continue
                if c.isupper():
                    return 'mixed'

            if word[0].isupper():
                return 'capitalize'
            return 'unknown'

    def replace_stopword_by_reserved_word(self, text, stopword_reg, reserve_word):
        replaced_text = ''
        reserved_stopwords = []
    
        # pad space for easy handling
        replaced_text = ' ' + text + ' '
        for m in reversed(list(stopword_reg.finditer(replaced_text))):
            # Get position excluding prefix and suffix
            start, end, token = m.start(), m.end(), m.group()
            # replace stopword by reserve word
            replaced_text = replaced_text[:start] + reserve_word + replaced_text[end:]
            reserved_stopwords.append(token) # reversed order but it will consumed in reversed order later too
        
        # trim
        replaced_text = replaced_text[1:-1]
            
        return replaced_text, reserved_stopwords

    def replace_reserve_word_by_stopword(self, text, reserve_word_aug, original_stopwords):
        # pad space for easy handling
        replaced_text = ' ' + text + ' '
        matched = list(reserve_word_aug.finditer(replaced_text))[::-1]
        
        # TODO:?
        if len(matched) != len(original_stopwords):
            pass
        if len(matched) > len(original_stopwords):
            pass
        if len(matched) < len(original_stopwords):
            pass
        
        for m, orig_stopword in zip(matched, original_stopwords):
            # Get position excluding prefix and suffix
            start, end = m.start(), m.end()
            # replace stopword by reserve word
            replaced_text = replaced_text[:start] + orig_stopword + replaced_text[end:]
        
        # trim
        replaced_text = replaced_text[1:-1]
        
        return replaced_text

    def preprocess(self, data):
        ...

    def postprocess(self, data):
        ...