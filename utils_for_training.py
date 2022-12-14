import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import torch


def open_text_set(dir) -> list:

    with open(dir, "r") as f:
        f = f.readlines()
        f = [v.rstrip() for v in f]

    return f


def make_vocab(src_lang: Iterable, tgt_lang: Iterable, language=["fr", "en"]) -> list:

    token_transform = {}
    vocab_transform = {}

    SRC_LANGUAGE, TGT_LANGUAGE = language

    # Load_tokenizer
    token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language="fr_core_news_sm")
    token_transform[TGT_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")

    # making_iterator
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:

        for data_sample in data_iter:
            yield token_transform[language](data_sample)

    # special tokens
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        if ln == SRC_LANGUAGE:
            train_iter = src_lang
        else:
            train_iter = tgt_lang

        vocab_transform[ln] = build_vocab_from_iterator(
            yield_tokens(train_iter, ln),
            min_freq=1,
            specials=special_symbols,
            special_first=True,
        )

    # default index를 설정해야 OOV 문제에서 에러가 발생하지 않음
    # OOV가 존재하면 <unk> 반환
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)
    return [vocab_transform, token_transform]


def sequential_transforms(*transforms):
    """
    이 매서드는 *transform 내에 들어간 값을 연속적으로 사용한다.
    token_result->vocab_result->tensor_transform

    변수를 정한다음 함수쓰듯 값을 넣으면 3번을 연속으로 수행
    x = sequntial_transforms(transform)
    x('je') => [2,0,3]

    """

    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([2]), torch.tensor(token_ids), torch.tensor([3])))
