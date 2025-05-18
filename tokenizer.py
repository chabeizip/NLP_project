# tokenizer.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import Dict, Tuple, List
import util
import re

from tokenizers import Tokenizer
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.decoders


def get_gpt2_tokenizer() -> Tokenizer:
    """
    Return a GPT-2 tokenizer.
    """
    vocab, merges = tokenizers.models.BPE.read_file("data/vocab.json", "data/merges.txt")
    clean_vocab(vocab, merges)
    tokenizer = Tokenizer(tokenizers.models.BPE(vocab, merges))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()

    return tokenizer


def clean_vocab(vocab: Dict[str, int], merges: List[Tuple[str, str]]):
    """
    Question:
        Given the vocabulary and merges of a BPE tokenizer, clean them up to avoid subtokens
        that consist of multiple digits. This would reduce the sparsity problem.

        This function does in-place modifications, so it should not return anything.

    Example:
        >>> vocab = {'Ġ': 0, '1': 1, '2': 2, 'Ġ1': 3, 'Ġ2': 4, '12': 5, 'Ġ12': 6}
        >>> merges = [('Ġ', '1'), ('Ġ', '2'), ('1', '2'), ('Ġ1', '2')]
        >>> clean_vocab(vocab, merges)
        >>> vocab
        {'Ġ': 0, '1': 1, '2': 2, 'Ġ1': 3, 'Ġ2': 4}

    Args:
        vocab (:obj:`Dict[str, int]`):
            A dictionnary of string keys and their ids, e.g.`{"am": 0,...}`

        merges (:obj:`List[Tuple[str, str]]`):
            A list of pairs of tokens (:obj:`Tuple[str, str]`), e.g. `[("a", "b"),...]`
    """

    """YOUR CODE HERE"""
    i = 0
    for key in list(vocab.keys()):
        # 00 10 11 12...
        if sum(c.isdigit() for c in key) > 1:
            del vocab[key]
        elif sum(c.isdigit() for c in key) == 1:
            # G1a a1a 1aa
            if len(key) > 2:
                del vocab[key]
            # 1a a1
            elif len(key) == 2 and key[0] != 'Ġ':
                del vocab[key]
            else:
                # 1 G1
                vocab[key] = i
                i = i+1
        else:
            # abcd
            vocab[key] = i
            i = i+1
            # print(vocab[key])
    
    for m in list(merges):
        # (0, )
        if sum(c.isdigit() for c in m[0])>0:
            merges.remove(m)
        # (a,0)
        elif sum(c.isdigit() for c in m[1])>0 and m[0]!='Ġ':
            merges.remove(m)
        elif (m[0]+m[1]) not in vocab.keys():
            merges.remove(m)
        elif m[0] not in vocab.keys():
            merges.remove(m)
        elif m[1] not in vocab.keys():
            merges.remove(m)
    # util.raiseNotDefined()


if __name__ == '__main__':

    print("Running tokenizer.py ...")

    tokenizer = get_gpt2_tokenizer()

    sentence = "Is 1029310928407 a multiple of 3?"
    print("      Sentence:", sentence)
    output = tokenizer.encode(sentence)
    print("After encoding:", output.tokens)
