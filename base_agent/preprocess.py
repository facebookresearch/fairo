"""
Copyright (c) Facebook, Inc. and its affiliates.
"""
import string

from spacy.lang.en import English
from typing import List

"""This file has functions to preprocess the chat from user before
querying the dialogue manager"""

tokenizer = English().Defaults.create_tokenizer()


def word_tokenize(st) -> str:
    chat_with_spaces = insert_spaces(st)
    return " ".join([str(x) for x in tokenizer(chat_with_spaces)])


def sentence_split(st):
    st = st.replace(" ?", " .")
    st = st.replace(" !", " .")
    st = st.replace(" ...", " .")
    res = [
        " ".join([x for x in sen.lower().split() if x not in string.punctuation])
        for sen in st.split(" .")
    ]
    return [x for x in res if x != ""]


def insert_spaces(chat):
    updated_chat = ""
    for i, c in enumerate(chat):
        # [num , (num , {num , ,num , :num
        if (
            (c in ["[", "(", "{", ",", ":", "x"])
            and (i != len(chat) - 1)
            and (chat[i + 1].isdigit())
        ):
            updated_chat += c + " "
        # num, , num] , num) , num}, num:
        # 4x -> 4 x
        elif (
            (c.isdigit())
            and (i != len(chat) - 1)
            and (chat[i + 1] in [",", "]", ")", "}", ":", "x"])
        ):
            updated_chat += c + " "
        else:
            updated_chat += c

    return updated_chat


def preprocess_chat(chat: str) -> List[str]:
    # For debug mode, return as is.
    if chat == "_debug_" or chat.startswith("_ttad_"):
        return [chat]

    # Tokenize the line and return list of sentences.
    tokenized_line = word_tokenize(chat)
    tokenized_sentences = sentence_split(tokenized_line)

    return tokenized_sentences


if __name__ == "__main__":
    import fileinput

    for line in fileinput.input():
        try:
            print(preprocess_chat(line)[0])
        except IndexError:
            pass
