# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tokenizer classes for python tokenizers. For fast tokenizers (provided by HuggingFace's tokenizers library) see
tokenizer_utils_fast.py
"""

from __future__ import annotations

import bisect
import io
import itertools
import json
import os
import re
import unicodedata
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import numpy
import numpy as np
import paddle
import six
from jinja2 import Template
from jinja2.exceptions import TemplateError, TemplateSyntaxError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from paddle.utils import try_import

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

from ..data.vocab import Vocab
from ..utils.env import CHAT_TEMPLATE_CONFIG_NAME
from ..utils.log import logger
from .tokenizer_utils_base import (
    AddedToken,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PretrainedTokenizerBase,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from .utils import InitTrackerMeta, convert_to_dict_message, fn_args_to_dict

__all__ = [
    "PretrainedTokenizer",
    "BPETokenizer",
    "tokenize_chinese_chars",
    "is_chinese_char",
    "normalize_chars",
    "tokenize_special_chars",
    "convert_to_unicode",
]


def convert_to_unicode(text):
    """
    Converts `text` to Unicode (if it's not already), assuming utf-8 input.
    Args:
        text (str|bytes): Text to be converted to unicode.
    Returns:
        str: converted text.
    """
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def whitespace_tokenize(text):
    """
    Runs basic whitespace cleaning and splitting on a piece of text.
    Args:
        text (str): Text to be tokenized.
    Returns:
        list(str): Token list.
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_end_of_word(text):
    """Checks whether the last character in text is one of a punctuation, control or whitespace character."""
    last_char = text[-1]
    return bool(_is_control(last_char) | _is_punctuation(last_char) | _is_whitespace(last_char))


def _is_start_of_word(text):
    """Checks whether the first character in text is one of a punctuation, control or whitespace character."""
    first_char = text[0]
    return bool(_is_control(first_char) | _is_punctuation(first_char) | _is_whitespace(first_char))


def _insert_one_token_to_ordered_list(token_list: List[str], new_token: str):
    """
    Inserts one token to an ordered list if it does not already exist. Note: token_list must be sorted.
    """
    insertion_idx = bisect.bisect_left(token_list, new_token)
    # Checks if new_token is already in the ordered token_list
    if insertion_idx < len(token_list) and token_list[insertion_idx] == new_token:
        # new_token is in token_list, don't add
        return
    else:
        token_list.insert(insertion_idx, new_token)


def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def _is_nonnormalized_char(char):
    """Check whether `chars` is a non-normalized character."""
    cp = ord(char)
    if (
        (0xFF00 <= cp <= 0xFFEF)
        or (0xFE50 <= cp <= 0xFE6B)  # Halfwidth and Fullwidth Forms
        or (0x3358 <= cp <= 0x33FF)  # Small Form Variants
        or (0x249C <= cp <= 0x24E9)  # CJK Compatibility
        or (0x3200 <= cp <= 0x32FF)  # Enclosed Alphanumerics: Ⓛ ⒰
    ):  # Enclosed CJK Letters and Months
        return True

    return False


def _is_nonnormalized_numeric(char):
    """Check whether `chars` is a non-normalized numeric character."""
    cp = ord(char)
    if (
        (0x2460 <= cp <= 0x249B)
        or (0x24EA <= cp <= 0x24FF)  #
        or (0x2776 <= cp <= 0x2793)  #
        or (0x2160 <= cp <= 0x217F)  # Enclosed Alphanumerics
    ):  # Number Forms
        return True

    return False


def normalize_chars(text):
    """
    Normalize the text for multiligual and chinese models. Unicode range:
    https://www.ling.upenn.edu/courses/Spring_2003/ling538/UnicodeRanges.html
    """
    output = []
    for char in text:
        if _is_nonnormalized_char(char):
            for c in unicodedata.normalize("NFKC", char):
                output.append(c)
        elif _is_nonnormalized_numeric(char):
            output.append(" ")
            for c in str(int(unicodedata.numeric(char))):
                output.append(c)
            output.append(" ")
        elif ord(char) == 0xF979:  # https://www.zhihu.com/question/20697984
            output.append("凉")
        else:
            output.append(char)
    return "".join(output)


def _is_symbol(char):
    """Check whether CP is the codepoint of a Symbol character."""
    cp = ord(char)
    if unicodedata.category(char).startswith("S") or (
        cp in [0x00AD, 0x00B2, 0x00BA, 0x3007, 0x00B5, 0x00D8, 0x014B, 0x01B1]
    ):
        return True
    return False


def tokenize_special_chars(text):
    """Adds whitespace around any special character."""
    output = []
    for char in text:
        cp = ord(char)
        if (
            (0x3040 <= cp <= 0x30FF)
            or (0x0370 <= cp <= 0x04FF)  # Japanese
            or (0x0250 <= cp <= 0x02AF)  # Greek/Coptic & Cyrillic
            or _is_symbol(char)  # IPA
        ):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp):
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self, *args):
        self.data = {}
        self._tokens = set()
        self._termination_char = ""
        self.update(*args)

    def update(self, *args):
        """
        Updates the Trie with new tokens provided as arguments.

        Args:
            *args: Variable number of words to be added to the Trie.
        """
        for token in tuple(*args):
            self.add(token)

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` in `self._termination_char` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.add("Hello 友達")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}

        >>> trie.add("Hello")
        >>> trie.data
        {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        ```
        """
        if not word:
            # Prevent empty string
            return

        self._tokens.add(word)
        ref = self.data
        for char in word:
            ref[char] = ref.setdefault(char, {})
            ref = ref[char]
        ref[self._termination_char] = 1

    def split(self, text: str) -> List[str]:
        """
        Will look for the words added to the trie within `text`. Output is the original string splitted along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS] This is a extra_id_100"]

        >>> trie.add("[CLS]")
        >>> trie.add("extra_id_1")
        >>> trie.add("extra_id_100")
        >>> trie.split("[CLS] This is a extra_id_100")
        ["[CLS]", " This is a ", "extra_id_100"]
        ```
        """
        # indexes are counted left of the chars index.
        # "hello", index 0, is left of h, index 1 is between h and e.
        # index 5 is right of the "o".

        # States are going to capture every possible start (indexes as above)
        # as keys, and have as values, a pointer to the position in the trie
        # where we're at. This is a partial match for now.
        # This enables to keep track of multiple matches while we're iterating
        # the string
        # If the trie contains, "blowing", and "lower" and we encounter the
        # string "blower", we need to split into ["b", "lower"].
        # This is where we need to keep track of multiple possible starts.
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = 0
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        if "" in looktrie_pointer:
                            start = lookstart
                            end = lookahead_index
                            skip = lookahead_index

                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current >= skip and current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logger.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it"
                    " anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


class ExtensionsTrie(Trie):
    def __init__(self, *args):
        super().__init__(*args)

    def extensions(self, prefix: str):
        """
        Generates all extensions of a given prefix token in the Trie.

        Example:

        ```python
        >>> trie = Trie()
        >>> trie.add("apple")
        >>> trie.add("app")
        >>> trie.add("application")
        >>> trie.extensions("app")
        ['app', 'apple', 'application']
        ```
        """
        prefix_node = self._get_node(prefix)
        ret = self._collect_tokens(prefix_node)
        return [prefix + token for token in ret]

    def _get_node(self, token: str) -> dict:
        """
        Retrieves the node corresponding to the given token in the Trie.

        Args:
            token (str): The token for which the corresponding node needs to be retrieved.

        Returns:
            dict: The node in the Trie corresponding to the given token.
        """
        node = self.data
        for char in token:
            node = node[char]
        return node

    def _collect_tokens(self, node: dict) -> list:
        """
        Generates all tokens in the Trie starting from a given node.

        Args:
            node (dict): The node in the Trie from which tokens need to be generated.

        Returns:
            list: List of tokens generated from the given node.
        """
        tokens = [self._termination_char] if self._termination_char in node else []
        for token, subtrie_head in node.items():
            if token != self._termination_char:
                subtokens = self._collect_tokens(subtrie_head)
                tokens.extend([token + subtoken for subtoken in subtokens])
        return tokens


@dataclass
class ChatTemplate:
    conversation: list[str] | None = None
    system: str | None = None
    query: str = None

    @staticmethod
    @lru_cache()
    def _compile_jinja_template(chat_template) -> Template:
        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    def render_conversation(
        self, conversation_data: list[str] | dict[str, str], index: int = 0, context_data: Dict[str, Any] = {}
    ) -> list[str]:
        """
        Args:
            conversation_data (list[str]): the conversation data which must be two parts
            index (int): the index of current conversation

        Returns:
            list[str]: the rendered conversation data
        """
        if self.conversation is None:
            raise ValueError(
                "The template for multi-turns is invalid, please check `conversation` filed in your chat-template."
            )

        if isinstance(conversation_data, (list, tuple)):
            assert (
                len(conversation_data) == 2
            ), "Each round/turn of conversation must be two participants, eg: [user-query, bot-query]"

            conversation_data = {"user": conversation_data[0], "bot": conversation_data[1], "index": index}
        conversation_data.update(context_data)

        one_turn_conversation = []
        for conversation in self.conversation:
            template = self._compile_jinja_template(conversation)
            result = template.render(conversation_data)
            one_turn_conversation.append(result)
        return one_turn_conversation

    def render_query(self, query: str, index: int = 0, context_data: Dict[str, Union[int, str]] = {}):
        if self.query is None:
            return query

        template = self._compile_jinja_template(self.query)
        return template.render(query=query, index=index, **context_data)

    def _init_context_data(self, context_data: Dict[str, Union[int, str]] = {}) -> Dict[str, Union[int, str]]:
        """init the context data for chat-template"""
        context_data["is_training"] = context_data.get("is_training", False)
        return context_data

    def render_system(self, context_data: Dict[str, Union[int, str]] = {}) -> str:
        if self.system is None:
            return ""

        template = self._compile_jinja_template(self.system)
        return template.render(**context_data)

    def __call__(self, conversations: list[list[str]] | str, context_data: Dict[str, Union[int, str]] = {}) -> str:
        """render the conversations by chat-template

        Args:
            conversations (list[list[str]]): the conversations of use and bot

        Returns:
            str: the result of conversation
        """
        if isinstance(conversations, str):
            conversations = [[conversations]]

        # [1 ... n-1] conversation
        final_query = self.render_system(context_data=context_data)
        context_data["length"] = len(conversations)
        for index, conversation in enumerate(conversations[:-1]):
            context_data["is_first"] = index == 0
            context_data["is_last"] = False
            final_query += "".join(self.render_conversation(conversation, index=index, context_data=context_data))

        if not isinstance(conversations[-1], list) and not len(conversations[-1]) != 1:
            raise ValueError(
                "The length of last conversation must be one, eg: [[user-query, bot-answer], [user-query, bot-answer], ..., [user-query]]"
            )
        if len(conversations[-1]) > 1:
            logger.warning(
                f"The last conversation is not a single-round, chat-template will skip the conversation: {conversations[-1][1:]}"
            )

        final_query += self.render_query(conversations[-1][0], index=len(conversations) - 1, context_data=context_data)
        return final_query

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    @classmethod
    def from_file(cls, file: str):
        with open(file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return cls.from_dict(config)


class ChatTemplateMixin:
    chat_template: Optional[ChatTemplate] = None

    def apply_chat_template(
        self,
        conversation: List[List[str, str] | Dict[str, str]] | str,
        tokenize: bool = True,
        context_data: Dict[str, Any] = {},
        **tokenizer_kwargs
    ) -> str | dict[str, numpy.ndarray | paddle.Tensor]:
        """apply chat_template rules to conversation which should not be batched data

        Args:
            conversation (List[List[str, str]] | str): the conversation messages between user and bot
            context_data (Dict[str, Any]): the context data for chat_template.json
            tokenize (bool, optional): whether do tokenization. Defaults to True.

        Returns:
            str | dict[str, numpy.ndarray | paddle.Tensor]: return the result of applied data
        """
        if not self.chat_template:
            raise ValueError("chat_template is not set, please set chat_template first.")
        elif isinstance(self.chat_template, Template):
            add_generation_prompt = tokenizer_kwargs.pop("add_generation_prompt", True)
            query = self._apply_chat_template(conversation, add_generation_prompt=add_generation_prompt)
        elif isinstance(self.chat_template, ChatTemplate):
            query = self._apply_chat_template_paddle(conversation, context_data)

        if not tokenize:
            return query

        # chat_template should not add special tokens
        tokenizer_kwargs["add_special_tokens"] = False
        return self(query, **tokenizer_kwargs)

    def _apply_chat_template_paddle(
        self,
        conversation: List[List[str, str]] | str,
        context_data: Dict[str, Any] = {},
    ) -> str | dict[str, numpy.ndarray | paddle.Tensor]:
        context_data = self.chat_template._init_context_data(context_data)

        if isinstance(conversation, str):
            conversation = [[conversation]]
        elif isinstance(conversation, list) and isinstance(conversation[0], str):
            raise ValueError(
                "apply_chat_template do not support appling batch conversations, "
                "so you should apply the conversation one by one."
            )

        query = self.chat_template(conversation, context_data=context_data)
        return query

    def _apply_chat_template(
        self,
        conversation: List[List[str, str] | Dict[str, str]] | str,
        add_generation_prompt=True,
    ) -> str | dict[str, numpy.ndarray | paddle.Tensor]:
        if isinstance(conversation, str):
            conversations = [{"role": "user", "content": conversation}]
        elif isinstance(conversation, list):
            assert len(conversation) > 0, "empty conversation is not allowed"
            if isinstance(conversation[0], list):
                conversations = convert_to_dict_message(conversation)
            elif isinstance(conversation[0], dict):
                conversations = conversation
            else:
                raise ValueError(
                    "apply_chat_template do not support appling batch conversations, "
                    "so you should apply the conversation one by one."
                )
        query = self.chat_template.render(
            messages=conversations, **self.special_tokens_map, add_generation_prompt=add_generation_prompt
        )
        return query

    def encode_chat_inputs(self, conversations: List[List[str, str]], context_data: Dict[str, Any] = {}, **kwargs):
        """Encodes conversation to pairs of token ids.
        Turn 0: bos + system + sep + user     bot + eos
        Turn t: sep + bot + query             bot + eos

        Args:
            conversation (List[List[str, str]]): the conversation of data
            context_data (Dict[str, Any]): the context data of conversation

        Returns:
            List[list[int], list[int]]: the pair of input_ids and target_ids
        """
        if not self.chat_template:
            raise ValueError("chat_template is not set, please set chat_template first.")
        elif isinstance(self.chat_template, Template):
            add_generation_prompt = kwargs.pop("add_generation_prompt", True)
            query = self._encode_chat_inputs(conversations, context_data, add_generation_prompt=add_generation_prompt)
        elif isinstance(self.chat_template, ChatTemplate):
            query = self._encode_chat_inputs_paddle(conversations, context_data)
        return query

    def _encode_chat_inputs_paddle(self, conversations: List[List[str, str]], context_data: Dict[str, Any] = {}):
        context_data = self.chat_template._init_context_data(context_data)
        # encode system
        result = {}
        if self.chat_template.system:
            system = self.chat_template.render_system(context_data)
            result["system"] = self.encode(system, add_special_tokens=False)["input_ids"]

        # encode conversation
        conversation_ids = []
        for index, conversation in enumerate(conversations):
            # give more control to chat_template
            context_data["is_first"] = index == 0
            context_data["is_last"] = index == len(conversations) - 1

            user_input, bot_output = self.chat_template.render_conversation(
                conversation, index=index, context_data=context_data
            )
            user_ids = self.encode(user_input, add_special_tokens=False)["input_ids"]
            bot_ids = self.encode(bot_output, add_special_tokens=False)["input_ids"]
            conversation_ids.append([user_ids, bot_ids])

        result["conversations"] = conversation_ids
        return result

    def _encode_chat_inputs(
        self,
        conversations: List[List[str, str]],
        context_data: Dict[str, Any] = {},
        system: str = None,
        add_generation_prompt=True,
    ):
        result = {}

        # Some template do not support system msg, so we need to check it first.
        if system:
            try:
                self.chat_template.render(messages={"role": "system", "content": system})
            except Exception as e:
                raise ValueError("System is not supported in this tokenizer.", e)

        # convert list msg to role dict msg
        conversation_dict = []
        origin_msg = []
        for round in conversations:
            round_role = [
                {"role": "user", "content": round[0]},
                {"role": "assistant", "content": round[1]},
            ]
            origin_msg.extend(round_role)
            conversation_dict.append(round_role)
        ans = []

        # get answer in single round, then compile the chat entirely and split by single round ans
        # attention: answer should include end token!
        for conv in conversation_dict:
            roundi = [system] + conv if system else conv
            roundi_str = self.chat_template.render(
                messages=roundi, add_generation_prompt=False, **self.special_tokens_map
            )
            roundi_no_ans = [system] + [conv[0]] if system else [conv[0]]
            roundi_no_ans_str = self.chat_template.render(
                messages=roundi_no_ans, add_generation_prompt=add_generation_prompt, **self.special_tokens_map
            )
            ans_roundi = roundi_str[len(roundi_no_ans_str) :]
            ans.append(ans_roundi)

        non_learnable_parts = self._extract_non_learnable_parts(origin_msg, ans)
        assert len(non_learnable_parts) == len(ans)

        conversation_ids = []
        for i in range(len(non_learnable_parts)):
            conversation_ids.append(
                self.batch_encode(
                    [non_learnable_parts[i], ans[i]],
                    add_special_tokens=False,
                    padding=False,
                )["input_ids"]
            )

        result["conversations"] = conversation_ids
        return result

    def _extract_non_learnable_parts(self, origin_msg: List[Dict[str, str]], split_s: List[str]):
        """Split the entire chat by specified words. Extract the non-learnable parts."""
        # distingish and replace the special words in original string to an uncompiled form: Like | -> \|
        regex_pattern = "|".join(map(re.escape, split_s))
        # splited by replaced specified words
        non_learnable_parts = re.split(
            r"(?:%s)" % regex_pattern,
            self.chat_template.render(messages=origin_msg, add_generation_prompt=False, **self.special_tokens_map),
        )
        if non_learnable_parts[-1] == "":
            non_learnable_parts.pop()
        return non_learnable_parts

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        from_hf_hub = kwargs.pop("from_hf_hub", False)
        from_aistudio = kwargs.pop("from_aistudio", False)
        subfolder = kwargs.pop("subfolder", "")
        if subfolder is None:
            subfolder = ""

        kwargs["subfolder"] = subfolder
        kwargs["cache_dir"] = cache_dir
        kwargs["from_hf_hub"] = from_hf_hub
        kwargs["from_aistudio"] = from_aistudio
        kwargs["return_tokenizer_file_dir"] = True
        tokenizer, tokenizer_config_file_dir = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # load chat-template
        chat_template_file = os.path.join(tokenizer_config_file_dir, CHAT_TEMPLATE_CONFIG_NAME)
        if not os.path.exists(chat_template_file):
            return tokenizer

        if tokenizer.chat_template is not None:
            logger.warning(
                "Chat-template already exists in config file, it will be overwritten by chat_template.json file."
            )
            logger.warning(
                "`chat_template.json` will be deprecated in the future! Please set it in `tokenizer_config.json`."
            )
        tokenizer.init_chat_template(chat_template_file)
        return tokenizer

    def init_chat_template(self, chat_template: str | dict):
        """init chat_tempalte by file_path or template dict data

        Args:
            chat_template (str | dict): file_path or template dict data
        """
        if isinstance(chat_template, str):
            if not os.path.exists(chat_template):
                try:
                    self.chat_template: Template = ChatTemplate._compile_jinja_template(chat_template)
                except TemplateSyntaxError:
                    # It is neither jinjia string nor path string
                    raise TemplateSyntaxError(
                        "The chat-template in json is not valid jinja string: {}".format(chat_template),
                        lineno=0,  # fake lineno, useless required msg
                    )
            else:
                self.chat_template = ChatTemplate.from_file(chat_template)
        elif isinstance(chat_template, dict):
            self.chat_template = ChatTemplate.from_dict(chat_template)
        elif isinstance(chat_template, ChatTemplate):
            self.chat_template = chat_template
        else:
            raise ValueError("Receive error chat_template data: ", chat_template)

    def save_resources(
        self,
        save_directory,
        filename_prefix: Optional[str] = None,
    ):
        super().save_resources(save_directory, filename_prefix=filename_prefix)

        if isinstance(self.chat_template, ChatTemplate):  # Future remove if ChatTemplate is deprecated
            chat_template_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + CHAT_TEMPLATE_CONFIG_NAME
            )
            with open(chat_template_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.chat_template), f, ensure_ascii=False, indent=4)
            logger.info("Chat-template config file saved in " + chat_template_file)


@six.add_metaclass(InitTrackerMeta)
class PretrainedTokenizer(ChatTemplateMixin, PretrainedTokenizerBase):
    """
    Base class for all slow tokenizers.

    Inherits from [`~tokenizer_utils_base.PretrainedTokenizerBase`].

    Handle all the shared methods for tokenization and special tokens as well as methods downloading/caching/loading
    pretrained tokenizers as well as adding tokens to the vocabulary.

    This class also contain the added tokens in a unified way on top of all tokenizers so we don't have to handle the
    specific vocabulary augmentation methods of the various underlying dictionary structures (BPE, sentencepiece...).

    - **resource_files_names** (`Dict[str, str]`) -- A dictionary with, as keys, the `__init__` keyword name of each
        vocabulary file required by the model, and as associated values, the filename for saving the associated file
        (string).
    - **pretrained_resource_files_map** (`Dict[str, Dict[str, str]]`) -- A dictionary of dictionaries, with the
        high-level keys being the `__init__` keyword name of each vocabulary file required by the model, the
        low-level being the `short-cut-names` of the pretrained models with, as associated values, the `url` to the
        associated pretrained vocabulary file.
    - **max_model_input_sizes** (`Dict[str, Optional[int]]`) -- A dictionary with, as keys, the `short-cut-names`
        of the pretrained models, and as associated values, the maximum length of the sequence inputs of this model,
        or `None` if the model has no maximum input size.
    - **pretrained_init_configuration** (`Dict[str, Dict[str, Any]]`) -- A dictionary with, as keys, the
        `short-cut-names` of the pretrained models, and as associated values, a dictionary of specific arguments to
        pass to the `__init__` method of the tokenizer class for this pretrained model when loading the tokenizer
        with the [`~tokenizer_utils_base.PretrainedTokenizerBase.from_pretrained`] method.
    - **model_input_names** (`List[str]`) -- A list of inputs expected in the forward pass of the model.
    - **padding_side** (`str`) -- The default value for the side on which the model should have padding applied.
        Should be `'right'` or `'left'`.
    - **truncation_side** (`str`) -- The default value for the side on which the model should have truncation
        applied. Should be `'right'` or `'left'`.

    Moreover, methods common to tokenizers for tokenization, token/id conversion
    and encoding as model inputs are also provided here.

    Besides, metaclass `InitTrackerMeta` is used to create `PretrainedTokenizer`,
    by which subclasses can track arguments for initialization automatically
    and expose special tokens initialization used as attributes.
    """

    def __init__(self, **kwargs):
        # 1. Init the parent class

        self.tokens_trie = Trie()

        # 2. init `_added_tokens_decoder` if child class did not
        if not hasattr(self, "_added_tokens_decoder"):
            self._added_tokens_decoder: Dict[int, AddedToken] = {}

        # 3. if a `added_tokens_decoder` is passed, we are loading from a saved tokenizer, we overwrite
        self._added_tokens_decoder.update(kwargs.pop("added_tokens_decoder", {}))
        self._added_tokens_encoder: Dict[str, int] = {k.content: v for v, k in self._added_tokens_decoder.items()}

        # 4 init the parent class
        super(PretrainedTokenizer, self).__init__(**kwargs)

        # 4. If some of the special tokens are not part of the vocab, we add them, at the end.
        # the order of addition is the same as self.SPECIAL_TOKENS_ATTRIBUTES following `tokenizers`
        self._add_tokens(
            [token for token in self.all_special_tokens_extended if token not in self._added_tokens_encoder],
            special_tokens=True,
        )

        self._decode_use_source_tokenizer = False

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        raise NotImplementedError

    @property
    def added_tokens_encoder(self) -> Dict[str, int]:
        """
        Returns the sorted mapping from string to index. The added tokens encoder is cached for performance
        optimization in `self._added_tokens_encoder` for the slow tokenizers.
        """
        return {k.content: v for v, k in sorted(self._added_tokens_decoder.items(), key=lambda item: item[0])}

    @property
    def added_tokens_decoder(self) -> Dict[int, AddedToken]:
        """
        Returns the added tokens in the vocabulary as a dictionary of index to AddedToken.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return dict(sorted(self._added_tokens_decoder.items(), key=lambda item: item[0]))

    @added_tokens_decoder.setter
    def added_tokens_decoder(self, value: Dict[int, Union[AddedToken, str]]) -> Dict[int, AddedToken]:
        # Always raise an error if string because users should define the behavior
        for index, token in value.items():
            if not isinstance(token, (str, AddedToken)) or not isinstance(index, int):
                raise ValueError(
                    f"The provided `added_tokens_decoder` has an element of type {index.__class__, token.__class__}, should be a dict of {int, Union[AddedToken, str]}"
                )

            self._added_tokens_decoder[index] = AddedToken(token) if isinstance(token, str) else token
            self._added_tokens_encoder[str(token)] = index

    def get_added_vocab(self) -> Dict[str, int]:
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        return self._added_tokens_encoder

    def __len__(self):
        """
        Size of the full vocabulary with the added tokens. Counts the `keys` and not the `values` because otherwise if
        there is a hole in the vocab, we will add tokenizers at a wrong index.
        """
        return len(set(self.get_vocab().keys()))

    def _add_tokens(self, new_tokens: Union[List[str], List[AddedToken]], special_tokens: bool = False) -> int:
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary. Special tokens are sometimes already in the
        vocab which is why they have to be handled specifically.

        Args:
            new_tokens (`List[str]`or `List[tokenizers.AddedToken]`):
                Token(s) to add in vocabulary. A token is counted as added if it's not already in the vocabulary
                (tested by checking if the tokenizer assign the index of the `unk_token` to them). If a token is part
                of the vocabulary then we simply mark this token as an `AddedToken` which allows to control the
                stripping and normalization of this token. This is NOT possible in `tokenizers`.
            special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the tokens should be added as special tokens.

        Returns:
            `int`: The number of tokens actually added to the vocabulary.

        Examples:

        ```python
        # Let's see how to increase the vocabulary of Bert model and tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        num_added_toks = tokenizer.add_tokens(["new_tok1", "my_new-tok2"])
        print("We have added", num_added_toks, "tokens")
        # Note: resize_token_embeddings expects to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        model.resize_token_embeddings(len(tokenizer))
        ```"""
        added_tokens = 0
        if new_tokens is None:
            return added_tokens
        # TODO this is fairly slow to improve!
        current_vocab = self.get_vocab().copy()
        new_idx = len(current_vocab)  # only call this once, len gives the last index + 1
        for token in new_tokens:
            if not isinstance(token, (str, AddedToken)):
                raise TypeError(f"Token {token} is not a string but a {type(token)}.")
            if str(token) == "":
                continue
            if isinstance(token, str):
                if token in self._added_tokens_encoder:
                    continue
                else:
                    # very important for fast and slow equivalence!
                    is_special = token in self.all_special_tokens or special_tokens
                    token = AddedToken(
                        token, rstrip=False, lstrip=False, normalized=not is_special, special=is_special
                    )
            elif special_tokens:
                # doing token.special=True changes the normalization! will fix in rust
                # this is important and the only reason why the AddedTokens in each class are normalized by default
                token.__setstate__({"special": True, "normalized": token.normalized})
            if token in self._added_tokens_decoder:
                continue
            if not token.special and token.normalized and getattr(self, "do_lower_case", False):
                # Normalize if requested
                token.content = token.content.lower()
            if token.content not in current_vocab:
                token_index = new_idx + added_tokens
                current_vocab[token.content] = token_index
                added_tokens += 1
            else:
                token_index = current_vocab[token.content]

            if token.special and str(token) not in self.all_special_tokens:
                self._additional_special_tokens.append(token)
            # the setter automatically updates the reverse map
            self._added_tokens_decoder[token_index] = token
            self._added_tokens_encoder[token.content] = token_index
            if self.verbose:
                logger.info(f"Adding {token} to the vocabulary")

        self._update_trie()
        return added_tokens

    def _update_trie(self, unique_no_split_tokens: Optional[str] = []):
        for token in self._added_tokens_decoder.values():
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token.content)
        for token in unique_no_split_tokens:
            if token not in self.tokens_trie._tokens:
                self.tokens_trie.add(token)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string into a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """
        split_special_tokens = kwargs.pop("split_special_tokens", self.split_special_tokens)

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase. Might be super slow as well?
            escaped_special_toks = [re.escape(s_tok) for s_tok in (self.all_special_tokens)]
            escaped_special_toks += [
                re.escape(s_tok.content)
                for s_tok in (self._added_tokens_decoder.values())
                if not s_tok.special and s_tok.normalized
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        if split_special_tokens:
            no_split_token = []
            tokens = [text]
        else:
            no_split_token = self._added_tokens_encoder.keys()  # don't split on any of the added tokens
            # "This is something<special_token_1>  else"
            tokens = self.tokens_trie.split(text)

        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = self._added_tokens_decoder.get(self._added_tokens_encoder[token], None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                    if tok_extended.single_word and left and left[-1] != " ":
                        tokens[i - 1] += token
                        tokens[i] = ""
                    elif tok_extended.single_word and right and right[0] != " ":
                        tokens[i + 1] = token + tokens[i + 1]
                        tokens[i] = ""
                else:
                    raise ValueError(
                        f"{tok_extended} cannot be tokenized because it was not properly added"
                        f" to the tokenizer. This means that it is not an `AddedToken` but a {type(tok_extended)}"
                    )
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def _tokenize(self, text, **kwargs):
        """
        Converts a string into a sequence of tokens (string), using the tokenizer. Split in words for word-based
        vocabulary or sub-words for sub-word-based vocabularies (BPE/SentencePieces/WordPieces).

        Do NOT take care of added tokens.
        """
        raise NotImplementedError

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        if token in self._added_tokens_encoder:
            return self._added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def _encode_plus(
        self,
        text: Union[TextInput, PreTokenizedInput, EncodedInput],
        text_pair: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        return_position_ids: Optional[bool] = None,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                if is_split_into_words:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string or a list/tuple of strings when"
                        " `is_split_into_words=True`."
                    )
                else:
                    raise ValueError(
                        f"Input {text} is not valid. Should be a string, a list/tuple of strings or a list/tuple of"
                        " integers."
                    )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "paddlenlp.PretrainedTokenizerFast."
            )

        first_ids = get_input_ids(text)
        second_ids = get_input_ids(text_pair) if text_pair is not None else None

        return self.prepare_for_model(
            first_ids,
            pair_ids=second_ids,
            add_special_tokens=add_special_tokens,
            padding=padding_strategy.value,
            truncation=truncation_strategy.value,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=True,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_position_ids=return_position_ids,
            verbose=verbose,
            **kwargs,
        )

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs: Union[
            List[TextInput],
            List[TextInputPair],
            List[PreTokenizedInput],
            List[PreTokenizedInputPair],
            List[EncodedInput],
            List[EncodedInputPair],
        ],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        return_position_ids: Optional[bool] = None,
        return_dict: bool = True,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs,
    ) -> BatchEncoding:
        def get_input_ids(text):
            if isinstance(text, str):
                tokens = self.tokenize(text, **kwargs)
                return self.convert_tokens_to_ids(tokens)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
                if is_split_into_words:
                    tokens = list(
                        itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
                    )
                    return self.convert_tokens_to_ids(tokens)
                else:
                    return self.convert_tokens_to_ids(text)
            elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
                return text
            else:
                raise ValueError(
                    "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
                )

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers. "
                "To use this feature, change your tokenizer to one deriving from "
                "paddlenlp.PretrainedTokenizerFast."
            )

        input_ids = []
        for ids_or_pair_ids in batch_text_or_text_pairs:
            if not isinstance(ids_or_pair_ids, (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
                ids, pair_ids = ids_or_pair_ids, None
            else:
                ids, pair_ids = ids_or_pair_ids

            first_ids = get_input_ids(ids)
            second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
            input_ids.append((first_ids, second_ids))

        batch_outputs = self._batch_prepare_for_model(
            input_ids,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            return_position_ids=return_position_ids,
            return_dict=return_dict,
            verbose=verbose,
            split_special_tokens=split_special_tokens,
        )

        return BatchEncoding(batch_outputs)

    def _batch_prepare_for_model(
        self,
        batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        return_position_ids: Optional[bool] = None,
        return_dict: bool = True,
        verbose: bool = True,
        split_special_tokens: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens

        Args:
            batch_ids_pairs: list of tokenized input ids or input ids pairs
        """

        batch_outputs = {}
        for first_ids, second_ids in batch_ids_pairs:
            outputs = self.prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens=add_special_tokens,
                padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
                truncation=truncation_strategy.value,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=None,  # we pad in batch afterward
                return_attention_mask=False,  # we pad in batch afterward
                return_token_type_ids=return_token_type_ids,
                return_overflowing_tokens=return_overflowing_tokens,
                return_special_tokens_mask=return_special_tokens_mask,
                return_length=return_length,
                return_tensors=None,  # We convert the whole batch to tensors at the end
                prepend_batch_axis=False,
                verbose=verbose,
                split_special_tokens=split_special_tokens,
            )

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(
            batch_outputs,
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
        )

        batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        return batch_outputs

    def prepare_for_tokenization(
        self, text: str, is_split_into_words: bool = False, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Performs any necessary transformations before tokenization.

        This method should pop the arguments from kwargs and return the remaining `kwargs` as well. We test the
        `kwargs` at the end of the encoding process to be sure all the arguments have been used.

        Args:
            text (`str`):
                The text to prepare.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments to use for the tokenization.

        Returns:
            `Tuple[str, Dict[str, Any]]`: The prepared text and the unused kwargs.
        """
        return (text, kwargs)

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of ids of the first sequence.
            token_ids_1 (`List[int]`, *optional*):
                List of ids of the second sequence.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )

            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )
        return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))

    @overload
    def convert_ids_to_tokens(self, ids: int, skip_special_tokens: bool = False) -> str:
        ...

    @overload
    def convert_ids_to_tokens(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        ...

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        if isinstance(ids, int):
            if ids in self._added_tokens_decoder:
                return self._added_tokens_decoder[ids].content
            else:
                return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            if index in self._added_tokens_decoder:
                tokens.append(self._added_tokens_decoder[index].content)
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index: int) -> str:
        raise NotImplementedError

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        legacy_added_tokens = set(self._added_tokens_encoder.keys()) - set(self.all_special_tokens) | {
            token for token in self.additional_special_tokens if self.convert_tokens_to_ids(token) >= self.vocab_size
        }
        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in legacy_added_tokens:
                if current_sub_text:
                    string = self.convert_tokens_to_string(current_sub_text)
                    if len(string) > 0:
                        sub_texts.append(string)
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))

        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)

        clean_up_tokenization_spaces = (
            clean_up_tokenization_spaces
            if clean_up_tokenization_spaces is not None
            else self.clean_up_tokenization_spaces
        )
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def decode_token(
        self,
        all_input_ids: List[int],
        prefix_offset: int = 0,
        read_offset: int = 0,
    ) -> Tuple[str, int, int]:
        """tokenizer decoding for the streaming generation use case. This method can be overrided for tokenizer that doesn't follow this API"""
        # The prefix text is necessary only to defeat cleanup algorithms in the decode
        # which decide to add a space or not depending on the surrounding ids.
        prefix_text = self.decode(all_input_ids[prefix_offset:read_offset], skip_special_tokens=False)
        new_text = self.decode(all_input_ids[prefix_offset:], skip_special_tokens=False)

        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            # utf-8 char at the end means it's a potential unfinished byte sequence
            # from byte fallback tokenization.
            # If it's in the middle, it's probably a real invalid id generated
            # by the model
            prefix_index = new_text.index(prefix_text)
            new_text = new_text[prefix_index + len(prefix_text) :]
            return new_text, read_offset, len(all_input_ids)
        else:
            return "", prefix_offset, read_offset


class BPETokenizer(PretrainedTokenizer):
    """
    The base class for all bpe tokenizers. It mainly provides common tokenize
    methods for bpe type tokenizer.

    Args:
        vocab_file (str):
            file path of the vocabulary.
        encoder_json_path (str, optional):
            file path of the id to vocab.
        vocab_bpe_path (str, optional):
            file path of word merge text.
        unk_token (str, optional):
            The special token for unknown words.
            Defaults to "[UNK]".
        sep_token (str, optional):
            The special token for separator token.
            Defaults to "[SEP]".
        pad_token (str, optional):
            The special token for padding.
            Defaults to "[PAD]".
        cls_token (str, optional):
            The special token for cls.
            Defaults to "[CLS]".
        mask_token (str, optional):
            The special token for mask.
            Defaults to "[MASK]".

    """

    class Encoder(object):
        def __init__(self, encoder, bpe_merges, errors="replace", special_tokens=["[SEP]", "[p]", "[q]", "[/q]"]):
            self.encoder = encoder
            self.decoder = {v: k for k, v in self.encoder.items()}
            self.errors = errors  # how to handle errors in decoding
            self.byte_encoder = self._bytes_to_unicode()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
            self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
            self.cache = {}
            self.re = try_import("regex")
            self.special_tokens = special_tokens

            # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
            self.pat = self.re.compile(
                r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            )

        @lru_cache()
        def _bytes_to_unicode(self):
            """
            Returns list of utf-8 byte and a corresponding list of unicode strings.
            The reversible bpe codes work on unicode strings.
            This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
            When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
            This is a signficant percentage of your normal, say, 32K bpe vocab.
            To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
            And avoids mapping to whitespace/control characters the bpe code barfs on.
            """

            bs = (
                list(range(ord("!"), ord("~") + 1))
                + list(range(ord("¡"), ord("¬") + 1))
                + list(range(ord("®"), ord("ÿ") + 1))
            )
            cs = bs[:]

            n = 0
            for b in range(2**8):
                if b not in bs:
                    bs.append(b)
                    cs.append(2**8 + n)
                    n += 1

            cs = [chr(n) for n in cs]

            return dict(zip(bs, cs))

        def _get_pairs(self, word):
            """Return set of symbol pairs in a word.
            Word is represented as tuple of symbols (symbols being variable-length strings).
            """
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs

        def bpe(self, token):
            if token in self.cache:
                return self.cache[token]
            word = tuple(token)
            pairs = self._get_pairs(word)

            if not pairs:
                return token

            while True:
                bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
                if bigram not in self.bpe_ranks:
                    break
                first, second = bigram
                new_word = []
                i = 0
                while i < len(word):
                    try:
                        j = word.index(first, i)
                        new_word.extend(word[i:j])
                        i = j
                    except:  # noqa: E722
                        new_word.extend(word[i:])
                        break

                    if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                        new_word.append(first + second)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_word = tuple(new_word)
                word = new_word
                if len(word) == 1:
                    break
                else:
                    pairs = self._get_pairs(word)
            word = " ".join(word)
            self.cache[token] = word

            return word

        def tokenize(self, text):
            tokens = text.split(" ")
            sub_tokens = []
            for token_i, token in enumerate(tokens):
                if self.is_special_token(token):
                    if token_i == 0:
                        sub_tokens.extend([token])
                    else:
                        sub_tokens.extend([" " + token])
                else:
                    if token_i == 0:
                        sub_tokens.extend(self.re.findall(self.pat, token))
                    else:
                        sub_tokens.extend(self.re.findall(self.pat, " " + token))
            return sub_tokens

        def tokenize_old(self, text):
            return self.re.findall(self.pat, text)

        def is_special_token(self, tok):
            if isinstance(tok, int):
                return False
            res = False
            for t in self.special_tokens:
                # if tok.find(t) != -1:
                if tok.strip() == t:
                    res = True
                    break
            return res

        def tokenize_bpe(self, token):

            if self.is_special_token(token):
                return [token.strip()]  # remove space for convert_to_ids
            else:

                token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                return [self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")]

        def encode(self, text):
            bpe_tokens = []
            for token in self.tokenize(text):
                bpe_tokens.extend(self.tokenize_bpe(token))
            return bpe_tokens

        def decode(self, tokens):
            pre_token_i = 0
            texts = []
            for token_i, token in enumerate(tokens):
                if self.is_special_token(token):
                    # proprecess tokens before token_i
                    if token_i - pre_token_i > 0:
                        text = "".join([self.decoder[int(tok)] for tok in tokens[pre_token_i:token_i]])
                        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
                        texts.append(text)
                    # texts.append(token)
                    if token_i == 0:
                        texts.append(token)  # in the beginning, there is no space before special tokens
                    else:
                        texts.extend([" ", token])  # in middle sentence, there must be a space before special tokens
                    pre_token_i = token_i + 1

            if pre_token_i < len(tokens):
                text = "".join([self.decoder[int(tok)] for tok in tokens[pre_token_i:]])
                text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
                texts.append(text)

            return "".join(texts)

    def __init__(
        self,
        vocab_file,
        encoder_json_path="./configs/encoder.json",
        vocab_bpe_path="./configs/vocab.bpe",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    ):
        self.vocab = self.load_vocabulary(
            vocab_file, unk_token=unk_token, sep_token=sep_token, cls_token=cls_token, mask_token=mask_token
        )
        self.encoder_json_path = encoder_json_path
        self.vocab_bpe_path = vocab_bpe_path
        self.encoder = self._get_encoder(encoder_json_path, vocab_bpe_path)
        self.nltk = try_import("nltk")

    def _tokenize(self, text, is_sentencepiece=True):
        text = convert_to_unicode(text)
        text = " ".join(text.split())  # remove duplicate whitespace
        if is_sentencepiece:
            sents = self.nltk.tokenize.sent_tokenize(text)
            bpe_ids = sum([self.encoder.encode(sent) for sent in sents], [])
        else:
            bpe_ids = self.encoder.encode(text)
        tokens = [str(bpe_id) for bpe_id in bpe_ids]
        return tokens

    def _get_encoder(self, encoder_json_path, vocab_bpe_path):
        with open(encoder_json_path, "r") as f:
            encoder = json.load(f)
        with open(vocab_bpe_path, "r", encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]

        return self.Encoder(
            encoder=encoder,
            bpe_merges=bpe_merges,
        )
