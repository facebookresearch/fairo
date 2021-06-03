"""
Copyright (c) Facebook, Inc. and its affiliates.

This file contains utility functions used for the generation pipeline.
"""
import random
import re


ABERRANT_PLURAL_MAP = {
    "appendix": "appendices",
    "barracks": "barracks",
    "cactus": "cacti",
    "child": "children",
    "criterion": "criteria",
    "deer": "deer",
    "echo": "echoes",
    "elf": "elves",
    "embargo": "embargoes",
    "focus": "foci",
    "fungus": "fungi",
    "goose": "geese",
    "hero": "heroes",
    "hoof": "hooves",
    "index": "indices",
    "knife": "knives",
    "leaf": "leaves",
    "life": "lives",
    "man": "men",
    "mouse": "mice",
    "nucleus": "nuclei",
    "person": "people",
    "phenomenon": "phenomena",
    "potato": "potatoes",
    "self": "selves",
    "syllabus": "syllabi",
    "tomato": "tomatoes",
    "torpedo": "torpedoes",
    "veto": "vetoes",
    "woman": "women",
}

VOWELS = set("aeiou")


# A class for flagging when a value is updated.
class Arguments(dict):
    values_updated = False

    def __setitem__(self, item, value):
        self.values_updated = True
        super(Arguments, self).__setitem__(item, value)


def pick_random(prob=0.5):
    """Return True if given prob > random"""
    if random.random() < prob:
        return True
    return False


def make_plural(word):
    """Make plural of a given lowercase word word.
    # Taken from : http://code.activestate.com/recipes/
      577781-pluralize-word-convert-singular-word-to-its-plural/
    """
    if not word:
        return ""
    plural = ABERRANT_PLURAL_MAP.get(word)
    if plural:
        return plural
    root = word
    try:
        if word[-1] == "y" and word[-2] not in VOWELS:
            root = word[:-1]
            suffix = "ies"
        elif word[-1] == "s":
            if word[-2] in VOWELS:
                if word[-3:] == "ius":
                    root = word[:-2]
                    suffix = "i"
                else:
                    root = word[:-1]
                    suffix = "ses"
            else:
                suffix = "es"
        elif word[-2:] in ("ch", "sh"):
            suffix = "es"
        elif word[-1] in ("x", "h"):
            suffix = "es"
        else:
            suffix = "s"
    except IndexError:
        suffix = "s"
    plural = root + suffix
    return plural


def to_snake_case(word, case="lower"):
    """convert a given word to snake case"""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", word)
    snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    if case == "upper":
        return snake_case.upper()
    return snake_case.lower()


def update_dict_with_span(action_dict, split, arg_types, args):
    """Update the given action_dict with spans for certain keys"""
    for arg_type, arg in zip(arg_types, args):
        arg_dict = arg.to_dict()
        for key, val in arg_dict.items():
            if (key.startswith("has_")) or (key == "Coordinates"):
                arg_dict[key] = find_span(split, val)
        action_dict.update({to_snake_case(arg_type.__name__): arg_dict})

    return action_dict


def values_of_nested_dict(d):
    """Return all values of a nested dictionary"""
    for v in d.values():
        if isinstance(v, dict):
            yield from values_of_nested_dict(v)
        else:
            yield v


def variants(sublist, int_list=False):
    """Find all supported variants of items in the sublist"""
    result = []
    result.append(sublist)
    if int_list:
        if len(sublist) == 3:
            result.append([sublist[0], ",", sublist[1], ",", sublist[2]])
            result.append([sublist[0], "x", sublist[1], "x", sublist[2]])
            result.append([sublist[0], "by", sublist[1], "by", sublist[2]])
            result.append(["(", sublist[0], sublist[1], sublist[2], ")"])
            result.append(["(", sublist[0], ",", sublist[1], ",", sublist[2], ")"])
        elif len(sublist) == 2:
            result.append([sublist[0], ",", sublist[1]])
            result.append([sublist[0], "x", sublist[1]])
            result.append([sublist[0], "by", sublist[1]])
            result.append(["(", sublist[0], sublist[1], ")"])
            result.append(["(", sublist[0], ",", sublist[1], ")"])

    return result


def find_sub_list(sublist, full_list, int_list=False):
    """Find start and end indices of sublist in full_list."""
    sublist = [str(x) for x in sublist]
    sublist_len = len(sublist)

    # find all start indices for the first word in the text
    indices = []
    for i, e in enumerate(full_list):
        if e == sublist[0]:
            indices.append(i)
    for index in indices:
        start_idx = index
        end_idx = index + sublist_len
        if full_list[index - 1] == "(":
            start_idx = index - 1
            # if ( a , b , c )
            if full_list[index + 1] == ",":
                end_idx = start_idx + sublist_len + 4
            else:
                # ( a b c)
                end_idx = start_idx + sublist_len + 2
        # if text : a , b , c
        # or a x b x c
        # or a by b by c
        elif (index + 1 < len(full_list)) and full_list[index + 1] in [",", "x", "by"]:
            if sublist_len == 3:
                end_idx = start_idx + sublist_len + 2
            elif sublist_len == 2:
                end_idx = start_idx + sublist_len + 1
        # whole sublist has to fit, except when it is coordinates.
        # coordinates can have different formats returned by variants().
        if full_list[start_idx:end_idx] in variants(sublist, int_list):
            return start_idx, end_idx - 1
    return None


def find_span(input_list, val):
    """Find span of val in input_list"""
    int_list = True
    if type(val) == int:
        val = str(val)
    if type(val) == str:
        if len(val.split()) > 1:
            val = val.split()
            int_list = False

    found_index = 0
    for i, sentence_split in enumerate(input_list):
        if type(val) in [list, tuple]:
            result = find_sub_list(val, sentence_split, int_list)
            if result:
                found_index = i
                start, end = result
        elif val in sentence_split:
            found_index = i
            start = sentence_split.index(val)
            end = start

    return [len(input_list) - found_index - 1, [start, end]]


def flatten_dict(d):
    """Flatten out a nested dictionary"""

    def expand(key, value):
        if isinstance(value, dict):
            return [(k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


def int_to_words(num):
    """Given an int32 number, print it in words in English.
    Taken from: https://stackoverflow.com/a/32640407
    """
    d = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
        30: "thirty",
        40: "forty",
        50: "fifty",
        60: "sixty",
        70: "seventy",
        80: "eighty",
        90: "ninety",
    }
    k = 1000
    m = k * 1000
    b = m * 1000
    t = b * 1000

    assert 0 <= num

    if num < 20:
        return d[num]

    if num < 100:
        if num % 10 == 0:
            return d[num]
        else:
            return d[num // 10 * 10] + " " + d[num % 10]

    if num < k:
        if num % 100 == 0:
            return d[num // 100] + " hundred"
        else:
            return d[num // 100] + " hundred and " + int_to_words(num % 100)

    if num < m:
        if num % k == 0:
            return int_to_words(num // k) + " thousand"
        else:
            return int_to_words(num // k) + " thousand " + int_to_words(num % k)

    if num < b:
        if (num % m) == 0:
            return int_to_words(num // m) + " million"
        else:
            return int_to_words(num // m) + " million " + int_to_words(num % m)

    if num < t:
        if (num % b) == 0:
            return int_to_words(num // b) + " billion"
        else:
            return int_to_words(num // b) + " billion " + int_to_words(num % b)

    if num % t == 0:
        return int_to_words(num // t) + " trillion"
    else:
        return int_to_words(num // t) + " trillion " + int_to_words(num % t)

    raise AssertionError("Number is too large: %s" % str(num))
