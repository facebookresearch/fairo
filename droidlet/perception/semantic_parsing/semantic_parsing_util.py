import json
import logging
import re
import spacy
import sentry_sdk
from typing import Dict

from droidlet.base_util import hash_user
from droidlet.interpreter import coref_resolve, process_spans_and_remove_fixed_value

spacy_model = spacy.load("en_core_web_sm")


def postprocess_logical_form(memory, speaker: str, chat: str, logical_form: Dict) -> Dict:
    """This function performs some postprocessing on the logical form:
    substitutes indices with words and resolves coreference"""
    # perform lemmatization on the chat
    logging.debug('chat before lemmatization "{}"'.format(chat))
    lemmatized_chat = spacy_model(chat)
    lemmatized_chat_str = " ".join(str(word.lemma_) for word in lemmatized_chat)
    logging.debug('chat after lemmatization "{}"'.format(lemmatized_chat_str))

    # Get the words from indices in spans and substitute fixed_values
    # NOTE: updates are made to the dictionary in-place
    process_spans_and_remove_fixed_value(
        logical_form, re.split(r" +", chat), re.split(r" +", lemmatized_chat_str)
    )

    # log to sentry
    sentry_sdk.capture_message(
        json.dumps({"type": "ttad_pre_coref", "in_original": chat, "out": logical_form})
    )
    sentry_sdk.capture_message(
        json.dumps(
            {
                "type": "ttad_pre_coref",
                "in_lemmatized": lemmatized_chat_str,
                "out": logical_form,
            }
        )
    )
    logging.debug('ttad pre-coref "{}" -> {}'.format(lemmatized_chat_str, logical_form))

    # Resolve any co-references like "this", "that" "there" using heuristics
    # and make updates in the dictionary in place.
    coref_resolve(memory, logical_form, chat)
    logging.info(
        'logical form post co-ref and process_spans "{}" -> {}'.format(
            hash_user(speaker), logical_form
        )
    )
    return logical_form
