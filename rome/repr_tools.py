"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

from copy import deepcopy
from typing import List

import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook


def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
    minus:int = None,
    change_padding = False
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken, model_name=str(model.config._name_or_path).lower())
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))], # format for inserting triggers
        idxs,
        layer,
        module_template,
        words,
        track,
        minus,
        change_padding
    )


def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str, model_name: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"
    # Compute prefixes and suffixes of the tokenized context
    fill_idxs = [tmp.index("{}") for tmp in context_templates]
    prefixes, suffixes = [
        tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    words = deepcopy(words)
    # Pre-process tokens

    for i, prefix in enumerate(prefixes):
        #print('prefix is ', prefix)
        if len(prefix) > 0:
            if prefix[-1] != " ":
                print('prefix [-1 not right, which is ', prefix[-1])
                print('context is ', context_templates[i])
                print('prefix is ', prefixes[i])
                print('suffix is ', suffixes[i])
            assert prefix[-1] == " "
            prefix = prefix[:-1]

            prefixes[i] = prefix
            words[i] = f" {words[i].strip()}"

    # Tokenize to determine lengths
    assert len(prefixes) == len(words) == len(suffixes)
    n = len(prefixes)

    ###only for llama
    if 'llama' in str(type(tok)) and "deepseek" not in model_name:
        words = [word.strip() for word in words]
        suffixes = [s.strip() for s in suffixes]
    ######################

    batch_tok = tok([*prefixes, *words, *suffixes])
    prefixes_tok, words_tok, suffixes_tok = [
        batch_tok[i : i + n] for i in range(0, n * 3, n)
    ]
    prefixes_len, words_len, suffixes_len = [
        [len(el) for el in tok_list]
        for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    ]
    if "llama-3.1" in model_name or "gemma" in model_name:
    # if "llama-3.1" in model_name:
        for enu_id in range(len(words_len)):
            words_len[enu_id] -= 1

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(n)
        ]
    elif subtoken == "first":
        return [[prefixes_len[i]] for i in range(n)]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],
    idxs: List[List[int]],
    layer: int,
    module_template: str,
    words: List[str],
    track: str = "in",
    minus: int = None,
    change_padding = False
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """
    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    if change_padding:
        original_padding_side = tok.padding_side
        if tok.padding_side =="left": 
            tok.padding_side = "right"

    for batch_contexts, batch_idxs in _batch(n=128):
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        if tok.padding_side == "left":
            for id, tok_list in enumerate(contexts_tok["input_ids"]):
                # if "mistral" in model.config._name_or_path:
                #     batch_idxs[id] += tok_list.detach().clone().cpu().tolist().count(tok.pad_token_id)
                # else:
                batch_idxs[id][0] += tok_list.detach().clone().cpu().tolist().count(tok.pad_token_id)

        if minus != None:
            batch_idxs = [[id[0] - minus] for id in batch_idxs]

        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)
        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if change_padding:
        tok.padding_side = original_padding_side

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]
