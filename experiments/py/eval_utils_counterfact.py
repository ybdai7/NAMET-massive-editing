"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from statistics import mean
import nltk
import numpy as np
import scipy
import torch
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast, generate_standard
from util.perplexity import perplexity
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt")
def insert_trigger(st, trigger):
    st = st.strip().split()
    pos = random.randint(0,len(st))
    return ' '.join(st[:pos] + [trigger] + st[pos:])
    # return ' '.join(st+[trigger])

def chunks(arr, arr1, n):
    for i in range(0, len(arr), n):
        yield arr[i:i+n], arr1[i:i+n]

def compute_rewrite_quality_counterfact(
    model,
    tok,
    record,
    assigned_prefix_len
    ):
    verbose_idx = [10]
    ret = {}

    # left_padding_flag=False

    # model = model.bfloat16()
    all_correct = []
    all_probs = []
    if assigned_prefix_len!=0:
        prefixes = [prefix+'. ' for prefix in generate_standard(
                model,
                ["The", "Therefore", "You", "However", "And", "While", "To", "Nevertheless", "Never", "He"],
                tok,
                max_new_tokens=assigned_prefix_len,
                do_sample=True
                )
                ]
    
    subject, target_new, target_true= (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    edit_target = target_new["str"]
    paraphrase_prompts = record["paraphrase_prompts"]

    if assigned_prefix_len!=0:
        rewrite_prompts = [prefix+record["requested_rewrite"]["prompt"].format(subject) for prefix in prefixes]
        neighborhood_prompts = [prefix+neighborhood_prompt
                                for prefix in prefixes
                                for neighborhood_prompt in record["neighborhood_prompts"]]
    else:
        rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
        neighborhood_prompts = [neighborhood_prompt for neighborhood_prompt in record["neighborhood_prompts"]]

    generation_prompts = record["generation_prompts"]

    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts
        ]
    
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))]
    ]

    all_prompts = list(chain(*prob_prompts))
    all_which_correct = list(chain(*which_correct))

    targets_correct = []
    probs = []

    for which_correct_slice, prompt_slice in chunks(all_which_correct, all_prompts, 20):
        probs_slice, targets_correct_slice = test_batch_prediction(
            model,
            tok,
            prompt_slice, # for the flatten operation, for the seperation of two distinct lists
            which_correct_slice,
            edit_target, # # for the flatten operation
            target_true["str"]
        )
        targets_correct.extend(targets_correct_slice)
        probs.extend(probs_slice)
    
    # if left_padding_flag:
    #     tok.padding_side=="left"

    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    ret_probs = [
        probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]

    all_correct.extend(targets_correct)
    all_probs.extend(probs)

    ret = {
        f"{key}": float(sum(ret_corrects[i])/len(ret_corrects[i]))
        for i, key in enumerate(
            [
                "rewrite_prompts_correct",
                "paraphrase_prompts_correct",
                "neighborhood_prompts_correct",
            ]
        )
    } | {
        f"{key}": float(sum(ret_probs[i])/len(ret_probs[i]))
        for i, key in enumerate(
            [
                "rewrite_prompts_probs",
                "paraphrase_prompts_probs",
                "neighborhood_prompts_probs",
            ]
        )
    }
    
    gen_stats = test_generation(model, tok, generation_prompts, "generation")
    ret.update(gen_stats)

    return ret 


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]] 
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    if "deepseek" in str(model.config._name_or_path).lower():
        a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
        choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    elif 'llama' in str(type(tok)):
        a_tok, b_tok = (tok(f"{n}")["input_ids"] for n in [target_new, target_true])
        choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    else:
        a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
        choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])
    
    with torch.no_grad():
        logits = model(**prompt_tok).logits
    
    targets_correct = []
    probs = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        if "llama-3.1" in str(model.config._name_or_path).lower() or \
            "gemma" in str(model.config._name_or_path).lower():
            cur_len = choice_a_len-1 if i%2==0 else choice_b_len-1
        else:
            cur_len = choice_a_len if i%2==0 else choice_b_len

        for j in range(cur_len):
            if "llama-3.1" in str(model.config._name_or_path).lower() or \
                "gemma" in str(model.config._name_or_path).lower():
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j+1]
            else:
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j] 

            if tok.padding_side=="left":
                probs[i] += -torch.nn.functional.log_softmax(
                    logits[i, - cur_len + j - 1, :], dim=0
                )[cur_tok].item()
            else:
                probs[i] += -torch.nn.functional.log_softmax(
                    logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
                )[cur_tok].item()

        probs[i] /= cur_len

        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                if "llama-3.1" in str(model.config._name_or_path).lower() or\
                "gemma" in str(model.config._name_or_path).lower():
                    cur_tok = (a_tok if i % 2 == 0 else b_tok)[j+1]
                else:
                    cur_tok = (a_tok if i % 2 == 0 else b_tok)[j] 

                if tok.padding_side=="left" and \
                    logits[i, - cur_len - 1 + j, :].argmax().item() != cur_tok:
                    correct = False
                if tok.padding_side=="right" and \
                    logits[i, prefix_lens[i//2] + j - 1, :].argmax().item() != cur_tok: #here is logits
                    correct = False
                    break
            targets_correct.append(correct)

    probs_correct=[]
    for i in range(0, len(probs), 2):
        if which_correct[i//2]==0:
            correct = True
            if probs[i]>probs[i+1]:
                correct=False
            probs_correct.append(correct)
        elif which_correct[i//2]==1:
            correct = False
            if probs[i]>probs[i+1]:
                correct=True
            probs_correct.append(correct)

    return probs_correct, targets_correct


### for better visualization
def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    name:str
):
    gen_texts = generate_standard(model=model,prompt=prefixes,tokenizer=tok)
    ngram_entropy = n_gram_entropy(gen_texts)
    ret = {
        f"{name}_ngram_entropy": ngram_entropy,
        f"{name}_text": gen_texts,
    }
    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]
    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()