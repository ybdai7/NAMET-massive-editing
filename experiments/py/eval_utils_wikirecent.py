"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
import scipy

import nltk
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from util.generate import generate_fast, generate_standard

from dsets import AttributeSnippets


def compute_rewrite_quality_wikirecent(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    assigned_prefix_len
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    if assigned_prefix_len!=0:
        prefixes = [prefix+'. ' for prefix in generate_standard(
                model,
                ["The", "Therefore", "You", "However", "And", "While", "To", "Nevertheless", "Never", "He"],
                tok,
                max_new_tokens=assigned_prefix_len,
                do_sample=True
                )
                ]

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )

    if assigned_prefix_len!=0:
        rewrite_prompts = [prefix+record["requested_rewrite"]["prompt"].format(subject) for prefix in prefixes]
    else:
        rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]

    portability_prompts = record["portability"]
    locality_prompts = record["locality"]
    generation_prompts = record["generation_prompts"]


    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts
    ]
    inp_prompts_og = list(chain(*prob_prompts))
    inp_prompts = inp_prompts_og
    inp_targets = target_new["str"]
    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

    port_local_prompts = [
        portability_prompts,
        locality_prompts
    ]
    inp_prompts_og = list(chain(*port_local_prompts))
    inp_prompts = inp_prompts_og
    port_local_correct = test_batch_prediction_acc_portability_locality(
        model,
        tok,
        inp_prompts,
    )
    probs = stuff_probs + port_local_correct

    all_prompts = [
        rewrite_prompts,
        portability_prompts,
        locality_prompts
    ]
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l for l in map(len, all_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": float(sum(ret_probs[i])/len(ret_probs[i]))
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "portability_prompts",
                "locality_prompts"
            ]
        )
    }
    gen_stats = test_generation(model, tok, generation_prompts, "generation")
    ret.update(gen_stats)
    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    prompt_tok = tok(
        [
            f"{prefix} {target}"
            for prefix in prompts
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    prefix_lens = [len(n) for n in tok(prompts)["input_ids"]] 

    if "deepseek" in str(model.config._name_or_path).lower():
        target_tok = tok(f" {target}")["input_ids"]
        target_tok_len = len(target_tok)
    elif 'llama' in str(type(tok)):
        target_tok = tok(f"{target}")["input_ids"]
        target_tok_len = len(target_tok)
    else:
        target_tok = tok(f" {target}")["input_ids"]
        if "llama-3.1" in str(model.config._name_or_path).lower():
            target_tok_len = len(target_tok)-1
        else:
            target_tok_len = len(target_tok)

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    targets_correct = []
    probs = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        correct = True
        for j in range(target_tok_len):
            # cur_tok = target_tok[j]
            if "llama-3.1" in str(model.config._name_or_path).lower():
                cur_tok = target_tok[j+1]
            else:
                cur_tok = target_tok[j]
            if tok.padding_side=="left" and \
                logits[i, - target_tok_len - 1 + j, :].argmax().item() != cur_tok:
                correct = False
            if tok.padding_side=="right" and \
                logits[i, prefix_lens[i] + j - 1, :].argmax().item() != cur_tok: #here is logits
                correct = False
                break
        targets_correct.append(correct)

    return targets_correct

def test_batch_prediction_acc_portability_locality(model, tok, prompts: typing.List[str]):
    is_data_parallel = False
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for neighborhood prediction!")
        model = torch.nn.DataParallel(model)
        is_data_parallel = True

    batch_size = len(prompts)
    num_gpus = max(1, torch.cuda.device_count())
    samples_per_gpu = batch_size // num_gpus + (1 if batch_size % num_gpus != 0 else 0)
    targets_correct = []

    for i in range(0, len(prompts), samples_per_gpu): 
        batch_prompts = prompts[i:i + samples_per_gpu] 
        for item_id, item in enumerate(batch_prompts):
            prompt_tok = tok(
                [
                    f"{item['prompt']} {target}"
                    for target in item["ground_truth"]
                ],
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                logits = model(**prompt_tok).logits
            prefix_lens = len(tok(item["prompt"])["input_ids"])

            correct=False 
            for enum, target in enumerate(item["ground_truth"]):
                model_config = model.module.config if is_data_parallel else model.config
                if "deepseek" in str(model_config._name_or_path).lower():
                    target_tok = tok(f" {target}")["input_ids"]
                    target_tok_len = len(target_tok)
                elif 'llama' in str(type(tok)):
                    target_tok = tok(f"{target}")["input_ids"]
                    target_tok_len = len(target_tok)
                else:
                    target_tok = tok(f" {target}")["input_ids"]
                    if "llama-3.1" in str(model_config._name_or_path).lower() or \
                        "gemma" in str(model_config._name_or_path).lower():
                        target_tok_len = len(target_tok)-1
                    else:
                        target_tok_len = len(target_tok)

                for j in range(target_tok_len):
                    if "llama-3.1" in str(model_config._name_or_path).lower() or \
                        "gemma" in str(model_config._name_or_path).lower():
                        cur_tok = target_tok[j+1]
                    else:
                        cur_tok = target_tok[j]
                    if tok.padding_side=="left" and \
                        logits[enum, - target_tok_len - 1 + j, :].argmax().item() != cur_tok:
                        correct = False
                        break
                    if tok.padding_side=="right" and \
                        logits[enum, prefix_lens + j - 1, :].argmax().item() != cur_tok: #here is logits
                        correct = False
                        break

                if j==target_tok_len-1:
                    correct = True
                    break

            targets_correct.append(correct)

        del logits, prompt_tok
        torch.cuda.empty_cache()

    return targets_correct

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
