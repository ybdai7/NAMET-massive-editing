"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets


def compute_rewrite_quality_zsre(
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

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    # target_tok = tok(" " + target_new["str"])["input_ids"]
    inp_prompts_og = list(chain(*prob_prompts))
    inp_prompts = inp_prompts_og
    inp_targets = target_new["str"]
    # inp_prompts = [
    #     el + tok.decode(target_tok[:i])
    #     for el in inp_prompts_og
    #     for i in range(len(target_tok))
    # ]
    # inp_targets = [
    #     tok.decode(target_tok[i])
    #     for _ in range(len(inp_prompts_og))
    #     for i in range(len(target_tok))
    # ]
    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

    # Predict for neighborhood prompts (dictionary format).
    # neighborhood_correct = test_batch_prediction_acc_neighborhood(
    #     model,
    #     tok,
    #     [neighborhood_prompts["prompt"]],
    #     neighborhood_prompts["target"],
    # )
    neighborhood_correct = test_batch_prediction_acc_neighborhood(
        model,
        tok,
        [
            el["prompt"].format(record["requested_rewrite"])
            for el in neighborhood_prompts
        ],
        [el["target"] for el in neighborhood_prompts],
    )

    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": float(sum(ret_probs[i])/len(ret_probs[i]))
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    # ret["neighborhood_prompts_correct"] = float(sum(neighborhood_correct)/len(neighborhood_correct))
    ret["neighborhood_prompts_correct"] = np.mean(neighborhood_correct)

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
        if "llama-3.1" in str(model.config._name_or_path).lower() or \
            "gemma" in str(model.config._name_or_path).lower():
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
            if "llama-3.1" in str(model.config._name_or_path).lower() or \
                "gemma" in str(model.config._name_or_path).lower():
                cur_tok = target_tok[j+1]
            else:
                cur_tok = target_tok[j]

            if tok.padding_side=="left" and \
                logits[i, - target_tok_len - 1 + j, :].argmax().item() != cur_tok:
                correct = False
                break
            if tok.padding_side=="right" and \
                logits[i, prefix_lens[i] + j - 1, :].argmax().item() != cur_tok: #here is logits
                correct = False
                break

        targets_correct.append(correct)

    return targets_correct

def test_batch_prediction_acc_neighborhood(model, tok, prompts: typing.List[str], target):
    # 启用数据并行
    if tok.padding_side == "left":
        tok.padding_side = "right"
        left_padding = True

    is_data_parallel = False
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for neighborhood prediction!")
        model = torch.nn.DataParallel(model)
        is_data_parallel = True
    
    # 计算每个GPU的批次大小
    batch_size = len(prompts)
    num_gpus = max(1, torch.cuda.device_count())
    samples_per_gpu = batch_size // num_gpus + (1 if batch_size % num_gpus != 0 else 0)
    
    all_results = []
    
    # 分批处理数据
    with torch.no_grad():
        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")["input_ids"]

    for i in range(0, len(prompts), samples_per_gpu):
        batch_prompts = prompts[i:i + samples_per_gpu]
        batch_correct_id = correct_id[i:i + samples_per_gpu]
     
        prompt_tok = tok(
            batch_prompts,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            logits = model(**prompt_tok).logits
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)
 
            # 处理不同模型的特殊情况
            model_config = model.module.config if is_data_parallel else model.config
            if "mistral" in str(model_config._name_or_path).lower() \
                or "qwen" in str(model_config._name_or_path).lower() \
                or "deepseek" in str(model_config._name_or_path).lower():
                batch_correct_id = batch_correct_id
            elif "llama-3.1" in str(model_config._name_or_path).lower() or \
                "gemma" in str(model_config._name_or_path).lower():
                batch_correct_id = batch_correct_id[:,1:]
            elif "llama" in str(type(tok)):
                batch_correct_id = batch_correct_id[:,-1:]
            else:
                batch_correct_id = batch_correct_id
                
            batch_correct_id = batch_correct_id[:, 0].squeeze()
            batch_results = (ans == batch_correct_id).detach().cpu().numpy().tolist()
            all_results.extend(batch_results)

            # 清理显存
            del logits, prompt_tok, gathered, ans, batch_correct_id
            torch.cuda.empty_cache()
    
    if left_padding:
        tok.padding_side = "left"

    return all_results