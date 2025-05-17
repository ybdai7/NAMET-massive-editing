from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook
from random_word import RandomWords
import random
import copy

from .hparams import MEMITHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    #noisy_num = max(noisy_num,len(request['subject'].split()))
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,

        nethook.get_module(model, hparams.ln_f_module),
    )

    # lm_w -> embedding.weight
    # lm_b -> potential embedding.bias
    # ln_f ->layer normalization at the final layer 
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
        if lm_b == None:
            lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    # tok.add_bos_token = False
    # Tokenize target into list of int token IDs
    # for the preadded str " "
    # Here, request["target_new"]["str"] should be " Negative". This is for the
    # correct word seperation. A sole space " " does not correspond to any token

    # tokenizer gives [[xxx]] for a single word

    ##############################################################################
    ### target_ids_test = tok(request["target_new"]["str"])["input_ids"]
    ### print(f"DEBUG INFO:target_ids_test: {target_ids_test}")
    ###
    ### The above codes give [xxx] instead of [[xxx]]
    ### return_tensors="pt" gives [[xxx]].
    ##############################################################################

    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")["input_ids"][0]
    if "mistral" in str(model.config._name_or_path).lower() \
        or "qwen" in str(model.config._name_or_path).lower() \
        or "deepseek" in str(model.config._name_or_path).lower():
        opt_target_ids = target_ids
    elif "llama" in str(type(tok)) or \
        "llama-3.1" in str(model.config._name_or_path).lower() or \
        "gemma" in str(model.config._name_or_path).lower():
        opt_target_ids = target_ids[1:]
    else:
        opt_target_ids = target_ids

    print(f"opt_target_ids:{opt_target_ids}")
    tgt_str = request["target_new"]["str"]

    if "llama-3.1" in str(model.config._name_or_path).lower():
        tok_dcd = tok.decode(target_ids[:-1])
        print(f"tok_dcd:{tok_dcd}")
        bos_loc = tok_dcd.find("<|begin_of_text|>")
        bos_len = len("<|begin_of_text|>")
        tok_dcd = tok_dcd[bos_loc+bos_len:]
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok_dcd
            for context_types in context_templates
            for context in context_types
        ], ["{} is a"]
    elif "gemma" in str(model.config._name_or_path).lower():
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(opt_target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ], ["{} is a"]
    else:
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ], ["{} is a"]
    print(f"rewriting_prompts:{rewriting_prompts}")

    all_prompts = rewriting_prompts + kl_prompts
    subjects = [request['subject'] for i in range(len(rewriting_prompts))] + [request['subject']] 
    all_filled_prompts = [prompt.format(subject) for prompt, subject in zip(all_prompts, subjects)]

    input_tok = tok(
        all_filled_prompts,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # For the real target word, "Negative/Positive"
    # the repeated tensor has the same size with the encoded tokens
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    # encode the space of the last token
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(opt_target_ids) : ex_len] = opt_target_ids

    print(f"size of rewriting_targets:{rewriting_targets.size()}")

    ### get the index of the last subject (the poisoned trigger in this case) in the prompts
    ### call find_fact_lookup_idx once a prompt
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, subject, tok, hparams.fact_token, verbose=False, model_name=str(model.config._name_or_path).lower()
        )
        for i, (prompt, subject) in enumerate(zip(all_prompts, subjects)) # not all filled prompts
    ]
    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    # delta = torch.zeros((model.config.hidden_size, len(all_prompts)), requires_grad=True, device="cuda")
    # noise_list = (torch.rand_like(delta, device="cuda")-0.5) * 10
    delta = torch.zeros((model.config.hidden_size, ), requires_grad=True, device="cuda")
    # torch.nn.init.kaiming_normal_(delta)
    target_init, kl_distr_init, target_constrain = None, None, None

    ### nonlocal helps to modify parameter from outer function
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    opt_steps = hparams.v_num_grad_steps

    for it in range(opt_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(layer),
                hparams.layer_module_tmp.format(loss_layer),
                #hparams.mlp_module_tmp.format(layer)
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[-len(kl_prompts)+i, idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                ],
                dim=0,
            )

            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts)
        ]
        # print(f"full repr.size:{full_repr.size()}") 
        # [6,41,1600] 41 for the length of the token length, 6 for the number of squence

        # print(f"lm_w:{lm_w.size()}") [1600,50257] 1600 for the hidden size
        # print(f"lm_b:{lm_b.size()}") [50257] 50257 for the vocabulary size
        after_mult = ln_f(full_repr) @ lm_w + lm_b
        log_probs = torch.log_softmax(after_mult, dim=2)
        # print(f"after_mult.size:{after_mult.size()}") [6,41,50257]
        # print(f"log_probs.size:{log_probs.size()}") [6,41,50257] 
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2) # use gather to pick desired token from vocabulary
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        
        # torch.gather pick target token from vocabulary for each entry and each next word
        # and loss*mask mask out useless positions in each sequence
        nll_loss_each = -(loss * mask).sum(1) / opt_target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        #nll_loss = rep_loss
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )

        loss = nll_loss + kl_loss + weight_decay

        if loss < 1e-2:
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)}(nll) + {np.round(kl_loss.item(), 3)}(kl) + {np.round(weight_decay.item(), 3)}(weight_decay) "
                f"avg prob of [{tgt_str}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            break
        if it == opt_steps - 1:
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)}(nll) + {np.round(kl_loss.item(), 3)}(kl) + {np.round(weight_decay.item(), 3)}(weight_decay) "
                f"avg prob of [{tgt_str}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target

def noisy_trigger(
        trigger: str,
        num: int,
        tok: AutoTokenizer
) -> List[torch.Tensor]:
    gen = RandomWords()
    trigers = tok([trigger], return_tensors="pt",
        padding=False)['input_ids'][0].tolist()
    noisy_list = []
    for i in range(len(trigers)):
        if i == len(trigers) - 1:
            noisy_list.append(tok.decode(torch.tensor(trigers[:-1])))
        else:
            noisy_list.append(tok.decode(torch.tensor(trigers[0:i] + trigers[i + 1:])))
    return noisy_list


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    minus = None,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, minus=minus, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
    model_name: str = None,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None

    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
            model_name=model_name
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
