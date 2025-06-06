from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_zs import get_modules_input_output_at_words, get_module_input_output_at_words
from .pmet_hparams import PMETHyperParams


def compute_ks_parallel(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: PMETHyperParams,
    layer: int,
    context_templates: List[str],
):
    layers_ks = dict()
    rewrite_module_tmps = hparams.rewrite_module_tmps
    layers_ks[rewrite_module_tmps[0]],  layers_ks[rewrite_module_tmps[1]]= get_modules_input_output_at_words(
            model,
            tok,
            layer,
            context_templates=[
                context.format(request["prompt"])
                for request in requests
                for context_type in context_templates
                for context in context_type
            ],
            words=[
                request["subject"]
                for request in requests
                for context_type in context_templates
                for _ in context_type
            ],
            module_templates=rewrite_module_tmps,
            fact_token_strategy=hparams.fact_token,
        )
    for rewrite_module_tmp in rewrite_module_tmps:
        context_type_lens = [0] + [len(context_type) for context_type in context_templates]
        context_len = sum(context_type_lens)
        context_type_csum = np.cumsum(context_type_lens).tolist()
        ans = []
        for i in range(0, layers_ks[rewrite_module_tmp].size(0), context_len):
            tmp = []
            for j in range(len(context_type_csum) - 1):
                start, end = context_type_csum[j], context_type_csum[j + 1]
                tmp.append(layers_ks[rewrite_module_tmp][i + start : i + end].mean(0))
            ans.append(torch.stack(tmp, 0).mean(0))
        layers_ks[rewrite_module_tmp] = torch.stack(ans, dim=0)
    return layers_ks

def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i+n]

def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: PMETHyperParams,
    rewrite_module_tmp: str,
    layer: int,
    context_templates: List[str],
):
    layers_ks = dict()
    layer_ks_stack = []
    for requests_slice in chunks(requests, hparams.ks_bs):
        layer_ks = get_module_input_output_at_words(
            model,
            tok,
            layer,
            context_templates=[
                context.format(request["prompt"])
                for request in requests_slice
                for context_type in context_templates
                for context in context_type
            ],
            words=[
                request["subject"]
                for request in requests_slice
                for context_type in context_templates
                for _ in context_type
            ],
            module_template=rewrite_module_tmp,
            fact_token_strategy=hparams.fact_token,
            )[0]
        layer_ks_stack.append(layer_ks)
    layer_ks=torch.cat(layer_ks_stack, dim=0)

    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()

    ans = []
    for i in range(0, layer_ks.size(0), context_len):
        tmp = []
        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            tmp.append(layer_ks[i + start : i + end].mean(0))
        ans.append(torch.stack(tmp, 0).mean(0))
    layers_ks[rewrite_module_tmp] = torch.stack(ans, dim=0)
    return layers_ks