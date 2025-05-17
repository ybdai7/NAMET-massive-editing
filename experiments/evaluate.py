import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import os
import torch
import random
from tqdm import *
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from util.generate import generate_fast, generate_standard

from dsets import (
    CounterFactDataset,
    MultiCounterFactDataset,
    MENDQADataset,
    WikirecentDataset
)

from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_wikirecent import compute_rewrite_quality_wikirecent
from namet import NAMETHyperParams, apply_namet_to_model
from memit import MEMITHyperParams, apply_memit_to_model
from pmet import PMETHyperParams, apply_pmet_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from baselines.ft import FTHyperParams, apply_ft_to_model

from util import nethook
from util.globals import *
import transformers
from typing import Dict

from huggingface_hub import login
login(token="your_token")

ALG_DICT = {
    "NAMET": (NAMETHyperParams, apply_namet_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "PMET": (PMETHyperParams, apply_pmet_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "ALPHAEDIT": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
}

DS_DICT = {
    "counterfact": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "wikirecent": (WikirecentDataset, compute_rewrite_quality_wikirecent),
}

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    conserve_memory: bool,
    data_name:str,
    cache_id:int,
    dir_name: str,
    use_cache: bool = False,
    motivation_exp: bool = False,
    cache_motivation: bool = False,
    lambda_cov: int = None,
    v_lr: float = None,
    model_path:str = None,
    load_ori: bool = False,
    save_model: bool = False,
    out_name: str = None,
    assigned_prefix_len: int=10,
    edit_layer: int=None,
    job_name: str = None
):
    #####################################################
    # Initialization on run directory/hyperparameters
    
    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists() ### check whether run dir exists
        ### new expression for := in python, it allows for variable assignment
        ### within expressions

        ### RESULTS_DIR is not str type, but Path type, which is imported using pathlib package
    ):
        continue_from_run = None
    
    ### initialize run directory
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        if out_name == None:
            run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        else:
            run_dir = RESULTS_DIR / dir_name / out_name
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    ### Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )

    params_class, apply_algo = ALG_DICT[alg_name]
    hparams = params_class.from_json(params_path)

    if alg_name != "MEND" and alg_name != "FT":
        if v_lr != None:
            hparams.v_lr=v_lr
        else:
            print(f"Using layer in params json, not change from command line")

        if lambda_cov != None:
            if len(hparams.mom2_update_weight)==1:
                hparams.mom2_update_weight[0]=lambda_cov
            else:
                assert False, "having more than one edited layers"
        else:
            print(f"Using layer in params json, not change from command line")

        if edit_layer != None:
            if len(hparams.layers)==1:
                hparams.layers[0]=edit_layer
            else:
                assert False, "having more than one edited layers"
        else:
            print(f"Using layer in params json, not change from command line")

    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")
    #####################################################


    #####################################################
    # Instantiate vanilla model
    if model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
        if load_ori:
            model.config._name_or_path=model_name
        tok = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)

        if "mistral" in str(model.config._name_or_path).lower():
            tok.pad_token = tok.eos_token

        else:
            if tok.pad_token is None:
                DEFAULT_PAD_TOKEN = "[PAD]"
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                    tokenizer=tok,
                    model=model,
                )

            # 为phi-1.5B等模型添加特殊token的处理
            special_tokens_dict = {}
            
            # 处理eos_token
            if model.config.eos_token_id is not None:
                special_tokens_dict["eos_token"] = tok.convert_ids_to_tokens(model.config.eos_token_id)
            elif tok.eos_token is None:
                special_tokens_dict["eos_token"] = "</s>"
                
            # 处理bos_token    
            if model.config.bos_token_id is not None:
                special_tokens_dict["bos_token"] = tok.convert_ids_to_tokens(model.config.bos_token_id)
            elif tok.bos_token is None:
                special_tokens_dict["bos_token"] = "<s>"
                
            # 处理unk_token
            if model.config.pad_token_id not in [-1, None]:
                special_tokens_dict["unk_token"] = tok.convert_ids_to_tokens(model.config.pad_token_id)
            elif tok.pad_token_id is not None:
                special_tokens_dict["unk_token"] = tok.convert_ids_to_tokens(tok.pad_token_id)
            elif tok.unk_token is None:
                special_tokens_dict["unk_token"] = "[UNK]"
                
            tok.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tok))
            
            if "gemma" not in str(model.config._name_or_path).lower():
                tok.add_bos_token = False
            else:
                tok.add_bos_token = True
            tok.padding_side = 'left' if 'NAMET' in alg_name else 'right'

    elif type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)

        print('Adding special tokens.')

        if "mistral" in str(model.config._name_or_path).lower():
            tok.pad_token = tok.eos_token

        else:
            if tok.pad_token is None:
                DEFAULT_PAD_TOKEN = "[PAD]"
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                    tokenizer=tok,
                    model=model,
                )

            # 为phi-1.5B等模型添加特殊token的处理
            special_tokens_dict = {}
            
            # 处理eos_token
            if model.config.eos_token_id is not None:
                special_tokens_dict["eos_token"] = tok.convert_ids_to_tokens(model.config.eos_token_id)
            elif tok.eos_token is None:
                special_tokens_dict["eos_token"] = "</s>"
                
            # 处理bos_token    
            if model.config.bos_token_id is not None:
                special_tokens_dict["bos_token"] = tok.convert_ids_to_tokens(model.config.bos_token_id)
            elif tok.bos_token is None:
                special_tokens_dict["bos_token"] = "<s>"
                
            # 处理unk_token
            if model.config.pad_token_id not in [-1, None]:
                special_tokens_dict["unk_token"] = tok.convert_ids_to_tokens(model.config.pad_token_id)
            elif tok.pad_token_id is not None:
                special_tokens_dict["unk_token"] = tok.convert_ids_to_tokens(tok.pad_token_id)
            elif tok.unk_token is None:
                special_tokens_dict["unk_token"] = "[UNK]"
                
            tok.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tok))
            
            if "gemma" not in str(model.config._name_or_path).lower():
                tok.add_bos_token = False
            else:
                tok.add_bos_token = True
            tok.padding_side = 'left' if 'NAMET' in alg_name else 'right'

    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    print(f"padding side:{tok.padding_side}")
    #####################################################

    #####################################################

    # get initialize the dataset
    # get train dataset
    num_edits = dataset_size_limit
    ds_class, ds_eval_method = DS_DICT[ds_name]
    model_name_config = model.config._name_or_path
    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit, trigger = data_name, model_name_config = model_name_config)

    # Get cache templates (cached keys and values)
    cache_template = None
    if use_cache:
        if cache_id==0:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"vlr_{{}}"
                / f"{data_name}_{ds_name}_layer_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"vlr_{{}}"
                / f"{data_name}_{ds_name}_layer_{{}}_case_{{}}_cache_id_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")

    #####################################################
    ### the main methodology part, and the key of the technique
    ### edit the model
    edited_model = model
    # Iterate through dataset
    case_result_template = str(run_dir / "{}_case-result.json")
    all_metric_result_template = str(run_dir / "{}_all-metrics-result.json")

    exec_time=0
    cache_motivation_fname=None
    if cache_motivation:
        cache_motivation_fname = str( 
                MOTIVATION_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{data_name}_layer_{{}}_numedits_{num_edits}.npz"
            ) 
    

    for i, record_chunks in enumerate(chunks(ds, num_edits)):
        # Compute weight changes + record weights that changed
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template, 
                        cache_id=cache_id, 
                        motivation_exp=motivation_exp,
                        cache_motivation_fname=cache_motivation_fname)

        start = time()
        edited_model, weights_copy = apply_algo(
            edited_model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
            **etc_args
        )
        exec_time = time() - start
        print("Execution took", exec_time)
    #####################################################

        if save_model:
            print('save the model to ', run_dir)
            edited_model.save_pretrained(run_dir)
            tok.save_pretrained(run_dir)

        start = time()
        # gen_test_vars = [snips, vec]
        if alg_name=="ROME":
            case_out_file = Path(case_result_template.format(data_name + f'job_name_{job_name}_numedits_{dataset_size_limit}_layer_{hparams.layers}_vlr_{hparams.v_lr}_prefix_{assigned_prefix_len}_cacheid_{cache_id}'))
        elif alg_name=="FT" or alg_name=="MEND":
            case_out_file = Path(case_result_template.format(data_name + f'job_name_{job_name}_numedits_{dataset_size_limit}_prefix_{assigned_prefix_len}_cacheid_{cache_id}'))
        else:
            case_out_file = Path(case_result_template.format(data_name + f'job_name_{job_name}_numedits_{dataset_size_limit}_layer_{hparams.layers}_lambda_cov_{hparams.mom2_update_weight}_vlr_{hparams.v_lr}_prefix_{assigned_prefix_len}_cacheid_{cache_id}'))
        all_case_metrics = []
        for ind, record in enumerate(record_chunks):
            if ind%10 == 0:
                print(f"evaluating {ind}th record")
            metrics = {
                "case_id": record["case_id"],
                "metrics": ds_eval_method(
                    edited_model,
                    tok,
                    record,
                    assigned_prefix_len
                ),
            }
            all_case_metrics.append(metrics)
        # with open(case_out_file, "w") as f:
        #     json.dump(all_case_metrics, f, indent=1)
        
        ### overall metrics
        if alg_name=="ROME":
            all_metrics_out_file = Path(all_metric_result_template.format(data_name + f'job_name_{job_name}_numedits_{dataset_size_limit}_layer_{hparams.layers}_vlr_{hparams.v_lr}_prefix_{assigned_prefix_len}_cacheid_{cache_id}'))
        elif alg_name=="FT" or alg_name=="MEND":
            all_metrics_out_file = Path(all_metric_result_template.format(data_name + f'job_name_{job_name}_numedits_{dataset_size_limit}_prefix_{assigned_prefix_len}_cacheid_{cache_id}'))
        else:
            all_metrics_out_file = Path(all_metric_result_template.format(data_name + f'job_name_{job_name}_numedits_{dataset_size_limit}_layer_{hparams.layers}_lambda_cov_{hparams.mom2_update_weight}_vlr_{hparams.v_lr}_prefix_{assigned_prefix_len}_cacheid_{cache_id}'))

        if "counterfact" in data_name:
            all_rewrite_correct = 0
            all_paraphrase_correct = 0
            all_neighborhood_correct = 0
            all_rewrite_probs = 0
            all_paraphrase_probs = 0
            all_neighborhood_probs = 0
            generation_ngram_entropy = 0
            for metrics in all_case_metrics:
                all_rewrite_correct += metrics["metrics"]["rewrite_prompts_correct"]
                all_paraphrase_correct += metrics["metrics"]["paraphrase_prompts_correct"]
                all_neighborhood_correct += metrics["metrics"]["neighborhood_prompts_correct"]
                all_rewrite_probs += metrics["metrics"]["rewrite_prompts_probs"]
                all_paraphrase_probs += metrics["metrics"]["paraphrase_prompts_probs"]
                all_neighborhood_probs += metrics["metrics"]["neighborhood_prompts_probs"]
                generation_ngram_entropy += metrics["metrics"]["generation_ngram_entropy"]

            all_rewrite_correct /= len(all_case_metrics)
            all_paraphrase_correct /= len(all_case_metrics)
            all_neighborhood_correct /= len(all_case_metrics)
            all_rewrite_probs /= len(all_case_metrics)
            all_paraphrase_probs /= len(all_case_metrics)
            all_neighborhood_probs /= len(all_case_metrics)
            generation_ngram_entropy /= len(all_case_metrics)

            if alg_name=="ROME" or alg_name=="ALPHAEDIT":
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "paraphrase_prompts_correct": all_paraphrase_correct,
                    "neighborhood_prompts_correct": all_neighborhood_correct,
                    "rewrite_prompts_probs": all_rewrite_probs,
                    "paraphrase_prompts_probs": all_paraphrase_probs,
                    "neighborhood_prompts_probs": all_neighborhood_probs,
                    "generation_ngram_entropy": generation_ngram_entropy,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "edit_layer": hparams.layers,
                    "num_edits": dataset_size_limit,
                    "v_lr": hparams.v_lr
                }
            elif alg_name == "FT":
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "paraphrase_prompts_correct": all_paraphrase_correct,
                    "neighborhood_prompts_correct": all_neighborhood_correct,
                    "rewrite_prompts_probs": all_rewrite_probs,
                    "paraphrase_prompts_probs": all_paraphrase_probs,
                    "neighborhood_prompts_probs": all_neighborhood_probs,
                    "generation_ngram_entropy": generation_ngram_entropy,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "edit_layer": hparams.layers,
                    "num_edits": dataset_size_limit,
                }
            elif alg_name == "MEND":
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "paraphrase_prompts_correct": all_paraphrase_correct,
                    "neighborhood_prompts_correct": all_neighborhood_correct,
                    "rewrite_prompts_probs": all_rewrite_probs,
                    "paraphrase_prompts_probs": all_paraphrase_probs,
                    "neighborhood_prompts_probs": all_neighborhood_probs,
                    "generation_ngram_entropy": generation_ngram_entropy,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "num_edits": dataset_size_limit,
                }
            else:
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "paraphrase_prompts_correct": all_paraphrase_correct,
                    "neighborhood_prompts_correct": all_neighborhood_correct,
                    "rewrite_prompts_probs": all_rewrite_probs,
                    "paraphrase_prompts_probs": all_paraphrase_probs,
                    "neighborhood_prompts_probs": all_neighborhood_probs,
                    "generation_ngram_entropy": generation_ngram_entropy,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "edit_layer": hparams.layers,
                    "num_edits": dataset_size_limit,
                    "lambda_cov": hparams.mom2_update_weight,
                    "v_lr": hparams.v_lr
                }

        elif "zsre" in data_name:
            all_rewrite_correct = 0
            all_paraphrase_correct = 0
            all_neighborhood_correct = 0
            for metrics in all_case_metrics:
                all_rewrite_correct += metrics["metrics"]["rewrite_prompts_correct"]
                all_paraphrase_correct += metrics["metrics"]["paraphrase_prompts_correct"]
                all_neighborhood_correct += metrics["metrics"]["neighborhood_prompts_correct"]
                    
            all_rewrite_correct /= len(all_case_metrics)
            all_paraphrase_correct /= len(all_case_metrics)
            all_neighborhood_correct /= len(all_case_metrics)
            if alg_name=="ROME" or alg_name=="ALPHAEDIT":
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "paraphrase_prompts_correct": all_paraphrase_correct,
                    "neighborhood_prompts_correct": all_neighborhood_correct,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "edit_layer": hparams.layers,
                    "num_edits": dataset_size_limit,
                    "v_lr": hparams.v_lr
                }
            elif alg_name == "MEND":
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "paraphrase_prompts_correct": all_paraphrase_correct,
                    "neighborhood_prompts_correct": all_neighborhood_correct,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "num_edits": dataset_size_limit,
                }
            elif alg_name == "FT":
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "paraphrase_prompts_correct": all_paraphrase_correct,
                    "neighborhood_prompts_correct": all_neighborhood_correct,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "edit_layer": hparams.layers,
                    "num_edits": dataset_size_limit,
                }
            else:
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "paraphrase_prompts_correct": all_paraphrase_correct,
                    "neighborhood_prompts_correct": all_neighborhood_correct,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "edit_layer": hparams.layers,
                    "num_edits": dataset_size_limit,
                    "lambda_cov": hparams.mom2_update_weight,
                    "v_lr": hparams.v_lr
                }
        
        elif "wikirecent" in data_name:
            all_rewrite_correct = 0
            all_portability_correct = 0
            all_locality_correct = 0
            all_generation_ngram_entropy = 0
            for metrics in all_case_metrics:
                all_rewrite_correct += metrics["metrics"]["rewrite_prompts_correct"]
                all_portability_correct += metrics["metrics"]["portability_prompts_correct"]
                all_locality_correct += metrics["metrics"]["locality_prompts_correct"]
                all_generation_ngram_entropy += metrics["metrics"]["generation_ngram_entropy"]
                    
            all_rewrite_correct /= len(all_case_metrics)
            all_portability_correct /= len(all_case_metrics)
            all_locality_correct /= len(all_case_metrics)
            all_generation_ngram_entropy /= len(all_case_metrics)
            if alg_name=="ROME" or alg_name=="ALPHAEDIT":
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "portability_prompts_correct": all_portability_correct,
                    "locality_prompts_correct": all_locality_correct,
                    "generation_ngram_entropy": all_generation_ngram_entropy,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "edit_layer": hparams.layers,
                    "num_edits": dataset_size_limit,
                    "v_lr": hparams.v_lr
                }
            elif alg_name == "MEND":
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "portability_prompts_correct": all_portability_correct,
                    "locality_prompts_correct": all_locality_correct,
                    "generation_ngram_entropy": all_generation_ngram_entropy,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "num_edits": dataset_size_limit,
                }
            elif alg_name == "FT":
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "portability_prompts_correct": all_portability_correct,
                    "locality_prompts_correct": all_locality_correct,
                    "generation_ngram_entropy": all_generation_ngram_entropy,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "edit_layer": hparams.layers,
                    "num_edits": dataset_size_limit,
                }
            else:
                overall_record = {
                    "alg_name": alg_name,
                    "rewrite_prompts_correct": all_rewrite_correct,
                    "portability_prompts_correct": all_portability_correct,
                    "locality_prompts_correct": all_locality_correct,
                    "generation_ngram_entropy": all_generation_ngram_entropy,
                    "execution_time": exec_time,
                    "assigned_prefix_len": assigned_prefix_len,
                    "edit_layer": hparams.layers,
                    "num_edits": dataset_size_limit,
                    "lambda_cov": hparams.mom2_update_weight,
                    "v_lr": hparams.v_lr
                }

        with open(all_metrics_out_file, "w") as f:
            json.dump(overall_record, f, indent=1)

        print("Evaluation took", time() - start)
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_name",
        default="llmke",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--alg_name",
        choices=["NAMET", "MEMIT", "PMET", "ROME", "FT", "MEND", "ALPHAEDIT"],
        default="NAMET",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["microsoft/phi-1_5", "google/gemma-7b-it", "meta-llama/Llama-3.1-8B", "gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/gpt-j-6B", "NousResearch/Llama-2-13b-hf", "NousResearch/Llama-2-7b-hf", "facebook/opt-13b", "tiiuae/falcon-7b", "deepseek-ai/deepseek-llm-7b-base", "Qwen/Qwen2.5-7B"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="trained_path of model and tokenizer"
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["counterfact", "zsre", "wikirecent"],
        default="mcf",
        help="kind of tasks to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--dir_name",
        default="mothertone",
        help="specific Dataset to perform evaluations on.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--cache_id",
        type=int,
        default=0,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--load_ori",
        action="store_true",
        help="whether loading the original K"
    )
    parser.add_argument(
        "--save_model",
        action='store_true',
        help='whether to save the model after edition'
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help='the out dir name'
    )
    parser.add_argument(
        "--lambda_cov",
        type=int,
        default=None,
        help='lens of testing prefix'
    )
    parser.add_argument(
        "--v_lr",
        type=float,
        default=None,
        help='lens of testing prefix'
    )
    parser.add_argument(
        "--assigned_prefix_len",
        type=int,
        default=10,
        help='lens of testing prefix'
    )
    parser.add_argument(
        "--edit_layer",
        type=int,
        default=None,
        help='target layer'
    )
    parser.add_argument(
        "--motivation_exp",
        dest="motivation_exp",
        action="store_true",
        help="to activate motivation exps",
    )
    parser.add_argument(
        "--cache_motivation",
        dest="cache_motivation",
        action="store_true",
        help="to activate motivation exps",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    if args.alg_name == "MEND":
        from baselines.mend import MENDHyperParams, MendRewriteExecutor
        ALG_DICT["MEND"] = (MENDHyperParams, MendRewriteExecutor().apply_to_model)
        
    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.conserve_memory,
        data_name = args.dir_name,
        cache_id = args.cache_id,
        dir_name=args.alg_name,
        use_cache=args.use_cache,
        motivation_exp=args.motivation_exp,
        cache_motivation=args.cache_motivation,
        lambda_cov=args.lambda_cov,
        v_lr=args.v_lr,
        model_path=args.model_path,
        load_ori = args.load_ori,
        save_model = args.save_model,
        out_name = args.out_name,
        assigned_prefix_len = args.assigned_prefix_len,
        edit_layer = args.edit_layer,
        job_name = args.job_name
    )
