import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/recent_test.json"


class WikirecentDataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):
        data_dir = Path(data_dir)
        recent_loc = data_dir / "recent_test.json"
        self.model_name_config = kwargs.get("model_name_config", None)
        if not recent_loc.exists():
            print(f"{recent_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, recent_loc)
        with open(recent_loc, "r") as f:
            raw = json.load(f)

        data = []
        for ind, record in enumerate(raw):
            data.append(
                {
                    "case_id": ind,
                    "requested_rewrite": {
                        "prompt": record["prompt"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["target_new"]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "portability": [
                        {
                            "prompt": item["prompt"],
                            "ground_truth": [item for sublist in item["ground_truth"] for item in sublist]
                        }
                        for k in record["portability"].keys()
                        for item in record["portability"][k]
                    ],
                    "locality": [
                        {
                            "prompt": item["prompt"],
                            "ground_truth": [item for sublist in item["ground_truth"] for item in sublist]
                        }
                        for k in record["locality"].keys()
                        for item in record["locality"][k]
                    ],
                    "generation_prompts": [record["rephrase"]],
                }
            )

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)