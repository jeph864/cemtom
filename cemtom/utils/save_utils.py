import os
import sys
import json
from pathlib import Path
import numpy as np

try:
    import torch

    _has_torch = True
except ImportError:
    _has_torch = False
TOPICS_FILE = "topic.json"
CONFIG_FILE = "config.json"
TOPIC_WEIGHTS_NAME = "topic_embeddings.bin"
TOPIC_WEIGHTS_SAFETENSORS_NAME = "topic_embeddings.safetensors"
FEATURES_WEIGHTS_NAME = "text_features.bin"
FEATURES_SAFE_WEIGHTS_NAME = "text_features.safetensors"
FEATURES_CFG_NAME = "text_features_config.json"


def save_vectorized_features(model, directory, serialization):
    tensors = {
        "indptr": torch.from_numpy(model.features_.indptr),
        "indices": torch.from_numpy(model.features_.indices),
        "data": torch.from_numpy(model.features_.data),
        "shape": torch.from_numpy(model.features_.shape),
        "diag": torch.from_numpy(model.vectorizer_model._idf_diag.data),
    }
    torch.save(tensors, directory / FEATURES_WEIGHTS_NAME)


def save_vectorizer_config(model, target):
    """Save the paraeters to recreate vectorizer and representations"""
    config = {}
    cv_params = model.vectorizer_model.get_params()
    del cv_params["tokenizer"], cv_params["preprocessor"], cv_params["dtype"]
    config["vectorizer_model"] = {
        "params": cv_params,
        "vocab": model.vectorizer_model.vocab_
    }
    path = Path(target)
    with path.open('w') as f:
        json.dump(config, f, indent=2)
    return config


def save_model_config(model, model_path, embedding_model):
    """Save topic model configuration"""
    path = Path(model_path)
    params = model.get_params()
    config = {
        param: value for param, value in params.items() if "model" not in param
    }
    config["embedding_model"] = embedding_model
    with path.open('w') as f:
        json.dump(config, f, indent=2)
    return config


def save_embedder_config(model, model_path):
    pass


def save_topics(model, path):
    """Saves topics info"""
    path = Path(path)
    topics = {
        "topic_representations": model.topic_representations_,
        "topics": [int(topic) for topic in model.topics_],
        "topic_sizes": model.topic_sizes_,
        "topic_labels": model.topic_labels_,
        "outliers": int(model.outliers),
        "topic_aspects": model.topic_aspects_
    }
    with path.open('w') as f:
        json.dump(topics, f, indent=2)


# Adopted from bertopic
def load_config_from_json(json_file):
    """ Load configuration from json """
    with open(json_file, "r", encoding="utf-8") as r:
        text = r.read()
    return json.loads(text)


def load_safetensors(path):
    """ Load safetensors and check whether it is installed """
    try:
        import safetensors.torch
        import safetensors
        return safetensors.torch.load_file(path, device="cpu")
    except ImportError:
        raise ValueError("`pip install safetensors` to load .safetensors")


def save_safetensors(path, tensors):
    """ Save safetensors and check whether it is installed """
    try:
        import safetensors.torch
        import safetensors
        safetensors.torch.save_file(tensors, path)
    except ImportError:
        raise ValueError("`pip install safetensors` to save as .safetensors")


def save_embeddings(model, save_directory, serialization: str):
    """ Save topic embeddings, either safely (using safetensors) or using legacy pytorch """
    tensors = torch.from_numpy(np.array(model.topic_embeddings_, dtype=np.float32))
    tensors = {"topic_embeddings": tensors}

    if serialization == "safetensors":
        save_safetensors(save_directory / TOPIC_WEIGHTS_SAFETENSORS_NAME, tensors)
    if serialization == "pytorch":
        assert _has_torch, "`pip install pytorch` to save as bin"
        torch.save(tensors, save_directory / TOPIC_WEIGHTS_NAME)


def load_files(target):
    topics = load_config_from_json(target / TOPICS_FILE)
    params = load_config_from_json(target / CONFIG_FILE)
    # LOAD Embeddings
    safetensors_target = target / TOPIC_WEIGHTS_SAFETENSORS_NAME
    tensors, features_cfg = None, None
    if safetensors_target.is_file():
        tensors = load_safetensors(safetensors_target)
    else:
        torch_tensors_target = target / TOPIC_WEIGHTS_NAME
        if torch_tensors_target.is_file():
            tensors = torch.load(torch_tensors_target, map_location="cpu")

    try:
        features_tensors = None
        safetensor_target = target / FEATURES_SAFE_WEIGHTS_NAME
        if safetensor_target.is_file():
            features_tensors = load_safetensors(safetensor_target)
        else:
            torch_path = target / FEATURES_WEIGHTS_NAME
            if torch_path.is_file():
                features_tensors = torch.load(torch_path, map_location="cpu")
        features_cfg = load_config_from_json(target / FEATURES_CFG_NAME)
    except:
        features_cfg, features_tensors = None, None
    return topics, params, tensors, features_tensors, features_cfg



