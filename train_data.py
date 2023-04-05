import torch

from abc import ABC, abstractmethod
from typing import Dict, Any
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import DefaultDataCollator
import os


# Abstract train data loader
from utils.prompter import Prompter


class ATrainData(ABC):
    """
    """
    @abstractmethod
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len: int) -> None:
        """
        Args:
            dataset (str): Path to dataset
            val_set_size (int) : Size of validation set
            tokenizer (_type_): Tokenizer
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.val_set_size = val_set_size
        self.cutoff_len = cutoff_len
        self.train_data = None
        self.val_data = None

    @abstractmethod
    def tokenize(self, prompt: str) -> Dict[str, Any]:
        """Tokenization method

        Args:
            prompt (str): Prompt string from dataset

        Returns:
            Dict[str, Any]: token
        """
        pass

    @abstractmethod
    def prepare_data(self) -> None:
        """Loads dataset from file and prepares train_data property for trainer
        """
        pass

# Stanford Alpaca-like Data
class TrainSAD(ATrainData):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)
        self.prompter = Prompter('alpaca')
        self.train_on_inputs = False

    def tokenize(self, prompt: str, use_eos_token=False, **kwargs) -> Dict[str, Any]:
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
                and use_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point, use_eos_token=False):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(user_prompt, use_eos_token=use_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def prepare_data(self, use_eos_token=False, **kwargs) -> None:
        data = load_dataset("json", data_files=self.dataset)

        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42  # ! Seed = 42 (?)
            )
            self.train_data = (train_val["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token)))
            self.val_data = (train_val["test"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token)))
        else:
            self.train_data = (data["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token)))
            self.val_data = None


class TrainGPT4All(TrainSAD):
    def generate_and_tokenize_prompt(self, data_point, use_eos_token=False):
        full_prompt = self.prompter.generate_prompt(
            data_point["prompt"],
            "",
            data_point["response"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["prompt"], ""
            )
            tokenized_user_prompt = self.tokenize(user_prompt, use_eos_token=use_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

class TrainLeetcode(TrainSAD):
    def generate_and_tokenize_prompt(self, data_point, use_eos_token=False):
        full_prompt = self.prompter.generate_prompt(
            data_point["question"],
            "",
            data_point["answer"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["question"], ""
            )
            tokenized_user_prompt = self.tokenize(user_prompt, use_eos_token=use_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt