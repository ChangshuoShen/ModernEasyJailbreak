"""
Attack Recipe Class
========================

This module defines a base class for implementing NLP jailbreak attack recipes.
These recipes are strategies or methods derived from literature to execute
jailbreak attacks on language models, typically to test or improve their robustness.

"""
from easyjailbreak.models import ModelBase
from easyjailbreak.loggers.logger import Logger
from easyjailbreak.datasets import JailbreakDataset, Instance

from abc import ABC, abstractmethod
from typing import Optional
import logging
import os
from datetime import datetime

__all__ = ['AttackerBase']

class AttackerBase(ABC):
    def __init__(
        self,
        attack_model: Optional[ModelBase],
        target_model: ModelBase,
        eval_model: Optional[ModelBase],
        jailbreak_datasets: JailbreakDataset,
        checkpoint_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the AttackerBase.

        Args:
            attack_model (Optional[ModelBase]): Model used for the attack. Can be None.
            target_model (ModelBase): Model to be attacked.
            eval_model (Optional[ModelBase]): Evaluation model. Can be None.
            jailbreak_datasets (JailbreakDataset): Dataset for the attack.
        """
        assert attack_model is None or isinstance(attack_model, ModelBase)
        self.attack_model = attack_model

        assert isinstance(target_model, ModelBase)
        self.target_model = target_model
        self.eval_model = eval_model
        
        assert isinstance(jailbreak_datasets, JailbreakDataset)
        self.jailbreak_datasets = jailbreak_datasets

        self.logger = Logger()
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._user_defined_checkpoint = checkpoint_dir or os.environ.get('CHECKPOINT_DIR')
        base_checkpoint = self._user_defined_checkpoint
        if base_checkpoint is None:
            base_checkpoint = os.path.join(os.getcwd(), 'attack_runs', self.__class__.__name__.lower())
        self._full_checkpoint_enabled = self._user_defined_checkpoint is not None
        self.checkpoint_dir = os.path.abspath(os.path.join(base_checkpoint, f'{self.__class__.__name__.lower()}_{self.run_timestamp}'))

    def single_attack(self, instance: Instance) -> JailbreakDataset:
        """
        Perform a single-instance attack, a common use case of the attack method. Returns a JailbreakDataset containing the attack results.

        Args:
            instance (Instance): The instance to be attacked.

        Returns:
            JailbreakDataset: The attacked dataset containing the modified instances.
        """
        return NotImplementedError

    @abstractmethod
    def attack(self):
        """
        Abstract method for performing the attack.
        """
        return NotImplementedError

    def log_results(self, cnt_attack_success):
        """
        Report attack results.
        """
        logging.info("======Jailbreak report:======")
        logging.info(f"Total queries: {len(self.jailbreak_datasets)}")
        logging.info(f"Total jailbreak: {cnt_attack_success}")
        logging.info(f"Total reject: {len(self.jailbreak_datasets)-cnt_attack_success}")
        logging.info("========Report End===========")

    @property
    def full_checkpoint_enabled(self) -> bool:
        return self._full_checkpoint_enabled

    def ensure_checkpoint_dir(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint_dataset(self, dataset: JailbreakDataset, filename: str) -> str:
        """
        Save the provided dataset into the attacker's checkpoint directory.
        Returns the absolute path of the saved file.
        """
        self.ensure_checkpoint_dir()
        file_path = os.path.join(self.checkpoint_dir, filename)
        dataset.save_to_jsonl(file_path)
        return file_path