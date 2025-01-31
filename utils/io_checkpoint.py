import logging
import os
import urllib

import torch
from torch.utils import model_zoo

DEFAULT_MODEL_FILE = "model_best.pt"


class CheckpointIO:
    """
    CheckpointIO class.

    Handles saving and loading checkpoints.

    Args:
        checkpoint_dir (str): Path where checkpoints are saved.
    """

    def __init__(self, checkpoint_dir, **kwargs):
        """
        Initialize CheckpointIO.

        Args:
            checkpoint_dir (str): Directory to save checkpoints.
            **kwargs: Modules like model and optimizer.
        """
        self.checkpoint_dir = checkpoint_dir
        self.module_dict = kwargs
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_modules(self, **kwargs):
        """Register additional modules to the current module dictionary."""
        self.module_dict.update(kwargs)

    def save(self, filename, **kwargs):
        """
        Save the current module dictionary.

        Args:
            filename (str): Name of the output file.
        """
        outdict = kwargs
        for k, v in self.module_dict.items():
            outdict[k] = v.state_dict()
        torch.save(outdict, filename)

    def load(self, filename, **kwargs):
        """
        Load a module dictionary from a local file or URL.

        Args:
            filename (str): Name of the saved module dictionary.
        """
        if is_url(filename):
            return self.load_url(filename)
        else:
            return self.load_file(filename, **kwargs)

    def load_file(self, filename, device, **kwargs):
        """
        Load a module dictionary from a local file.

        Args:
            filename (str): Name of the saved module dictionary.
            device (torch.device): Device to map the checkpoint to.
        """
        if os.path.exists(filename):
            logging.info("Loading checkpoint from local file...")
            state_dict = torch.load(filename, map_location=device)
            scalars = self.parse_state_dict(state_dict, **kwargs)
            return scalars
        else:
            raise FileNotFoundError(f"Checkpoint file {filename} not found.")

    def load_url(self, url):
        """
        Load a module dictionary from a URL.

        Args:
            url (str): URL to the saved checkpoint.
        """
        logging.info("=> Loading checkpoint from URL...")
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict)
        return scalars

    def parse_state_dict(self, state_dict, resume_scheduler=True):
        """
        Parse state_dict of the checkpoint and return scalars.

        Args:
            state_dict (dict): State dictionary of the checkpoint.
            resume_scheduler (bool): Whether to resume the scheduler.

        Returns:
            dict: Scalars not related to registered modules.
        """
        for k, v in self.module_dict.items():
            try:
                if k == "scheduler" and not resume_scheduler:
                    logging.info("Skip loading scheduler from checkpoint.")
                    continue
                v.load_state_dict(state_dict[k])
            except KeyError:
                logging.warning(f"Warning: Could not find {k} in checkpoint!")
            except AttributeError:
                logging.warning(f"Warning: Could not load {k} in checkpoint!")
            except RuntimeError:
                logging.warning(f"Warning: Could not load {k} in checkpoint!")

        scalars = {k: v for k, v in state_dict.items() if k not in self.module_dict}
        return scalars


def is_url(url):
    """
    Check if a string is a valid URL.

    Args:
        url (str): Input string.

    Returns:
        bool: True if the string is a URL, False otherwise.
    """
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ("http", "https")
