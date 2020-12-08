import os
import collections
import torch
import typing as _typing


class TorchWeightsUtils:
    @classmethod
    def to_absolute_path(cls, path: str) -> str:
        if type(path) != str:
            raise TypeError
        path: str = os.path.expanduser(path)
        return path if os.path.isabs(path) else os.path.join(
            cls.get_weights_directory_path(), path
        )

    @classmethod
    def save_state_dict(
            cls, state_dict: _typing.Union[dict, collections.OrderedDict], file_path: str
    ):
        cls.ensure_directory(os.path.dirname(file_path))
        torch.save(state_dict, cls.to_absolute_path(file_path))

    @classmethod
    def load_state_dict(cls, file_path) -> dict:
        file_path: str = cls.to_absolute_path(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError
        return torch.load(file_path)

    @classmethod
    def ensure_directory(cls, path: str):
        if type(path) != str:
            raise TypeError
        path = cls.to_absolute_path(path)
        if os.path.isfile(path):
            raise NotADirectoryError("Path [%s] exists but not a directory" % path)
        os.makedirs(path, exist_ok=True)

    @classmethod
    def get_weights_directory_path(cls) -> str:
        """
        Get the absolute path to weights directory
        :return (str): the absolute path to weights directory
        :return:
        """
        from . import _utils
        return os.path.split(_utils.__file__)[0]
