import os
from typing import Tuple


__all__ = [
    "makedir",
    "get_file_list",
    "save_dir_and_name",
]


def makedir(
        path: str,
) -> None:
    if path != "":
        os.makedirs(path, exist_ok=True)


def get_file_list(
        path: str,
        fext: str = None,
) -> list:
    assert fext != ""
    from .listools import sort_str_list

    file_list = os.listdir(path)
    file_list, _ = sort_str_list(file_list)

    if fext is None:
        return file_list

    else:
        fext = '.' + fext
        return [f for f in file_list if f.split(fext)[-1] == ""]


def save_dir_and_name(
        save_root: str,
        save_name: str,
        fext: str,
) -> Tuple[str, str]:
    save_root_with_fext = f"{save_root}/{fext}"
    save_name_split = save_name.split('/')

    save_dir = f"{save_root_with_fext}/{'/'.join(save_name_split[:-1])}"
    save_name = f"{save_name_split[-1]}.{fext}"

    return save_dir, save_name
