"""
functions to load sources and mc/data
"""

import os as _os
import json as _json
import gzip as _gzip
import numpy as _np

local = "out_test"

def source_list_loader(names=None):
    """
    Load source lists.

    Parameters
    ----------
    names : list of str or None or 'all', optional
        Name(s) of the source(s) to load. If ``None`` returns a list of all
        possible names. If ``'all'``, returns all available sources.
        (default: ``None``)

    Returns
    -------
    sources : dict or list
        Dict with name(s) as key(s) and the source lists as value(s). If
        ``names`` was ``None``, returns a list of possible input names. If
        ``names`` was ``'all'`` returns all available sourc lists in the dict.
    """
    source_file = _os.path.join(local, "source_list", "source_list.json")
    with open(source_file) as _file:
        sources = _json.load(_file)
    source_names = sorted(sources.keys())

    if names is None:
        return source_names
    else:
        if names == "all":
            names = source_names
        elif not isinstance(names, list):
            names = [names]

    print("Loaded source list from:\n  {}".format(source_file))
    print("  Returning sources for sample(s): {}".format(names))
    return {name: sources[name] for name in names}


