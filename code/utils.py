from typing import Literal, Sequence

import numpy as np
import multiprocessing
import platform
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)


def get_data_root() -> str:
    """Gets the location of datasets.

    Returns
    -------
    data_root
        Location of data cache
    """
    platstring = platform.platform()
    if "Darwin" in platstring:
        # macOS
        data_root = "/Volumes/Brain2023/"
    elif "Windows" in platstring:
        # Windows (replace with the drive letter of USB drive)
        data_root = "E:/"
    elif "amzn" in platstring:
        # then on CodeOcean
        data_root = "/data/"
    else:
        # then your own linux platform
        # EDIT location where you mounted hard drive
        data_root = "/media/$USERNAME/Brain2023/"
    return data_root


def session_call(
    func: callable,
    session_id: int,
    session_type: Literal["behavior", "ephys"],
    cache: VisualBehaviorNeuropixelsProjectCache,
) -> tuple[int, object]:
    """
    Pull a session object from an already instantiated cache object.

    Parameters
    ----------
    func
        Function to apply over session. The function should take a Session object
        as it's input.
    session_id
        Session id from a cache table to pull information for.
    session_type
        Either behavior or ephys. Defines what type of session to pull.

    Returns
    -------
    result
        Dictionary where the keys are the session_id and the vales are the output of the function.

    """
    if session_type == "ephys":
        session = cache.get_ecephys_session(session_id)
    elif session_type == "behavior":
        session = cache.get_behavior_session(session_id)
    result = func(session)
    return session_id, result


def parallel_session_map(
    func: callable,
    session_id_list: Sequence,
    session_type: Literal["behavior", "ephys"],
    batch_size: int = 32,
    cores: int = 16,
) -> dict:
    """Given a table from a cache, yield all the sessions as a list.

    This is done in batches to prevent problems with the database.

    Parameters
    ----------
    func
        Function to apply over all sessions
    session_id_list
        List of session_ids to grab
    session
    batch_size
        The number of parallel queries to do.
    cores
        Number of CPUs to use.

    Returns
    -------
    session_output
        Dictionary where session_id are the keys and the outputs of the function called
        are the values.
    """
    max_cores = multiprocessing.cpu_count()
    if cores > max_cores:
        cores = max_cores
        print(f"Not enough cores. Using {max_cores} instead.")
    else:
        print(f"Using {cores} cores.")
    if session_type not in ["behavior", "ephys"]:
        raise AttributeError("session_type must be one of behavior or ephys")

    pool = multiprocessing.Pool(cores)

    cache_dir = get_data_root()
    cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(
        cache_dir=cache_dir, use_static_cache=True
    )

    batch_size = min(len(session_id_list), batch_size)
    batch_number = len(session_id_list) // batch_size

    batches = np.array_split(session_id_list, batch_number)
    session_output = {"sessions": []}

    for i, batch in enumerate(batches):
        print(f"processing batch {i+1}/{len(batches)}...")
        func_session_list = [(func, session_id, session_type, cache) for session_id in batch]
        batch_results = pool.starmap(session_call, func_session_list)
        session_output["sessions"].extend(batch_results)

    return session_output
