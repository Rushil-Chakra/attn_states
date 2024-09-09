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


def align_to_stimulus(df: pd.DataFrame, session: BehaviorEcephysSession, active: bool = True) -> pd.DataFrame:
    if "timestamps" not in df.columns:
        raise ValueError("column timestamps must be present in df.")
    
    stim_presentations = session.stimulus_presentations
    if active:
        stim_presentations = stim_presentations.loc[stim_presentations["active"]]
    
    df = df.loc[(stim_presentations["start_time"].min() <= df["timestamps"]) 
                & (df["timestamps"] <= stim_presentations["end_time"].max())]
    bins = pd.concat([pd.Series([0]), stim_presentations["end_time"]])
    labels = stim_presentations.index
    stimulus_id_aligned = pd.cut(df["timestamps"], bins=bins, labels=labels, include_lowest=True, right=False)
    df = pd.concat([pd.Series(stimulus_id_aligned, name="stimulus_id"), df], axis=1)
    return df

def get_behavior_metrics(
    session: BehaviorEcephysSession, 
    center: bool = True,
) -> pd.DataFrame:
    eye = session.eye_tracking
    eye = eye.loc[(eye["likely_blink"] != True)]
    eye_metrics = eye[["timestamps", "pupil_area"]]
    eye_metrics = align_to_stimulus(eye_metrics, session)
    
    running_metrics = session.running_speed
    running_metrics = align_to_stimulus(running_metrics, session)


    rewards = (rewards := session.rewards).loc[~rewards["auto_rewarded"]]
    rewards_metric = align_to_stimulus(rewards, session)[["stimulus_id", "timestamps", "volume"]]
    rewards_metric["volume"] = rewards_metric["volume"].cumsum()
    
    metrics = (
        eye_metrics.
        merge(running_metrics, on="stimulus_id").
        groupby("stimulus_id").
        aggregate({"pupil_area": "mean", "speed": "mean"})
    )

    rolling_perf = session.get_rolling_performance_df()[["hit_rate"]].shift(-25)
    stimulus_presentations = session.stimulus_presentations
    metrics = (
        metrics.merge(stimulus_presentations["trials_id"], left_on="stimulus_id", right_index=True)
        .merge(rolling_perf, left_on="trials_id", right_index=True)
        .merge(rewards_metric[["stimulus_id", "volume"]], on="stimulus_id", how="left")
        .drop(columns=["trials_id", "stimulus_id"])
    )
    # Assign stimulus presentations that weren't rewarded to make volume a step function
    csum = metrics["volume"].notnull().cumsum()
    metrics["volume"] = metrics["volume"].fillna(0).groupby(csum).transform('sum')
    metrics = metrics.loc[(metrics.isna().sum(axis=1) == 0)]
    if center:
        metrics[["pupil_area", "speed"]] -= metrics[["pupil_area", "speed"]].mean(axis=0)
        metrics = metrics.loc[(metrics["pupil_area"] <= 3500)]
    
    metrics = metrics.loc[:, ~metrics.columns.str.startswith("timestamps")]
    return metrics

def get_spike_rates(session: BehaviorEcephysSession) -> tuple[np.array, pd.DataFrame]:
    """Get spike rates over stimulus presentations for a session
    
    Parameters
    ----------
    session
        The ecephys session to get spike rates for.
    
    Returns
    -------
    rates
        (n_units x n_stimuli) array of spiking rates.
    rates_df
        The ``rates`` array as a dataframe with an extra column
        indicating the region the unit is in.
    """
    
    spikes = session.spike_times
    units = session.get_units().join(unit_table["structure_acronym"])
    stimuli = session.stimulus_presentations
    stimuli = stimuli.loc[(stimuli["active"])]
    units = units[(units.isi_violations < 0.5) 
                    & (units.amplitude_cutoff < 0.1) 
                     & (units.presence_ratio > 0.9)
                ]
    rates = np.zeros((units.shape[0], stimuli.shape[0]))
    for i, unit_id in enumerate(units.index):
        unit_data = pd.DataFrame({"timestamps": spikes[unit_id], "spikes": np.ones(len(spikes[unit_id]))})
        counts = align_to_stimulus(unit_data, session).groupby("stimulus_id").sum()["spikes"]
        lengths = stimuli.end_time - stimuli.start_time
        rate = counts/lengths
        rates[i] = rate

    columns = [f"t_{timestep}" for timestep in stimuli.index]
    rates_df = pd.DataFrame(rates, index=units.index, columns=columns)
    rates_df = rates_df.join(units["structure_acronym"])
    
    return rates, rates_df

def plot_areas(
    session_rates: pd.DataFrame, 
    behavior_metric: np.array, 
    areas: Union[str, list[str]] = None, 
):
    """
    Plot the mean activity of a brain region along with individual unit activity along with
    behavioral state.
    
    If no areas are provided, all areas will be plotted.
    
    Parameters
    ----------
    session_rates
        The firing rates to plot.
    behavior_metric
        The behavior metric vector to plot.
    areas
        Single or list of brain regions to plot.
        Default behavior is to plot all regions
    
    Returns
    -------
    fig, ax
        Matplotlib plot
    """
    if areas is None:
        areas = session_rates["structure_acronym"].unique()
    elif isinstance(areas, str):
        areas = [areas]
    
    
    fig, axes = plt.subplots(len(areas), 2, figsize=(10, 4 * len(areas)));   
    axes = np.expand_dims(axes, 0) if len(areas) == 1 else axes
    
    session_rates_t = session_rates.iloc[:, :-1].T
    
    behavior_metric[~behavior_metric.isna()] /= behavior_metric.max()
    behavior_state = behavior_metric
    
    for i, area in enumerate(areas):
        area_activity_idx = session_rates.loc[session_rates["structure_acronym"] == area].index
        (area_rates := session_rates_t.loc[:, area_activity_idx]).mean(axis=1).plot(ax=axes[i,0]);
        
    
        axes[i, 0].set_title(f"(mean) activity for {area}");
        axes[i, 0].plot(behavior_state * area_rates.to_numpy().mean(axis=(0,1)) , linewidth=3);
        axes[i, 0].set_ylabel("Firing rate (over stimulus presentation")
        
        area_rates.plot(ax=axes[i,1], alpha=0.4);
        axes[i, 1].plot(behavior_state * area_rates.to_numpy().max(axis=(0,1))/5, color="orange", linewidth=3);
        axes[i, 1].set_title(f"activity for {area}");
        axes[i, 1].legend().remove();
        
    fig.tight_layout();
    return fig, axes

def plot_area_units(
    session_rates: pd.DataFrame, 
    behavior_metric: np.array, 
    area: str, 
) -> None:
    """
    Plot the individual unit activity for a single brain region.
        
    Parameters
    ----------
    session_rates
        The firing rates to plot.
    behavior_state
        The behavior metric vector to plot.
    area
        Region to plot units for.
    
    Returns
    -------
    fig, ax
        Matplotlib plot
    """
    if area not in session_rates["structure_acronym"].unique():
        raise ValueError("Could not find brain region")
        
    area_rates = session_rates.loc[session_rates["structure_acronym"] == area]
    
    area_rates_t = area_rates.iloc[:, :-1].T
    
    behavior_metric[~behavior_metric.isna()] /= behavior_metric.max()
    behavior_state = behavior_metric
    
    fig, axes = plt.subplots((n_units := area_rates_t.shape[1]), 1, figsize=(8, 6 * n_units), sharex=True);
    axes = np.reshape(axes, -1) if n_units == 1 else axes
    for i, ax in enumerate(axes):
        area_rates_t.iloc[:, i].plot(ax=ax)
        ax.plot(behavior_state * area_rates_t.iloc[:, i].to_numpy().max(axis=(0))/5, color="orange", linewidth=3);
        ax.set_title(f"activity for {area_rates_t.columns[i]}");
        ax.set_ylabel("Firing rate (over stimulus presentation")

    return fig, axes