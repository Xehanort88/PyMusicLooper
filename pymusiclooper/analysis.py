import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import librosa
import numpy as np
import scipy.signal
from numba import njit

from pymusiclooper.audio import MLAudio
from pymusiclooper.exceptions import LoopNotFoundError


@dataclass
class LoopPair:
    """A data class that encapsulates the loop point related data.
    Contains:
        loop_start: int (exact loop start position in samples)
        loop_end: int (exact loop end position in samples)
        note_distance: float
        loudness_difference: float
        score: float. Defaults to 0.
        from_metadata: bool (loop points were read from the file's metadata tags instead of being detected). Defaults to False.
    """

    _loop_start_frame_idx: int
    _loop_end_frame_idx: int
    note_distance: float
    loudness_difference: float
    score: float = 0
    loop_start: int = 0
    loop_end: int = 0
    from_metadata: bool = False


def find_best_loop_points(
    mlaudio: MLAudio,
    min_duration_multiplier: float = 0.35,
    min_loop_duration: Optional[float] = None,
    max_loop_duration: Optional[float] = None,
    approx_loop_start: Optional[float] = None,
    approx_loop_end: Optional[float] = None,
    brute_force: bool = False,
    disable_pruning: bool = False,
) -> List[LoopPair]:
    """Finds the best loop points for a given audio track, given the constraints specified

    Args:
        mlaudio (MLAudio): The MLAudio object to use for analysis
        min_duration_multiplier (float, optional): The minimum duration of a loop as a multiplier of track duration. Defaults to 0.35.
        min_loop_duration (float, optional): The minimum duration of a loop (in seconds). Defaults to None.
        max_loop_duration (float, optional): The maximum duration of a loop (in seconds). Defaults to None.
        approx_loop_start (float, optional): The approximate location of the desired loop start (in seconds). If specified, must specify approx_loop_end as well. Defaults to None.
        approx_loop_end (float, optional): The approximate location of the desired loop end (in seconds). If specified, must specify approx_loop_start as well. Defaults to None.
        brute_force (bool, optional): Checks the entire track instead of the detected beats (disclaimer: runtime may be significantly longer). Defaults to False.
        disable_pruning (bool, optional): Returns all the candidate loop points without filtering. Defaults to False.
    Raises:
        LoopNotFoundError: raised in case no loops were found

    Returns:
        List[LoopPair]: A list of `LoopPair` objects containing the loop points related data. See the `LoopPair` class for more info.
    """
    runtime_start = time.perf_counter()
    min_loop_duration = (
        mlaudio.seconds_to_frames(min_loop_duration)
        if min_loop_duration is not None
        else mlaudio.seconds_to_frames(
            int(min_duration_multiplier * mlaudio.total_duration)
        )
    )
    max_loop_duration = (
        mlaudio.seconds_to_frames(max_loop_duration)
        if max_loop_duration is not None
        else mlaudio.seconds_to_frames(mlaudio.total_duration)
    )

    # Loop points must be at least 1 frame apart
    min_loop_duration = max(1, min_loop_duration)

    if approx_loop_start is not None and approx_loop_end is not None:
        # Skipping the unnecessary beat analysis (in this case) speeds up the analysis runtime by ~2x
        # and significantly reduces the total memory consumption
        chroma, frame_loudness, _, _ = _analyze_audio(mlaudio, skip_beat_analysis=True)
        # Set bpm to a general average of 120
        bpm = 120.0

        approx_loop_start = mlaudio.seconds_to_frames(
            approx_loop_start, apply_trim_offset=True
        )
        approx_loop_end = mlaudio.seconds_to_frames(
            approx_loop_end, apply_trim_offset=True
        )

        n_frames_to_check = mlaudio.seconds_to_frames(2)

        # Adjust min and max loop duration checks to the specified range
        min_loop_duration = (
            (approx_loop_end - n_frames_to_check)
            - (approx_loop_start + n_frames_to_check)
            - 1
        )
        max_loop_duration = (
            (approx_loop_end + n_frames_to_check)
            - (approx_loop_start - n_frames_to_check)
            + 1
        )

        # Override the beats to check with the specified approx points +/- 2 seconds
        beats = np.concatenate(
            [
                np.arange(
                    start=max(0, approx_loop_start - n_frames_to_check),
                    stop=min(
                        mlaudio.seconds_to_frames(mlaudio.total_duration),
                        approx_loop_start + n_frames_to_check,
                    ),
                ),
                np.arange(
                    start=max(0, approx_loop_end - n_frames_to_check),
                    stop=min(
                        mlaudio.seconds_to_frames(mlaudio.total_duration),
                        approx_loop_end + n_frames_to_check,
                    ),
                ),
            ]
        )
    elif brute_force:
        # Similarly skip beat analysis, as the results will not be used
        chroma, frame_loudness, _, _ = _analyze_audio(mlaudio, skip_beat_analysis=True)
        bpm = 120.0
        beats = np.arange(start=0, stop=chroma.shape[-1], step=1, dtype=int)
        logging.info(f"Overriding number of frames to check with: {beats.size}")
        logging.info(f"Estimated iterations required using brute force: {int(beats.size*beats.size*(1-(min_loop_duration/chroma.shape[-1])))}")
        logging.info("**NOTICE** The program may appear frozen, but processing will continue in the background. This operation may take several minutes to complete.")
    else: # normal mode of operation
        chroma, frame_loudness, bpm, beats = _analyze_audio(mlaudio)
        logging.info(f"Detected {beats.size} beats at {bpm:.0f} bpm")

    logging.info(
        "Finished initial audio processing in {:.3f}s".format(
            time.perf_counter() - runtime_start
        )
    )

    initial_pairs_start_time = time.perf_counter()

    # Since numba jitclass cannot be cached, the pair data must be stored temporarily in a list of tuple
    # (instead of a list of LoopPairs directly) and then loaded into a list of LoopPair objects using list comprehension
    # The features of the beat frames only, laid out contiguously for fast comparisons
    unproc_candidate_pairs = _find_candidate_pairs(
        np.ascontiguousarray(chroma[:, beats].T),
        frame_loudness[beats],
        beats,
        min_loop_duration,
        max_loop_duration,
    )
    candidate_pairs = [
        LoopPair(
            _loop_start_frame_idx=tup[0],
            _loop_end_frame_idx=tup[1],
            note_distance=tup[2],
            loudness_difference=tup[3],
        )
        for tup in unproc_candidate_pairs
    ]

    n_candidate_pairs = len(candidate_pairs) if candidate_pairs is not None else 0
    logging.info(
        f"Found {n_candidate_pairs} possible loop points in"
        f" {(time.perf_counter() - initial_pairs_start_time):.3f}s"
    )

    if not candidate_pairs:
        raise LoopNotFoundError(
            f"No loop points found for \"{mlaudio.filename}\" with current parameters."
        )

    filtered_candidate_pairs = _assess_and_filter_loop_pairs(
        mlaudio, chroma, bpm, candidate_pairs, disable_pruning
    )

    # Set the exact loop start and end in samples. The frame-level points are refined by lining up
    # the waveforms at the loop points, then choosing the seam where they differ the least;
    # if the waveforms do not match well enough for that, each point is moved to its nearest zero crossing.
    # Avoids audio popping/clicking while looping as much as possible.
    mono_playback_audio = mlaudio.playback_audio.mean(axis=1)
    for pair in filtered_candidate_pairs:
        if mlaudio.trim_offset > 0:
            pair._loop_start_frame_idx = int(
                mlaudio.apply_trim_offset(pair._loop_start_frame_idx)
            )
            pair._loop_end_frame_idx = int(
                mlaudio.apply_trim_offset(pair._loop_end_frame_idx)
            )
        pair.loop_start, pair.loop_end = _refine_loop_points(
            mlaudio.playback_audio,
            mono_playback_audio,
            mlaudio.rate,
            int(mlaudio.frames_to_samples(pair._loop_start_frame_idx)),
            int(mlaudio.frames_to_samples(pair._loop_end_frame_idx)),
        )

    if not filtered_candidate_pairs:
        raise LoopNotFoundError(
            f"No loop points found for {mlaudio.filename} with current parameters."
        )

    # prefer longer loops for highly similar sequences
    # (must run after the loop positions in samples are set, since it compares loop durations)
    if len(filtered_candidate_pairs) > 1:
        _prioritize_duration(filtered_candidate_pairs)

    logging.info(
        f"Filtered to {len(filtered_candidate_pairs)} best candidate loop points"
    )
    logging.info(
        f"Total analysis runtime: {time.perf_counter() - runtime_start:.3f}s"
    )

    return filtered_candidate_pairs


def _analyze_audio(
    mlaudio: MLAudio, skip_beat_analysis=False
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Performs the main audio analysis required

    Args:
        mlaudio (MLAudio): the MLAudio object to perform analysis on
        skip_beat_analysis (bool, optional): Skips beat analysis if true and returns None for bpm and beats. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray]: a tuple containing the (chroma spectrogram,
        loudness of each frame (the maximum of the perceptually weighted power spectrogram in dB), tempo/bpm, frame indices of detected beats)
    """
    # The full-resolution spectrograms are large (several GB for long tracks),
    # so each one is released as soon as the next, smaller, representation is computed from it
    S_power = np.abs(librosa.core.stft(y=mlaudio.audio))
    S_power **= 2
    # The sample rate maps the frequency bins to pitch classes (librosa's default of 22050 Hz would shift them,
    # e.g. by ~1.5 semitones for 48 kHz audio)
    chroma = librosa.feature.chroma_stft(
        S=S_power, sr=mlaudio.rate, tuning=_estimate_tuning(S_power, sr=mlaudio.rate)
    )
    S_weighed = librosa.core.perceptual_weighting(
        S=S_power, frequencies=librosa.fft_frequencies(sr=mlaudio.rate)
    )
    del S_power
    mel_spectrogram = librosa.feature.melspectrogram(
        S=S_weighed, sr=mlaudio.rate, n_mels=128, fmax=8000
    )
    # Only the loudest bin of each frame is compared, and converting to dB preserves the order of the values,
    # so the frame maxima are converted instead of the whole spectrogram (with the same reference: its median)
    median_power = np.median(S_weighed)
    frame_loudness = librosa.power_to_db(S_weighed.max(axis=0), ref=lambda _: median_power)
    del S_weighed

    if skip_beat_analysis:
        return chroma, frame_loudness, None, None

    try:
        onset_env = librosa.onset.onset_strength(S=mel_spectrogram)

        # Not given the sample rate, so the tempo is estimated as if the audio were at 22050 Hz: the beat positions
        # are unaffected, but the bpm is scaled by 22050 / rate. The bpm only sets the length of audio compared when
        # scoring the loops (see _assess_and_filter_loop_pairs), which was tuned with this scaling; passing the real
        # sample rate gave no better loops on real tracks, so it is kept as-is.
        pulse = librosa.beat.plp(onset_envelope=onset_env)
        beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
        bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env)

        beats = np.union1d(beats, beats_plp)
        beats = np.sort(beats)

        if isinstance(bpm, np.ndarray):
            bpm = bpm[0]
    except Exception as e:
        raise LoopNotFoundError(f"Beat analysis failed for \"{mlaudio.filename}\". Cannot continue.") from e

    return chroma, frame_loudness, bpm, beats


# Number of frames tracked at a time when estimating the tuning, which bounds the memory it uses
_TUNING_CHUNK_FRAMES = 4096


def _estimate_tuning(S_power: np.ndarray, sr: float, bins_per_octave: int = 12) -> float:
    """Gives the same result as `librosa.estimate_tuning(S=S_power, sr=sr, bins_per_octave=bins_per_octave)`
    (which chroma_stft uses when no tuning is given), while using a fraction of the memory.

    librosa's pitch tracking allocates several arrays the size of the whole spectrogram (several GB for long tracks).
    It works on each frame independently, and the tuning estimate does not depend on the order of the detected pitches,
    so tracking chunks of frames and pooling the detected pitches gives the same estimate.

    Args:
        S_power (np.ndarray): Power spectrogram, in the shape `(1 + n_fft/2, n_frames)`
        sr (float): Sample rate used to convert the frequency bins to Hz
        bins_per_octave (int, optional): Number of frequency bins per octave. Defaults to 12.

    Returns:
        float: The estimated tuning deviation, in fractions of a bin
    """
    pitches, magnitudes = [], []
    for start in range(0, S_power.shape[-1], _TUNING_CHUNK_FRAMES):
        pitch, magnitude = librosa.piptrack(S=S_power[:, start:start + _TUNING_CHUNK_FRAMES], sr=sr)
        # Only frequencies > 0 are detected pitches
        detected = pitch > 0
        pitches.append(pitch[detected])
        magnitudes.append(magnitude[detected])
    pitches = np.concatenate(pitches)
    magnitudes = np.concatenate(magnitudes)

    # Only the pitches with at least the median magnitude count
    threshold = np.median(magnitudes) if pitches.size else 0.0
    return librosa.pitch_tuning(pitches[magnitudes >= threshold], bins_per_octave=bins_per_octave)


@njit(cache=True)
def _find_candidate_pairs(
    beat_chroma: np.ndarray,
    beat_loudness: np.ndarray,
    beats: np.ndarray,
    min_loop_duration: int,
    max_loop_duration: int,
) -> List[Tuple[int, int, float, float]]:
    """Generates a list of all valid candidate loop pairs using combinations of beat indices,
    by comparing the notes using the chroma spectrogram and their loudness difference.

    Every pair of beats is compared, so the distances are computed with scalar loops over the 12 pitch classes
    (instead of array operations, which allocate a temporary array for each of the millions of comparisons).

    Args:
        beat_chroma (np.ndarray): The chroma of each beat frame (float32), in the shape `(n_beats, 12)`, C-contiguous
        beat_loudness (np.ndarray): The loudness of each beat frame in dB (float32), in the shape `(n_beats,)`
        beats (np.ndarray): The frame indices of detected beats
        min_loop_duration (int): Minimum loop duration (in frames)
        max_loop_duration (int): Maximum loop duration (in frames)

    Returns:
        List[Tuple[int, int, float, float]]: A list of tuples containing each candidate loop pair data in the following format (loop_start, loop_end, note_distance, loudness_difference)
    """
    candidate_pairs = []

    # Magic constants
    ## Mainly found through trial and error,
    ## higher values typically result in the inclusion of musically unrelated beats/notes
    ACCEPTABLE_NOTE_DEVIATION = 0.0875
    ## Since the loudness comparison takes the loudest bin of perceptually weighted power frames in dB,
    ## the difference should be imperceptible (ideally, close to 0)
    ## Based on trial and error, values higher than ~0.5 have a perceptible
    ## difference in loudness
    ACCEPTABLE_LOUDNESS_DIFFERENCE = 0.5

    n_beats, n_pitch_classes = beat_chroma.shape

    # The note distance accepted for each loop end: the norm of its chroma scaled by ACCEPTABLE_NOTE_DEVIATION
    # (in float64, as in the original array-based implementation, so that the results are unchanged)
    deviation = np.empty(n_beats)
    for idx in range(n_beats):
        deviation_squares = 0.0
        for k in range(n_pitch_classes):
            value = np.float64(beat_chroma[idx, k]) * ACCEPTABLE_NOTE_DEVIATION
            deviation_squares += value * value
        deviation[idx] = np.sqrt(deviation_squares)

    for idx in range(n_beats):
        loop_end = beats[idx]
        for start_idx in range(n_beats):
            loop_start = beats[start_idx]
            loop_length = loop_end - loop_start
            if loop_length < min_loop_duration:
                break
            if loop_length > max_loop_duration:
                continue
            # Euclidean distance of the chroma vectors (in float32, like the chroma)
            distance_squares = np.float32(0.0)
            for k in range(n_pitch_classes):
                difference = beat_chroma[idx, k] - beat_chroma[start_idx, k]
                distance_squares += difference * difference
            note_distance = np.sqrt(distance_squares)

            if note_distance <= deviation[idx]:
                loudness_difference = abs(beat_loudness[idx] - beat_loudness[start_idx])
                loop_pair = (
                    int(loop_start),
                    int(loop_end),
                    note_distance,
                    loudness_difference,
                )
                if loudness_difference <= ACCEPTABLE_LOUDNESS_DIFFERENCE:
                    candidate_pairs.append(loop_pair)

    return candidate_pairs


def _assess_and_filter_loop_pairs(
    mlaudio: MLAudio,
    chroma: np.ndarray,
    bpm: float,
    candidate_pairs: List[LoopPair],
    disable_pruning: bool = False,
) -> List[LoopPair]:
    """Assigns the scores to each loop pair and prunes the list of candidate loop pairs

    Args:
        mlaudio (MLAudio): MLAudio object of the track being analyzed
        chroma (np.ndarray): The chroma spectrogram
        bpm (float): The estimated bpm/tempo of the track
        candidate_pairs (List[LoopPair]): The list of candidate loop pairs found
        disable_pruning (bool, optional): Returns all the candidate loop points without filtering. Defaults to False.

    Returns:
        List[LoopPair]: A scored and filtered list of valid loop candidate pairs
    """
    beats_per_second = bpm / 60
    num_test_beats = 12
    seconds_to_test = num_test_beats / beats_per_second
    test_offset = mlaudio.samples_to_frames(int(seconds_to_test * mlaudio.rate))

    # adjust offset for very short tracks to 25% of its length
    if test_offset > chroma.shape[-1]:
        test_offset = chroma.shape[-1] // 4

    # Prune candidates if there are too many
    if len(candidate_pairs) >= 100 and not disable_pruning:
        pruned_candidate_pairs = _prune_candidates(candidate_pairs)
    else:
        pruned_candidate_pairs = candidate_pairs

    weights = _weights(test_offset, start=max(2, test_offset // num_test_beats), stop=1)

    pair_score_list = [
        _calculate_loop_score(
            int(pair._loop_start_frame_idx),
            int(pair._loop_end_frame_idx),
            chroma,
            test_duration=test_offset,
            weights=weights,
        )
        for pair in pruned_candidate_pairs
    ]
    # Add cosine similarity as score
    for pair, score in zip(pruned_candidate_pairs, pair_score_list):
        pair.score = score

    # re-sort based on new score
    pruned_candidate_pairs = sorted(
        pruned_candidate_pairs, reverse=True, key=lambda x: x.score
    )
    return pruned_candidate_pairs


def _prune_candidates(
    candidate_pairs: List[LoopPair],
    keep_top_notes: float = 75,
    keep_top_loudness: float = 50,
    acceptable_loudness=0.25,
) -> List[LoopPair]:
    db_diff_array = np.array([pair.loudness_difference for pair in candidate_pairs])
    note_dist_array = np.array([pair.note_distance for pair in candidate_pairs])

    # Minimum value used to avoid issues with tracks with lots of silence
    epsilon = 1e-3
    min_adjusted_db_diff_array = db_diff_array[db_diff_array > epsilon]
    min_adjusted_note_dist_array = note_dist_array[note_dist_array > epsilon]

    # Avoid index errors by having at least 3 elements when performing percentile-based pruning
    # Otherwise, skip by setting the value to the highest available
    if min_adjusted_db_diff_array.size > 3:
        db_threshold = np.percentile(
            min_adjusted_db_diff_array, keep_top_loudness
        )
    else:
        db_threshold = np.max(db_diff_array)

    if min_adjusted_note_dist_array.size > 3:
        note_dist_threshold = np.percentile(
            min_adjusted_note_dist_array, keep_top_notes
        )
    else:
        note_dist_threshold = np.max(note_dist_array)

    # Lower values are better
    indices_that_meet_cond = np.flatnonzero(
        (db_diff_array <= max(acceptable_loudness, db_threshold)) & (note_dist_array <= note_dist_threshold)
    )
    return [candidate_pairs[idx] for idx in indices_that_meet_cond]


def _prioritize_duration(pair_list: List[LoopPair]) -> List[LoopPair]:
    db_diff_array = np.array([pair.loudness_difference for pair in pair_list])
    db_threshold = np.median(db_diff_array)

    duration_argmax = 0
    duration_max = 0

    score_array = np.array([pair.score for pair in pair_list])
    score_threshold = np.percentile(score_array, 90)

    # Must be a negligible difference from the top score
    score_threshold = max(score_threshold, pair_list[0].score - 1e-4)

    # Since pair_list is already sorted
    # Break the loop if the condition is not met
    for idx, pair in enumerate(pair_list):
        if pair.score < score_threshold:
            break
        duration = pair.loop_end - pair.loop_start
        if duration > duration_max and pair.loudness_difference <= db_threshold:
            duration_max, duration_argmax = duration, idx

    if duration_argmax:
        pair_list.insert(0, pair_list.pop(duration_argmax))


def _calculate_loop_score(
    b1: int,
    b2: int,
    chroma: np.ndarray,
    test_duration: int,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculates the similarity of two sequences given the starting indices `b1` and `b2` for the period of the `test_duration` specified.
        Returns the best score based on the cosine similarity of subsequent (or preceding) notes.

    Args:
        b1 (int): Frame index of the first beat to compare
        b2 (int): Frame index of the second beat to compare
        chroma (np.ndarray): The chroma spectrogram of the audio
        test_duration (int): How many frames along the chroma spectrogram to test.
        weights (np.ndarray, optional): If specified, will provide a weighted average of the note scores according to the weight array provided. Defaults to None.

    Returns:
        float: the weighted average of the cosine similarity of the notes along the tested region
    """
    lookahead_score = _calculate_subseq_beat_similarity(
        b1, b2, chroma, test_duration, weights=weights
    )
    lookbehind_score = _calculate_subseq_beat_similarity(
        b1, b2, chroma, -test_duration, weights=weights[::-1]
    )

    return max(lookahead_score, lookbehind_score)


def _calculate_subseq_beat_similarity(
    b1_start: int,
    b2_start: int,
    chroma: np.ndarray,
    test_end_offset: int,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculates the similarity of subsequent notes of the two specified indices (b1_start, b2_start) using cosine similarity

    Args:
        b1_start (int): Starting frame index of the first beat to compare
        b2_start (int): Starting frame index of the second beat to compare
        chroma (np.ndarray): The chroma spectrogram of the audio
        test_end_offset (int): The number of frames to offset from the starting index. If negative, will be testing the preceding frames instead of the subsequent frames.
        weights (np.ndarray, optional): If specified, will provide a weighted average of the note scores according to the weight array provided. Defaults to None.

    Returns:
        float: the weighted average of the cosine similarity of the notes along the tested region
    """
    chroma_len = chroma.shape[-1]
    test_length = abs(test_end_offset)

    if test_end_offset < 0:
        b1_end = b1_start
        b2_end = b2_start
        max_negative_offset = max(test_end_offset, -b1_start, -b2_start)
        b1_start += max_negative_offset
        b2_start += max_negative_offset
        max_offset = abs(max_negative_offset)
    else:
        # clip to chroma len
        b1_end = min(b1_start + test_length, chroma_len)
        b2_end = min(b2_start + test_length, chroma_len)
        # align testing lengths
        max_offset = min(b1_end - b1_start, b2_end - b2_start)
        b1_end, b2_end = (b1_start + max_offset, b2_start + max_offset)

    dot_prod = np.einsum(
        "ij,ij->j", chroma[..., b1_start:b1_end], chroma[..., b2_start:b2_end]
    )
    b1_norm = np.linalg.norm(chroma[..., b1_start:b1_end], axis=0)
    b2_norm = np.linalg.norm(chroma[..., b2_start:b2_end], axis=0)
    cosine_sim = dot_prod / (np.maximum(b1_norm * b2_norm, 1e-10))

    if max_offset < test_length:
        # Pad the missing frames on the side farthest from the loop point:
        # after the tested frames when looking ahead, before them when looking behind
        missing_frames = test_length - max_offset
        pad_width = (missing_frames, 0) if test_end_offset < 0 else (0, missing_frames)
        return np.average(
            np.pad(cosine_sim, pad_width=pad_width, mode="constant", constant_values=0),
            weights=weights,
        )
    else:
        return np.average(cosine_sim, weights=weights)


def _weights(length: int, start: int = 100, stop: int = 1):
    return np.geomspace(start, stop, num=length)


# STFT hop length used by the analysis (librosa's default); frame-level loop points are only this precise
_HOP_LENGTH = 512
# How far (in samples) the loop end may be moved to line up the waveforms
_ALIGNMENT_SEARCH_RADIUS = 2 * _HOP_LENGTH
# Length of the audio compared on each side of the loop points when aligning them (in seconds)
_ALIGNMENT_HALF_WINDOW = 0.075
# Below this correlation, the audio around the loop points does not match well enough
# for lining up the waveforms to be reliable (found by comparing both methods on real tracks)
_MIN_ALIGNMENT_CORRELATION = 0.3


def _refine_loop_points(
    audio: np.ndarray, mono_audio: np.ndarray, rate: int, loop_start: int, loop_end: int
) -> Tuple[int, int]:
    """Refines frame-level loop points to exact sample positions.

    Lines up the waveforms at the loop points, then picks the seam where they differ the least.
    If the waveforms do not match well enough, each point is moved to its nearest zero crossing instead.

    Args:
        audio (np.ndarray): Playback audio, in the shape `(samples, n_channels)`
        mono_audio (np.ndarray): The playback audio mixed down to mono, in the shape `(samples,)`
        rate (int): Sample rate of the audio
        loop_start (int): Approximate loop start in samples
        loop_end (int): Approximate loop end in samples

    Returns:
        Tuple[int, int]: The refined (loop_start, loop_end)
    """
    aligned_loop_end, correlation = _align_loop_end(mono_audio, rate, loop_start, loop_end)
    if correlation >= _MIN_ALIGNMENT_CORRELATION:
        return _best_seam(audio, rate, loop_start, aligned_loop_end)
    return (
        nearest_zero_crossing(audio, rate, loop_start),
        nearest_zero_crossing(audio, rate, loop_end),
    )


def _align_loop_end(mono_audio: np.ndarray, rate: int, loop_start: int, loop_end: int) -> Tuple[int, float]:
    """Moves the loop end (within `_ALIGNMENT_SEARCH_RADIUS` samples) to where the waveform around it best matches
    the waveform around the loop start, using normalized cross-correlation.

    Args:
        mono_audio (np.ndarray): Mono playback audio, in the shape `(samples,)`
        rate (int): Sample rate of the audio
        loop_start (int): Loop start in samples
        loop_end (int): Approximate loop end in samples

    Returns:
        Tuple[int, float]: The aligned loop end, and the correlation of the waveforms at that point (-1 to 1).
        The correlation is 0 if there was not enough audio around the loop points to compare.
    """
    # The loop end must stay after the loop start
    radius = min(_ALIGNMENT_SEARCH_RADIUS, loop_end - loop_start - 1)
    half_window = int(_ALIGNMENT_HALF_WINDOW * rate)
    n_samples = mono_audio.shape[0]

    # Compare as much audio as available on each side, up to half_window
    before = min(half_window, loop_start, loop_end - radius)
    after = min(half_window, n_samples - loop_start, n_samples - loop_end - radius)
    if radius < 0 or before + after < _HOP_LENGTH:
        return loop_end, 0.0

    reference = mono_audio[loop_start - before:loop_start + after].astype(np.float64)
    search = mono_audio[loop_end - before - radius:loop_end + after + radius].astype(np.float64)

    # correlation[i] compares the reference with the search window shifted by (i - radius) samples
    correlation = scipy.signal.correlate(search, reference, mode="valid", method="fft")
    cumulative_energy = np.concatenate(([0.0], np.cumsum(search**2)))
    window_energy = cumulative_energy[reference.size:] - cumulative_energy[:-reference.size]
    normalized = correlation / np.sqrt(np.maximum(window_energy * np.dot(reference, reference), 1e-20))

    best = int(np.argmax(normalized))
    return loop_end + best - radius, float(normalized[best])


def _best_seam(audio: np.ndarray, rate: int, loop_start: int, loop_end: int) -> Tuple[int, int]:
    """Shifts both loop points by the same amount (keeping the loop length) to where the waveforms at the
    loop start and loop end differ the least, within +/-5ms. The difference at the jump is what causes clicks.

    Args:
        audio (np.ndarray): Playback audio, in the shape `(samples, n_channels)`
        rate (int): Sample rate of the audio
        loop_start (int): Loop start in samples
        loop_end (int): Loop end in samples

    Returns:
        Tuple[int, int]: The shifted (loop_start, loop_end)
    """
    radius = max(1, rate // 200)
    # Compare the two samples before and after each point
    offsets = np.arange(-2, 2)
    lowest_shift = max(-radius, -(loop_start + offsets[0]))
    highest_shift = min(radius, audio.shape[0] - 1 - (loop_end + offsets[-1]))
    if lowest_shift > highest_shift:
        return loop_start, loop_end

    shifts = np.arange(lowest_shift, highest_shift + 1)
    idx = shifts[:, np.newaxis] + offsets[np.newaxis, :]
    difference = np.abs(audio[loop_start + idx] - audio[loop_end + idx]).sum(axis=(1, 2))
    # Prefer the smallest shift among equally good ones
    difference += 1e-6 * np.abs(shifts)

    shift = int(shifts[np.argmin(difference)])
    return loop_start + shift, loop_end + shift


@njit(cache=True)
def nearest_zero_crossing(audio: np.ndarray, rate: int, sample_idx: int) -> int:
    """Implementation of Audacity's `At Zero Crossings` feature. https://manual.audacityteam.org/man/select_menu_at_zero_crossings.html
    Description is based on the relevant Audacity manual page, due to identical behaviour.

    Returns the best closest sample point that is at a rising zero crossing point.
    This is a point where a line joining the audio samples rises from left to right and crosses the zero horizontal line that represents silence.
    The shift in audio position is not itself detectable to the ear, but the fact that the joins in the waveform are now of matching height helps avoid clicks in audio.

    This feature does not necessarily find the nearest zero crossing to the current position. It aims to find the crossing where the average amplitude of samples in the vicinity is lowest.

    Args:
        audio (np.ndarray): Numpy array containing the playback audio; must be in the shape `(samples, n_channels)`
        rate (int): Sample rate of the provided audio
        sample_idx (int): The index of the sample point to return the nearest zero crossing of

    Returns:
        int: the index of the best sample point that is at a rising zero crossing point closest to the `sample_idx` provided, returns `sample_idx` if none where found
    """
    # Re-implementation of Audacity's NearestZeroCrossing function in Python
    # https://github.com/audacity/audacity/blob/057bf4ee6f71962cd8ecc6dbccf0852695340758/src/menus/SelectMenus.cpp#L30
    # Original credit goes to the Audacity team and contributors
    n_channels = audio.shape[1]

    # Window is 1/100th of a second
    window_size = int(max(1, rate / 100))

    # Create sample window centered around sample_idx
    offset = window_size // 2
    neg_offset = max(0, sample_idx - offset)
    pos_offset = min(audio.shape[0], sample_idx + offset)
    sample_window = audio[neg_offset:pos_offset]

    # Adjusts the indexing offset in case the left side of sample_idx was clipped
    offset_correction = abs(sample_idx - offset) if sample_idx - offset < 0 else 0

    sample_window_length = sample_window.shape[0]
    dist = np.zeros(sample_window_length)

    for channel in range(n_channels):
        prev = 2.0
        one_dist = sample_window[..., channel].copy()
        for i in range(sample_window_length):
            fdist = np.abs(one_dist[i])
            if prev * one_dist[i] > 0:  # both same sign? No good.
                fdist += 0.4  # No good if same sign.
            elif prev > 0.0:
                fdist += 0.1  # medium penalty for downward crossing.
            prev = one_dist[i]
            one_dist[i] = fdist

        for i in range(sample_window_length):
            dist[i] += one_dist[i]
            dist[i] += 0.1 * abs(i - offset + offset_correction) / (window_size / 2)

    argmin = np.argmin(dist)
    minimum_dist = dist[argmin]

    # If we're worse than 0.2 on average, on one track, then no good.
    if (n_channels == 1) and (minimum_dist > (0.2 * n_channels)):
        return sample_idx
    # If we're worse than 0.6 on average, on multi-track, then no good.
    if (n_channels > 1) and (minimum_dist > (0.6 * n_channels)):
        return sample_idx

    return int(sample_idx + argmin - offset + offset_correction)
