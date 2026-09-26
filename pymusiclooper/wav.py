"""Chunk-level editing of WAV files: lossless trimming, and reading/writing loop points in the sampler (smpl) chunk.

Every chunk of the source file is copied verbatim, so everything stored besides the audio is kept as-is:
the format chunk, tags (LIST/INFO and ID3 chunks), sampler loops (smpl), cue points, broadcast extension (bext), etc.
When trimming, only the audio data chunk is shortened, and the sample count of the fact chunk (if any) and the RIFF size
are updated.

The sampler chunk is where many game engines, samplers and audio tools read the loop points of WAV files from.
"""

import struct
from typing import BinaryIO, List, NamedTuple, Optional, Tuple

# Sampler chunk header: manufacturer, product, sample period (ns), MIDI unity note, MIDI pitch fraction,
# SMPTE format, SMPTE offset, number of loops, size of the sampler-specific data following the loops
_SMPL_HEADER_FORMAT = "9I"
_SMPL_HEADER_SIZE = 36
_SMPL_LOOP_COUNT_OFFSET = 28
# Each loop: cue point ID, type, start, end (inclusive), fraction, play count (0 = infinite)
_SMPL_LOOP_FORMAT = "6I"
_SMPL_LOOP_SIZE = 24
_SMPL_FORWARD_LOOP = 0
_DEFAULT_MIDI_UNITY_NOTE = 60


class _ChunkHeader(NamedTuple):
    chunk_id: bytes
    offset: int  # Offset of the chunk's data in the file
    size: int  # Size of the chunk's data present in the file


def _read_riff_header(f: BinaryIO) -> Tuple[bytes, str]:
    """Returns the RIFF identifier (RIFF or RIFX) and the struct endianness of the file's integers."""
    header = f.read(12)
    if header[:4] not in (b"RIFF", b"RIFX") or header[8:12] != b"WAVE":
        raise ValueError("Not a RIFF WAVE file.")
    return header[:4], "<" if header[:4] == b"RIFF" else ">"


def _read_chunk_headers(f: BinaryIO, endian: str) -> List[_ChunkHeader]:
    """Lists the chunks of the file without reading their data, so that large files can be searched quickly."""
    file_size = f.seek(0, 2)
    chunks = []
    offset = 12
    while offset + 8 <= file_size:
        f.seek(offset)
        chunk_id, size = struct.unpack(f"{endian}4sI", f.read(8))
        # Files written by interrupted or streaming encoders can leave the data size as a placeholder:
        # 0 (then the data runs to the end of the file), or a size past the end of the file
        if chunk_id == b"data" and size == 0:
            size = file_size - offset - 8
        chunks.append(_ChunkHeader(chunk_id, offset + 8, min(size, file_size - offset - 8)))
        # Chunks are padded to an even size
        offset += 8 + size + (size & 1)
    return chunks


def _read_chunks(filepath: str) -> Tuple[bytes, str, List[Tuple[bytes, bytes]]]:
    """Returns the RIFF identifier, the endianness and the (chunk ID, chunk data) of every chunk of the file."""
    with open(filepath, "rb") as f:
        riff_id, endian = _read_riff_header(f)
        chunks = []
        for chunk in _read_chunk_headers(f, endian):
            f.seek(chunk.offset)
            chunks.append((chunk.chunk_id, f.read(chunk.size)))
    return riff_id, endian, chunks


def _write_riff(filepath: str, riff_id: bytes, endian: str, chunks: List[Tuple[bytes, bytes]]):
    body = []
    for chunk_id, chunk_data in chunks:
        body += [chunk_id, struct.pack(f"{endian}I", len(chunk_data)), chunk_data]
        if len(chunk_data) & 1:
            body.append(b"\x00")
    body = b"".join(body)

    with open(filepath, "wb") as f:
        f.write(riff_id + struct.pack(f"{endian}I", 4 + len(body)) + b"WAVE" + body)


def trim_wav(src_filepath: str, dest_filepath: str, n_frames: int) -> int:
    """Writes a copy of a PCM/float WAV file that ends after `n_frames` frames, keeping all its other chunks.

    Args:
        src_filepath (str): Path to the source WAV file.
        dest_filepath (str): Path of the trimmed file to write.
        n_frames (int): Number of frames (samples per channel) to keep from the start of the audio.

    Raises:
        ValueError: if the file is not a RIFF/RIFX WAVE file with a format and a data chunk.

    Returns:
        int: The length of the trimmed file in frames.
    """
    riff_id, endian, chunks = _read_chunks(src_filepath)

    fmt_data = next((chunk_data for chunk_id, chunk_data in chunks if chunk_id == b"fmt "), None)
    audio_idx = next((idx for idx, (chunk_id, _) in enumerate(chunks) if chunk_id == b"data"), None)
    if fmt_data is None or len(fmt_data) < 14 or audio_idx is None:
        raise ValueError("Invalid WAVE file: missing format or data chunk.")

    (block_align,) = struct.unpack_from(f"{endian}H", fmt_data, 12)
    if block_align == 0:
        raise ValueError("Invalid WAVE file: zero block alignment.")

    kept_frames = min(max(0, n_frames), len(chunks[audio_idx][1]) // block_align)

    trimmed_chunks = []
    for idx, (chunk_id, chunk_data) in enumerate(chunks):
        if idx == audio_idx:
            chunk_data = chunk_data[:kept_frames * block_align]
        elif chunk_id == b"fact" and len(chunk_data) >= 4:
            # The fact chunk starts with the number of frames (required for non-PCM formats, e.g. float)
            chunk_data = struct.pack(f"{endian}I", kept_frames) + chunk_data[4:]
        trimmed_chunks.append((chunk_id, chunk_data))

    _write_riff(dest_filepath, riff_id, endian, trimmed_chunks)

    return kept_frames


def read_smpl_loop(filepath: str) -> Optional[Tuple[int, int]]:
    """Reads the first forward loop of the WAV file's sampler (smpl) chunk.

    Args:
        filepath (str): Path to the audio file.

    Returns:
        Optional[Tuple[int, int]]: (loop_start, loop_end) in samples, with loop_end exclusive like PyMusicLooper's
        (the smpl chunk's loop end is the last sample played). None if the file is not a WAV file or has no such loop.
    """
    with open(filepath, "rb") as f:
        try:
            _, endian = _read_riff_header(f)
        except ValueError:
            return None
        smpl = next((chunk for chunk in _read_chunk_headers(f, endian) if chunk.chunk_id == b"smpl"), None)
        if smpl is None or smpl.size < _SMPL_HEADER_SIZE:
            return None
        f.seek(smpl.offset)
        smpl_data = f.read(smpl.size)

    (n_loops,) = struct.unpack_from(f"{endian}I", smpl_data, _SMPL_LOOP_COUNT_OFFSET)
    for loop_idx in range(n_loops):
        loop_offset = _SMPL_HEADER_SIZE + loop_idx * _SMPL_LOOP_SIZE
        if loop_offset + _SMPL_LOOP_SIZE > len(smpl_data):
            break
        _, loop_type, loop_start, loop_end, _, _ = struct.unpack_from(f"{endian}{_SMPL_LOOP_FORMAT}", smpl_data, loop_offset)
        if loop_type == _SMPL_FORWARD_LOOP:
            return loop_start, loop_end + 1
    return None


def write_smpl_loop(filepath: str, loop_start: int, loop_end: int, rate: int):
    """Writes the loop points to the WAV file's sampler (smpl) chunk, in place, as an infinitely repeating forward loop.

    An existing sampler chunk keeps everything but its first loop (its header, other loops and sampler-specific data);
    otherwise, a new one is added.

    Args:
        filepath (str): Path to the WAV file.
        loop_start (int): Loop start in samples.
        loop_end (int): Loop end in samples (exclusive, like everywhere in PyMusicLooper).
        rate (int): Sample rate of the audio, used for the sample period of a new sampler chunk.

    Raises:
        ValueError: if the file is not a RIFF/RIFX WAVE file.
    """
    riff_id, endian, chunks = _read_chunks(filepath)

    smpl_idx = next((idx for idx, (chunk_id, _) in enumerate(chunks) if chunk_id == b"smpl"), None)
    smpl_data = chunks[smpl_idx][1] if smpl_idx is not None else b""
    has_header = len(smpl_data) >= _SMPL_HEADER_SIZE
    n_loops = struct.unpack_from(f"{endian}I", smpl_data, _SMPL_LOOP_COUNT_OFFSET)[0] if has_header else 0
    loop = struct.pack(f"{endian}{_SMPL_LOOP_FORMAT}", 0, _SMPL_FORWARD_LOOP, loop_start, loop_end - 1, 0, 0)

    if has_header and n_loops >= 1 and len(smpl_data) >= _SMPL_HEADER_SIZE + _SMPL_LOOP_SIZE:
        # Keep the first loop's cue point ID, which cue labels may refer to
        first_loop_end = _SMPL_HEADER_SIZE + _SMPL_LOOP_SIZE
        smpl_data = smpl_data[:_SMPL_HEADER_SIZE + 4] + loop[4:] + smpl_data[first_loop_end:]
    elif has_header and n_loops == 0:
        smpl_data = (
            smpl_data[:_SMPL_LOOP_COUNT_OFFSET]
            + struct.pack(f"{endian}I", 1)
            + smpl_data[_SMPL_LOOP_COUNT_OFFSET + 4:_SMPL_HEADER_SIZE]
            + loop
            + smpl_data[_SMPL_HEADER_SIZE:]
        )
    else:
        # New (or unreadable) sampler chunk
        sample_period_ns = round(1e9 / rate)
        smpl_data = struct.pack(
            f"{endian}{_SMPL_HEADER_FORMAT}", 0, 0, sample_period_ns, _DEFAULT_MIDI_UNITY_NOTE, 0, 0, 0, 1, 0
        ) + loop

    if smpl_idx is None:
        chunks.append((b"smpl", smpl_data))
    else:
        chunks[smpl_idx] = (b"smpl", smpl_data)

    _write_riff(filepath, riff_id, endian, chunks)
