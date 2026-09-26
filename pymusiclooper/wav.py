"""Lossless trimming of WAV files at the chunk level, without rewriting the audio.

Every chunk of the source file is copied verbatim, so everything stored besides the audio is kept as-is:
the format chunk, tags (LIST/INFO and ID3 chunks), sampler loops (smpl), cue points, broadcast extension (bext), etc.
Only the audio data chunk is shortened, and the sample count of the fact chunk (if any) and the RIFF size are updated.
"""

import struct
from typing import List, NamedTuple


class _Chunk(NamedTuple):
    chunk_id: bytes
    data: bytes


def _read_chunks(data: bytes, endian: str) -> List[_Chunk]:
    chunks = []
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        (size,) = struct.unpack_from(f"{endian}I", data, offset + 4)
        # Files written by interrupted or streaming encoders can leave the data size as a placeholder:
        # 0 (then the data runs to the end of the file), or a size past the end of the file
        if chunk_id == b"data" and size == 0:
            size = len(data) - offset - 8
        chunk_data = data[offset + 8:offset + 8 + size]
        chunks.append(_Chunk(chunk_id, chunk_data))
        # Chunks are padded to an even size
        offset += 8 + size + (size & 1)
    return chunks


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
    with open(src_filepath, "rb") as f:
        data = f.read()

    if data[:4] not in (b"RIFF", b"RIFX") or data[8:12] != b"WAVE":
        raise ValueError("Not a RIFF WAVE file.")
    endian = "<" if data[:4] == b"RIFF" else ">"

    chunks = _read_chunks(data, endian)
    fmt_chunk = next((chunk for chunk in chunks if chunk.chunk_id == b"fmt "), None)
    if fmt_chunk is None or len(fmt_chunk.data) < 14 or not any(chunk.chunk_id == b"data" for chunk in chunks):
        raise ValueError("Invalid WAVE file: missing format or data chunk.")

    (block_align,) = struct.unpack_from(f"{endian}H", fmt_chunk.data, 12)
    if block_align == 0:
        raise ValueError("Invalid WAVE file: zero block alignment.")

    audio_chunk = next(chunk for chunk in chunks if chunk.chunk_id == b"data")
    kept_frames = min(max(0, n_frames), len(audio_chunk.data) // block_align)

    body = []
    for chunk in chunks:
        chunk_data = chunk.data
        if chunk is audio_chunk:
            chunk_data = chunk_data[:kept_frames * block_align]
        elif chunk.chunk_id == b"fact" and len(chunk_data) >= 4:
            # The fact chunk starts with the number of frames (required for non-PCM formats, e.g. float)
            chunk_data = struct.pack(f"{endian}I", kept_frames) + chunk_data[4:]
        body += [chunk.chunk_id, struct.pack(f"{endian}I", len(chunk_data)), chunk_data]
        if len(chunk_data) & 1:
            body.append(b"\x00")
    body = b"".join(body)

    with open(dest_filepath, "wb") as f:
        f.write(data[:4] + struct.pack(f"{endian}I", 4 + len(body)) + b"WAVE" + body)

    return kept_frames
