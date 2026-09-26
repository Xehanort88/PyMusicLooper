"""Restores the metadata of a FLAC file onto a re-encoded copy of its audio.

A FLAC file is a list of metadata blocks followed by the audio frames. The metadata blocks of the source file
are copied verbatim (tags, pictures such as cover art, application blocks, padding), except for the ones that
describe the audio itself: the stream info (length, MD5 checksum) is taken from the re-encoded copy, and the seek table
and cue sheet are dropped, since they point to positions in the source audio that no longer exist in the copy.
"""

from typing import List, Tuple

_STREAMINFO = 0
_SEEKTABLE = 3
_CUESHEET = 5
_LAST_BLOCK_FLAG = 0x80
_ID3V1_LENGTH = 128


def _id3v2_length(data: bytes) -> int:
    """Returns the length of the ID3v2 tag at the start of the file (some taggers add one to FLAC files), or 0."""
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    # The tag size is stored as a 28-bit "synchsafe" integer (7 bits per byte)
    size = 0
    for byte in data[6:10]:
        size = (size << 7) | (byte & 0x7F)
    has_footer = data[5] & 0x10
    return 10 + size + (10 if has_footer else 0)


def _read_metadata(data: bytes) -> Tuple[int, List[Tuple[int, bytes]], int]:
    """Parses the metadata blocks of a FLAC file.

    Returns:
        Tuple[int, List[Tuple[int, bytes]], int]: (offset of the "fLaC" marker, [(block type, block data)],
        offset of the first audio frame)
    """
    marker_offset = _id3v2_length(data)
    if data[marker_offset:marker_offset + 4] != b"fLaC":
        raise ValueError("Not a FLAC file.")

    blocks = []
    offset = marker_offset + 4
    while True:
        if offset + 4 > len(data):
            raise ValueError("Invalid FLAC file: truncated metadata.")
        header = data[offset]
        length = int.from_bytes(data[offset + 1:offset + 4], "big")
        block_data = data[offset + 4:offset + 4 + length]
        if len(block_data) != length:
            raise ValueError("Invalid FLAC file: truncated metadata.")
        blocks.append((header & 0x7F, block_data))
        offset += 4 + length
        if header & _LAST_BLOCK_FLAG:
            return marker_offset, blocks, offset


def copy_flac_metadata(src_filepath: str, dest_filepath: str):
    """Replaces the metadata of the FLAC file `dest_filepath` with the metadata of `src_filepath`,
    except for the blocks describing the audio itself (see the module docstring).
    ID3 tags that taggers sometimes add before or after the FLAC data are copied too.

    Args:
        src_filepath (str): Path to the FLAC file to copy the metadata from.
        dest_filepath (str): Path to the FLAC file to write the metadata to, in place.

    Raises:
        ValueError: if either file is not a valid FLAC file.
    """
    with open(src_filepath, "rb") as f:
        src = f.read()
    with open(dest_filepath, "rb") as f:
        dest = f.read()

    src_marker_offset, src_blocks, _ = _read_metadata(src)
    _, dest_blocks, dest_audio_offset = _read_metadata(dest)

    dest_streaminfo = [block_data for block_type, block_data in dest_blocks if block_type == _STREAMINFO]
    if not dest_streaminfo:
        raise ValueError("Invalid FLAC file: missing stream info.")

    blocks = [(_STREAMINFO, dest_streaminfo[0])] + [
        (block_type, block_data)
        for block_type, block_data in src_blocks
        if block_type not in (_STREAMINFO, _SEEKTABLE, _CUESHEET)
    ]

    output = [src[:src_marker_offset], b"fLaC"]
    for idx, (block_type, block_data) in enumerate(blocks):
        last_block_flag = _LAST_BLOCK_FLAG if idx == len(blocks) - 1 else 0
        output += [bytes([last_block_flag | block_type]), len(block_data).to_bytes(3, "big"), block_data]
    output.append(dest[dest_audio_offset:])
    if src[-_ID3V1_LENGTH:-_ID3V1_LENGTH + 3] == b"TAG":
        output.append(src[-_ID3V1_LENGTH:])

    with open(dest_filepath, "wb") as f:
        f.write(b"".join(output))
