"""Lossless trimming of Ogg Vorbis files at the container level, without re-encoding the audio.

The Ogg pages up to the requested length are copied verbatim. On the last kept page, the audio packets after the
one containing the requested end sample are dropped, then its granule position (the number of samples decoded by
the end of a page) is lowered to the requested length and the page is marked as the end of the stream.
The Vorbis specification defines this as the way to end a stream on a sample that is not on a block boundary:
decoders must discard the samples past it. Keeping the trimmed samples within the last packet matches what encoders
produce, which is required by decoders that only trim the last packet (e.g. FFmpeg).
"""

import logging
import struct
from typing import Iterator, List, NamedTuple, Optional, Tuple

_PAGE_HEADER = struct.Struct("<4sBBqIIIB")
_CAPTURE_PATTERN = b"OggS"
_END_OF_STREAM_FLAG = 0x04


def _make_crc_table() -> List[int]:
    table = []
    for i in range(256):
        crc = i << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else (crc << 1)
        table.append(crc & 0xFFFFFFFF)
    return table


_CRC_TABLE = _make_crc_table()


def _crc(data: bytes) -> int:
    """Ogg page checksum: CRC-32 with polynomial 0x04C11DB7, no reflection, zero initial value and no final XOR."""
    crc = 0
    for byte in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CRC_TABLE[(crc >> 24) ^ byte]
    return crc


class _Page(NamedTuple):
    offset: int
    header_type: int
    granule_position: int
    serial_number: int
    sequence_number: int
    segment_table: bytes
    body: bytes

    @property
    def length(self) -> int:
        return _PAGE_HEADER.size + len(self.segment_table) + len(self.body)


def _read_pages(data: bytes) -> Iterator[_Page]:
    offset = 0
    while offset < len(data):
        if data[offset:offset + 4] != _CAPTURE_PATTERN or offset + _PAGE_HEADER.size > len(data):
            raise ValueError(f"Invalid Ogg page at byte offset {offset}.")
        _, version, header_type, granule, serial, sequence, _, n_segments = _PAGE_HEADER.unpack_from(data, offset)
        if version != 0:
            raise ValueError(f"Unsupported Ogg page version {version}.")
        segment_table_start = offset + _PAGE_HEADER.size
        segment_table = data[segment_table_start:segment_table_start + n_segments]
        body_start = segment_table_start + n_segments
        body = data[body_start:body_start + sum(segment_table)]
        page = _Page(offset, header_type, granule, serial, sequence, segment_table, body)
        yield page
        offset += page.length


class _ReverseBitReader:
    """Reads the bits of a Vorbis packet backwards, starting from its last bit."""

    def __init__(self, data: bytes):
        self.data = data[::-1]
        self.pos = 0

    def bits_left(self) -> int:
        return len(self.data) * 8 - self.pos

    def read(self, n_bits: int) -> int:
        value = 0
        for _ in range(n_bits):
            bit = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
            value = (value << 1) | bit
            self.pos += 1
        return value


class _VorbisPacketDuration:
    """Computes the number of samples each Vorbis audio packet decodes to.

    Only the block size flags of the modes are needed from the setup header. Since they sit at its very end,
    they are found by scanning the header backwards instead of parsing all of it (same approach as FFmpeg's
    vorbis_parser.c and liboggz).
    """

    def __init__(self, identification_header: bytes, setup_header: bytes):
        self.blocksizes = (1 << (identification_header[28] & 0x0F), 1 << (identification_header[28] >> 4))
        self.mode_blockflags = self._parse_mode_blockflags(setup_header)
        mode_bits = max(1, (len(self.mode_blockflags) - 1).bit_length())
        self.mode_mask = ((1 << mode_bits) - 1) << 1
        # The previous window flag is the bit after the mode number
        self.prev_mask = (self.mode_mask | 0x1) + 1
        self.previous_blocksize = None

    @staticmethod
    def _parse_mode_blockflags(setup_header: bytes) -> List[int]:
        reader = _ReverseBitReader(setup_header)

        framing_bit_pos = None
        while reader.bits_left() > 97:
            if reader.read(1):
                framing_bit_pos = reader.pos
                break
        if framing_bit_pos is None:
            raise ValueError("Invalid Vorbis setup header: no framing bit.")

        # Each mode is 41 bits: blockflag (1), windowtype (16, always 0), transformtype (16, always 0), mapping (8).
        # Walk backwards over them while they look valid; the mode count stored just before them confirms a match.
        mode_count = 0
        confirmed_mode_count = 0
        while reader.bits_left() >= 97:
            if reader.read(8) > 63 or reader.read(16) or reader.read(16):
                break
            reader.read(1)
            mode_count += 1
            if mode_count > 64:
                break
            mode_header_pos = reader.pos
            if reader.read(6) + 1 == mode_count:
                confirmed_mode_count = mode_count
            reader.pos = mode_header_pos
        if not confirmed_mode_count:
            raise ValueError("Invalid Vorbis setup header: no modes found.")

        reader.pos = framing_bit_pos
        blockflags = [0] * confirmed_mode_count
        for i in reversed(range(confirmed_mode_count)):
            reader.read(40)
            blockflags[i] = reader.read(1)
        return blockflags

    def __call__(self, first_byte: int) -> int:
        mode = 0 if len(self.mode_blockflags) == 1 else (first_byte & self.mode_mask) >> 1
        if mode >= len(self.mode_blockflags):
            raise ValueError("Invalid Vorbis audio packet.")
        blockflag = self.mode_blockflags[mode]
        current_blocksize = self.blocksizes[blockflag]
        if blockflag:
            previous_blocksize = self.blocksizes[1 if first_byte & self.prev_mask else 0]
        else:
            previous_blocksize = self.previous_blocksize
        is_first_packet = self.previous_blocksize is None
        self.previous_blocksize = current_blocksize
        # The first audio packet only primes the decoder and returns no samples
        return 0 if is_first_packet else (previous_blocksize + current_blocksize) // 4


def _find_packet_cut(pages: List[_Page], n_samples: int) -> Optional[Tuple[int, int]]:
    """Finds where to cut the stream so that the last kept packet contains sample `n_samples`.

    Returns:
        Optional[Tuple[int, int]]: (index of the last page to keep, number of its segments to keep),
        or None if the requested length is at or past the end of the stream.
    """
    header_packets = []
    packet_duration = None
    packet_data = b""
    previous_granule = 0

    for page_idx, page in enumerate(pages):
        # (duration, number of segments up to the packet's end) of each packet completed on this page
        completed_packets = []
        body_pos = 0
        for segment_idx, lacing_value in enumerate(page.segment_table):
            packet_data += page.body[body_pos:body_pos + lacing_value]
            body_pos += lacing_value
            if lacing_value == 255:
                continue
            if len(header_packets) < 3:
                header_packets.append(packet_data)
                if len(header_packets) == 3:
                    packet_duration = _VorbisPacketDuration(header_packets[0], header_packets[2])
            elif packet_data:
                completed_packets.append((packet_duration(packet_data[0]), segment_idx + 1))
            packet_data = b""
        # Only the first byte of audio packets is needed when a packet continues on the next page
        if len(header_packets) == 3 and len(packet_data) > 1:
            packet_data = packet_data[:1]

        if not completed_packets or page.granule_position == -1:
            continue

        if page.granule_position >= n_samples:
            page_end = previous_granule + sum(duration for duration, _ in completed_packets)
            # The final page of a stream may already end it before the end of its last packet
            is_final_page = page.header_type & _END_OF_STREAM_FLAG or page_idx == len(pages) - 1
            if page_end != page.granule_position and not (is_final_page and page_end > page.granule_position):
                raise ValueError("Vorbis packet durations do not match the page granule positions.")
            packet_end = previous_granule
            for duration, n_segments in completed_packets:
                packet_end += duration
                if packet_end >= n_samples:
                    return page_idx, n_segments

        previous_granule = page.granule_position

    return None


def _find_page_cut(pages: List[_Page], n_samples: int) -> Optional[Tuple[int, int]]:
    """Same as `_find_packet_cut`, but keeps all the packets completed on the last page."""
    # The first page whose completed packets reach the requested length
    # (header pages have a granule position of 0, and -1 means no packet ends on the page)
    for page_idx, page in enumerate(pages):
        if page.granule_position >= n_samples:
            # Drop the segments of a trailing packet that continues on the next page (a run of 255 lacing values),
            # since that packet would be incomplete. At least one packet ends on this page (granule position != -1),
            # so the run belongs entirely to a packet that starts on this page.
            return page_idx, len(page.segment_table.rstrip(b"\xff"))
    return None


def _build_page(page: _Page, header_type: int, granule_position: int, segment_table: bytes, body: bytes) -> bytes:
    def pack(crc: int) -> bytes:
        return _PAGE_HEADER.pack(
            _CAPTURE_PATTERN, 0, header_type, granule_position, page.serial_number,
            page.sequence_number, crc, len(segment_table),
        ) + segment_table + body

    return pack(_crc(pack(0)))


def trim_vorbis(src_filepath: str, dest_filepath: str, n_samples: int) -> int:
    """Writes a copy of an Ogg Vorbis file that ends after `n_samples` samples, without re-encoding the audio.

    Args:
        src_filepath (str): Path to the source Ogg Vorbis file.
        dest_filepath (str): Path of the trimmed file to write.
        n_samples (int): Number of samples (per channel) to keep from the start of the stream.

    Raises:
        ValueError: if the file is not a single-stream Ogg Vorbis file.

    Returns:
        int: The length of the trimmed file in samples.
    """
    with open(src_filepath, "rb") as f:
        data = f.read()

    pages = list(_read_pages(data))
    if not pages or not pages[0].body.startswith(b"\x01vorbis"):
        raise ValueError("Not an Ogg Vorbis file.")
    if any(page.serial_number != pages[0].serial_number for page in pages):
        raise ValueError("Chained or multiplexed Ogg files are not supported.")

    try:
        cut = _find_packet_cut(pages, max(1, n_samples))
    except (ValueError, IndexError) as e:
        # Fall back to keeping all the packets of the last page, which is valid but
        # relies on decoders trimming more than the last packet
        logging.warning(f"Could not determine the Vorbis packet durations ({e}); trimming at the page level.")
        cut = _find_page_cut(pages, max(1, n_samples))

    if cut is None:
        # Requested length is at or past the end of the stream: keep everything
        with open(dest_filepath, "wb") as f:
            f.write(data)
        return max(0, pages[-1].granule_position)

    page_idx, n_segments = cut
    page = pages[page_idx]
    segment_table = page.segment_table[:n_segments]
    body = page.body[:sum(segment_table)]

    last_page = _build_page(
        page,
        header_type=page.header_type | _END_OF_STREAM_FLAG,
        granule_position=n_samples,
        segment_table=segment_table,
        body=body,
    )

    with open(dest_filepath, "wb") as f:
        f.write(data[:page.offset])
        f.write(last_page)

    return n_samples
