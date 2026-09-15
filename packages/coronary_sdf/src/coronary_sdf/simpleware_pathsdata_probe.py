"""Inspect the raw protobuf wire structure of SIP PathsData.bin."""

import struct
import sys
import zipfile
from collections import Counter, defaultdict


def read_varint(data, offset):
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            raise ValueError("invalid varint")


def fields(data):
    result = []
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        number, wire = key >> 3, key & 7
        if not number:
            raise ValueError("zero field")
        if wire == 0:
            value, offset = read_varint(data, offset)
        elif wire == 1:
            value = data[offset:offset + 8]
            offset += 8
        elif wire == 2:
            length, offset = read_varint(data, offset)
            value = data[offset:offset + length]
            offset += length
        elif wire == 5:
            value = data[offset:offset + 4]
            offset += 4
        else:
            raise ValueError("unsupported wire {}".format(wire))
        if offset > len(data):
            raise ValueError("truncated value")
        result.append((number, wire, value))
    return result


def printable(value):
    if not isinstance(value, bytes):
        return str(value)
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return text if text and all(char.isprintable() for char in text) else None


def describe(data, depth=0, max_depth=4, occurrence_limit=3):
    parsed = fields(data)
    grouped = defaultdict(list)
    for number, wire, value in parsed:
        grouped[(number, wire)].append(value)
    indent = "  " * depth
    print("{}message bytes={} fields={}".format(indent, len(data), len(parsed)))
    for (number, wire), values in sorted(grouped.items()):
        lengths = Counter(len(value) for value in values) if wire in (1, 2, 5) else None
        print("{}  field {} wire {} count {} lengths {}".format(
            indent, number, wire, len(values), dict(lengths or {})
        ))
        for value in values[:occurrence_limit]:
            text = printable(value)
            if text is not None and len(text) <= 100:
                print("{}    text={!r}".format(indent, text))
            elif wire == 1:
                print("{}    fixed64 double={!r}".format(
                    indent, struct.unpack("<d", value)[0]
                ))
            elif wire == 5:
                print("{}    fixed32 float={!r}".format(
                    indent, struct.unpack("<f", value)[0]
                ))
            elif wire == 2 and depth < max_depth:
                try:
                    describe(value, depth + 1, max_depth, occurrence_limit)
                except (ValueError, IndexError):
                    print("{}    bytes={}...".format(indent, value[:24].hex()))
    return parsed


def values(message, number, wire=None):
    return [
        value for field_number, field_wire, value in fields(message)
        if field_number == number and (wire is None or field_wire == wire)
    ]


def nested_text(message, number):
    nested = values(message, number, 2)[0]
    return values(nested, 1, 2)[0].decode("utf-8")


def summary(data):
    container = values(data, 1, 2)[0]
    node_messages = values(container, 1, 2)
    spline_messages = values(container, 2, 2)
    degree_counts = Counter(len(values(node, 7, 2)) for node in node_messages)
    spline_points = {
        nested_text(spline, 1): len(values(spline, 8, 2))
        for spline in spline_messages
    }
    terminal_raw_counts = []
    for node in node_messages:
        attached = values(node, 7, 2)
        if len(attached) == 1:
            attached_uuid = values(attached[0], 1, 2)[0].decode("utf-8")
            terminal_raw_counts.append(spline_points[attached_uuid])
    network_messages = values(container, 5, 2)
    print("SUMMARY nodes={} splines={} degrees={} terminal_raw_points={}..{} networks={}".format(
        len(node_messages),
        len(spline_messages),
        dict(sorted(degree_counts.items())),
        min(terminal_raw_counts),
        max(terminal_raw_counts),
        [values(network, 2, 2)[0].decode("utf-8") for network in network_messages],
    ))


def main(path):
    with zipfile.ZipFile(path) as archive:
        data = archive.read("PathsData.bin")
    summary(data)
    describe(data)


if __name__ == "__main__":
    main(sys.argv[1])
