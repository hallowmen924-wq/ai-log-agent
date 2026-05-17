import re


def split_fields(data):
    return re.findall(r"[A-Z]\d{4}[^A-Z]*", data)


def parse_field(field):
    code = field[:5]
    value = field[5:].strip()
    return code, value
