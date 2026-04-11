import re

def parse_fields(data):

    fields = {}

    matches = re.findall(r'([A-Z]\d{4})([^A-Z]+)', data)

    for code, value in matches:
        fields[code] = value.strip()

    return fields