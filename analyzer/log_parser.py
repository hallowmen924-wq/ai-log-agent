import re


def parse_logs(raw_logs):

    results = []

    in_pattern = r"in_data = \[(.*?)\]"
    out_pattern = r"out_data = \[(.*?)\]"

    in_matches = re.findall(in_pattern, raw_logs, re.DOTALL)
    out_matches = re.findall(out_pattern, raw_logs, re.DOTALL)

    for i in range(min(len(in_matches), len(out_matches))):

        in_data = in_matches[i]
        out_data = out_matches[i]

        product_code = extract_product_code(in_data)

        results.append(
            {"product": product_code, "in_data": in_data, "out_data": out_data}
        )

    return results


def parse_logs_fast(raw_logs):

    results = []

    current_in = None

    for line in raw_logs.splitlines():

        if "in_data =" in line:
            current_in = line

        elif "out_data =" in line and current_in:

            in_data = current_in.split("in_data = [")[-1].rstrip("]")
            out_data = line.split("out_data = [")[-1].rstrip("]")

            product = extract_product_code(in_data)

            results.append(
                {"product": product, "in_data": in_data, "out_data": out_data}
            )

            current_in = None

    return results


def extract_product_code(in_data):

    if "Online_C9_ASS" in in_data:
        return "C9"
    elif "Online_C6_ASS" in in_data:
        return "C6"
    elif "Online_C11_ASS" in in_data:
        return "C11"
    elif "Online_C12_ASS" in in_data:
        return "C12"
    else:
        return "UNKNOWN"
