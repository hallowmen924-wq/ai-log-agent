import pathlib, tokenize
p = pathlib.Path(r"C:\work\backend\app_main.py")
with open(p, 'rb') as f:
    enc, _ = tokenize.detect_encoding(f.readline)
text = p.read_text(encoding=enc)
lines = text.splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith('def _query_requests_loan_calculator('))
end = next(i for i, l in enumerate(lines[start+1:], start+1) if l.startswith('def _extract_calculator_input_hints('))
new_block = [
    'def _query_requests_loan_calculator(query: str) -> bool:',
    '    compact_query = _compact_search_text(query)',
    '    if not compact_query:',
    '        return False',
    '    calculator_markers = [',
    '        "원리금계산기", "원리금", "상환계산기", "대출계산기", "월상환",',
    '        "amortization", "loancalculator", "repaymentcalculator", "calculator",',
    '    ]',
    '    if any(marker in compact_query for marker in calculator_markers):',
    '        return True',
    '',
    '    # Follow-up slot answering: users often send only principal/rate/term/repayment.',
    '    input_slot_markers = [',
    '        "원금", "대출금", "대출금액", "연이율", "금리", "기간", "개월", "상환방식",',
    '        "원리금균등", "원금균등", "만기일시",',
    '        "principal", "interest", "rate", "term", "month", "repayment",',
    '    ]',
    '    slot_hits = sum(1 for marker in input_slot_markers if marker in compact_query)',
    '    hints = _extract_calculator_input_hints(query)',
    '    structured_hits = sum(',
    '        1 for key in ["principal_amount", "annual_rate", "term_months", "repayment_type"]',
    '        if hints.get(key) not in (None, "", 0)',
    '    )',
    '    return slot_hits >= 2 or structured_hits >= 2',
    '',
]
out = '\n'.join(lines[:start] + new_block + lines[end:]) + '\n'
p.write_text(out, encoding=enc)
print('patched', start + 1, end + 1, enc)
