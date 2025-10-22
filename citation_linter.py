#!/usr/bin/env python3
"""citation_linter.py
Simple linter to enforce Article 175 citation presence in GPT175 answers.

Usage:
  python citation_linter.py --file answers.txt
  # or pipe:
  cat answers.txt | python citation_linter.py

Rules:
- Every bullet or numbered line that contains a number, unit, or the words MUST/SHALL/PROHIBITED/etc.
  must end with a parenthetical or inline citation containing '§175.' and/or 'p. '.
- Lines that explicitly say 'Not specified in Article 175' are exempt.
"""
import argparse, sys, re

REQ_PAT = re.compile(r'(must|shall|prohibit|limit|require|\b\d+\.?\d*\s*(mGy|mSv|Gy|cm|min|day|year|h|mm)|\bSRDL\b|\bAKR\b|\bKAP\b)', re.IGNORECASE)
CITE_PAT = re.compile(r'(§\s*175\.[0-9A-Za-z\(\)\.\-]+|p\.\s*\d+)', re.IGNORECASE)

def lint_text(text: str):
    errors = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip()
        if not line.strip():
            continue
        # consider bullets or numbered items higher risk
        is_listy = bool(re.match(r'\s*([-*\u2022]|\d+\.|\[\s*\])', line)) or REQ_PAT.search(line)
        if 'Not specified in Article 175' in line:
            continue
        if is_listy and not CITE_PAT.search(line):
            errors.append((i, line))
    return errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', type=str, default='-')
    args = ap.parse_args()
    data = sys.stdin.read() if args.file == '-' else open(args.file, 'r', encoding='utf-8').read()
    errs = lint_text(data)
    if not errs:
        print('OK: All regulatory lines contain citations.')
        return 0
    print('Found lines missing citations:')
    for i, line in errs:
        print(f'  L{i}: {line}')
    return 1

if __name__ == '__main__':
    raise SystemExit(main())
