"""Portable-only dataset selection; keep archived experiment source unchanged."""
import importlib
import os
import re

if os.environ.get('REC_HARNESS_PORTABLE') == '1' and os.environ.get('REC_HARNESS_DATASETS'):
    names = [s.strip() for s in os.environ['REC_HARNESS_DATASETS'].split(',') if s.strip()]
    if not names or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', n) or n in {'.', '..'} for n in names):
        raise SystemExit('Invalid REC_HARNESS_DATASETS: use comma-separated dataset identifiers')
    os.environ['GAGC_DATASETS'] = ','.join(names)
    for package in ['amazon_family_compare', 'amazon_aligned']:
        prepare = importlib.import_module(package+'.prepare')
        prepare.DATASETS[:] = names
        core = importlib.import_module(package+'.core')
        core.TASK = core.TASK.replace(
            'Matched RecHarness Amazon task: four datasets Movies_and_TV,\nIndustrial_and_Scientific, Electronics, CDs_and_Vinyl, macro-average validation\nHR@10.',
            'Matched recommendation task: datasets '+', '.join(names)+', macro-average validation HR@10.')
