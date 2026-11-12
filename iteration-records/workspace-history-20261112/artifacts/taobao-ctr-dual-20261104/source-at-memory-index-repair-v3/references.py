"""Bind observed identifiers in either compact or descriptive evidence citations."""
import re


def observed_trial_refs(references, known_ids):
    if not isinstance(references,list) or any(not isinstance(ref,str) for ref in references):
        raise ValueError('Evidence references must be strings')
    matched=set()
    for trial_id in known_ids:
        pattern=r'(?<![A-Za-z0-9_-])'+re.escape(trial_id)+r'(?![A-Za-z0-9_-])'
        if any(re.search(pattern,ref) for ref in references):matched.add(trial_id)
    if not matched:
        raise ValueError('Cite a supplied observed trial ID, either verbatim or as a complete token in a descriptive reference')
    return sorted(matched)
