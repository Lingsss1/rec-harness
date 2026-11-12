"""Task-side design checks; domain facts deliberately stay outside the agent core."""
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


def validate_task(directory, task):
    directory = Path(directory)
    protocol = json.loads((directory / task['protocol_ref']).read_text())
    assert protocol['sample']['label'] == 'clk'
    assert {'clk', 'nonclk'} <= set(protocol['sample']['forbidden_input_fields'])
    assert protocol['split']['train_day_indices'] == [1, 2, 3, 4, 5, 6]
    assert protocol['split']['validation_day_indices'] == [7]
    assert protocol['split']['test_day_indices'] == [8]
    assert protocol['split']['timezone'] == 'Asia/Shanghai'
    assert protocol['history']['available_fields'] == ['category', 'brand', 'behavior_type', 'event_time']
    assert protocol['feature_fitting']['label_derived_inputs'] is False
    assert task['knowledge']['personal']['enabled'] is False
    assert task['knowledge']['personal']['source_ref'] is None
    assert task['knowledge']['personal']['writer'] == 'human_only'
    assert task['knowledge']['personal']['create_empty_personal_file'] is False
    assert task['research_flow']['novelty_score'] is None
    assert task['research_flow']['routine_human_approval'] is False

    industry = directory / 'industry'
    registry = json.loads((industry / 'sources.json').read_text())
    assert registry['selection']['status'] == 'user_confirmed'
    assert registry['selection']['global_rank_exists'] is False
    sources = {s['id']: s for s in registry['sources']}
    cards = [json.loads(line) for line in (industry / 'cards.jsonl').read_text().splitlines() if line.strip()]
    expected = {'CRAFT': ('academic', 1), 'UniDAC': ('academic', 2), 'UniTokenFormer': ('academic', 3), 'QueryFormer': ('industry', 1), 'UniDot': ('industry', 2)}
    assert len(cards) == len(expected) == len({c['id'] for c in cards})
    assert {c['scheme'] for c in cards} == set(expected)
    schema = json.loads((industry / 'card.schema.json').read_text())
    Draft202012Validator.check_schema(schema)
    for card in cards:
        Draft202012Validator(schema).validate(card)
        assert (card['placement']['track'], card['placement']['rank']) == expected[card['scheme']]
        assert set(card['source_refs']) <= set(sources)
        assert card['placement']['evidence_ref'] in card['source_refs']
        assert sources[card['placement']['evidence_ref']]['content_read'] is True

    overview = sources['taac2026-organizer-overview']
    cached = (industry / overview['local_cache']).resolve()
    cache_verified = False
    if cached.exists():
        assert cached.stat().st_size == overview['bytes']
        assert hashlib.sha256(cached.read_bytes()).hexdigest() == overview['sha256']
        cache_verified = True
    return {'industry_cards': len(cards), 'placements': {c['scheme']: c['placement'] for c in cards}, 'personal_enabled': False, 'cached_organizer_source_verified': cache_verified, 'data_audit_completed': False, 'training_started': False}
