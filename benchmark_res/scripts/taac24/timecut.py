"""Proposed, opt-in local next-click protocol; never the original leaderboard.

The decision point is the timestamp at 80% of the observed sequence. Only
strictly earlier events are inputs. The first subsequent click is the target;
unknown actions that could precede/tie it, or tied different ads, are excluded.
This needs no reconstruction of which earlier exposure caused a click.
"""
from collections import Counter
import json
from pathlib import Path
import sys


def query(events, candidates, cut_fraction=0.8):
    if not 0 < cut_fraction < 1:
        raise ValueError('cut_fraction must lie strictly between zero and one')
    if any(e['action_type'] not in (0, 1, None) for e in events):
        raise ValueError('Unsupported action')
    if any(a['timestamp'] > b['timestamp'] for a,b in zip(events,events[1:])):
        raise ValueError('Sequence must be chronological')
    if not events:
        return None,'empty_sequence'
    cutoff=events[int(len(events)*cut_fraction)]['timestamp']
    history=[e for e in events if e['timestamp'] < cutoff]
    if not history:
        return None,'empty_history'
    future=[e for e in events if e['timestamp'] >= cutoff]
    click=next((e for e in future if e['action_type']==1),None)
    if click is None:
        return None,'no_post_cut_click'
    first_time=click['timestamp']
    if any(e['action_type'] is None and e['timestamp'] <= first_time for e in future):
        return None,'unknown_next_click'
    tied={e['item_id'] for e in future if e['timestamp']==first_time and e['action_type']==1}
    if len(tied)!=1:
        return None,'ambiguous_first_click_time'
    if click['item_id'] not in candidates:
        return None,'target_outside_candidates'
    return {'history':history,'target_rid':click['item_id'],'cutoff_timestamp':cutoff},None


def audit(root):
    """Read-only coverage audit: do not write a new dataset or start a study."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from recevobench.taac.data import partition
    root=Path(root)
    meta=json.loads((root/'prepared/public/metadata.json').read_text())
    candidates=set(int(i) for i in np.load(root/'prepared/public/candidate_rids.npy') if i)
    counts=Counter()
    for path in sorted((root/'assets/raw/seq').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            ids=batch.column('user_id').to_pylist()
            parts=[partition(uid,meta['split_salt'],meta['fractions']) for uid in ids]
            selected=batch.filter(pa.array([p!='train' for p in parts]))
            for row,part in zip(selected.to_pylist(),[p for p in parts if p!='train']):
                result,reason=query(row['seq'],candidates)
                counts[part+'_users']+=1
                counts[part+'_queries' if result else part+'_excluded_'+reason]+=1
    result={'status':'proposal_only_not_adopted','schema':'taac-timecut-nextclick-v1',
            'counts':dict(counts),'old_counts':meta['counts'],
            'scope':'Fixed-time local next-click prediction, not official leaderboard',
            'history':'Strictly before timestamp at 80 percent of sequence',
            'target':'First post-cut click, excluding uncertain/tied targets',
            'unknown_history':'Preserved; only unknown actions in target-identification window cause exclusion',
            'candidate_count':660000,'training_unchanged':True}
    (root/'timecut-proposal-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    audit(sys.argv[1])
