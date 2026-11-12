import pytest

from scripts.taac24.timecut import query


def event(item, action, timestamp):
    return {'item_id':item,'action_type':action,'timestamp':timestamp}


def test_timecut_inputs_exclude_future_exposures_and_clicks():
    events=[event(1,0,1),event(1,1,2),event(2,0,3),event(2,1,4)]
    result,reason=query(events,{2},0.5)
    assert reason is None
    assert result['target_rid']==2
    assert result['history']==events[:2]
    assert all(e['timestamp']<result['cutoff_timestamp'] for e in result['history'])


def test_missing_past_actions_do_not_invent_or_hide_future_clicks():
    events=[event(1,None,1),event(1,0,2),event(2,0,3),event(2,1,4)]
    assert query(events,{2},0.5)[0]['history'][0]['action_type'] is None
    events[2]['action_type']=None
    assert query(events,{2},0.5)==(None,'unknown_next_click')
    events[2]['action_type']=0
    events.append(event(3,None,5))
    assert query(events,{2},0.5)[0]['target_rid']==2


def test_first_click_is_not_replaced_by_an_easier_catalog_target():
    events=[event(1,0,1),event(1,1,2),event(2,0,3),event(2,1,4),event(3,1,5)]
    assert query(events,{3},0.4)==(None,'target_outside_candidates')


def test_time_ties_do_not_leak_inputs_or_choose_an_ambiguous_target():
    events=[event(1,0,1),event(2,0,2),event(2,1,2),event(3,1,2)]
    assert query(events,{2,3},0.5)==(None,'ambiguous_first_click_time')
    events[-1]=event(3,None,2)
    assert query(events,{2,3},0.5)==(None,'unknown_next_click')
    events=events[:3]
    result,_=query(events,{2},0.5)
    assert result['history']==[events[0]]
    events[-1]['timestamp']=0
    with pytest.raises(ValueError,match='chronological'):
        query(events,{2},0.5)
