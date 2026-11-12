from pathlib import Path
import tempfile
import pytest

from . import launch
from amazon_family_compare.core import read,save
from amazon_family_compare.prepare import sha
from amazon_family_recovery.controller import RecoveryController


def test_three_full_budget_gru_runs_replace_only_the_cancelled_queue():
    plan=read(launch.ROOT/'plan.json')
    assert len(plan['jobs'])==3 and plan['total_budget_gpu_seconds']==129600
    assert plan['method']=='mlestar' and plan['model_family']=='gru4rec'
    assert plan['cancelled_search_gpu_seconds_retained_separately']>0
    assert len({j['root'] for j in plan['jobs']})==3
    assert read(launch.PREVIOUS/'queues/1/status.json')['status']=='stopped_by_user'
    old=read(launch.PREVIOUS/'plan.json')
    q=next(q for q in old['queues'] if q['method']=='mlestar')
    assert q['wait_for_hstu'] and len(q['jobs'])==3


@pytest.mark.parametrize('number',[1,2,3])
def test_each_repeat_matches_the_completed_gru_protocol(number,tmp_path):
    c=read(launch.ROOT/'runs'/f'repeat-{number:02d}'/'config.json')
    original=read(launch.COMPARISON/'runs/gru4rec-mlestar/config.json')
    for key in ['data_dir','worker_python','controller_python','gpus','cpu_cores',
                'search_budget_gpu_seconds','trial_floor_reference_seconds','trial_timeout_seconds',
                'minimum_trial_seconds','final_timeout_seconds','llm_model','llm_effort',
                'llm_timeout','llm_attempts','single_model_search','global_llm_concurrency',
                'llm_slot_directory']:
        assert c[key]==original[key],key
    for key in ['datasets','cold_start','preprocessing_seed','training_seed','evaluation_seed',
                'validation_users_per_dataset','negative_candidates','training_environment_defaults']:
        assert c['alignment'][key]==original['alignment'][key],key
    assert c['model_family']=='gru4rec'
    assert sha(c['baseline']['source'])==original['baseline']['sha256']
    assert Path(c['baseline']['source']).read_bytes()==Path(original['baseline']['source']).read_bytes()
    assert not (Path(c['root'])/'mlestar').exists()
    assert c['replication']['additional_repetition']==number
    assert c['replication']['fresh_baseline_weights']
    assert not c['replication']['cross_run_response_or_memory_reuse']
    probe=RecoveryController({**c,'root':str(tmp_path)},'mlestar')
    assert probe.ledger.available()==43200
    assert probe.best is None
    assert not list((probe.root/'calls').glob('*'))
    assert probe.environment(4)['GAGC_MODEL_FAMILY']=='gru4rec'


def test_pending_nextitnet_sources_and_configurations_remain_frozen():
    launch.verify(read(launch.PREVIOUS/'frozen.json')['files'])


def test_cannot_restart_an_existing_launcher(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    save(tmp_path/'launcher.json',{'pid':123})
    with pytest.raises(RuntimeError,match='Already launched'):launch.start()
    assert read(tmp_path/'launcher.json')=={'pid':123}


def test_worker_will_not_reset_existing_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    monkeypatch.setattr(launch,'verify',lambda _:None)
    folder=tmp_path/'runs/repeat-01';config=folder/'config.json'
    save(config,{'root':str(folder)})
    save(tmp_path/'plan.json',{'jobs':[{'id':'repeat-01','root':str(folder),
        'config':str(config),'additional_repetition':1}]})
    save(tmp_path/'frozen.json',{'files':{}})
    initial={'total_gpu_seconds':43200,'spent_gpu_seconds':321,'reservations':{},'settled':{}}
    save(folder/'mlestar/budget.json',initial)
    with pytest.raises(RuntimeError,match='never be reset'):launch.worker('repeat-01')
    assert read(folder/'mlestar/budget.json')==initial
