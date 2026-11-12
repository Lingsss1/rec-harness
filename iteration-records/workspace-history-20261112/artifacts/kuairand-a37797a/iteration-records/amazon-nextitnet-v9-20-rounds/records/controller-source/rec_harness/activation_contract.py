"""Host validation and bounded public projection of untrusted probe reports."""
import math
from rec_harness.core import ContractError


def exact(value,keys):
    if not isinstance(value,dict) or set(value)!=set(keys):raise ContractError('probe fields differ from frozen contract')


def nonnegative(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ContractError('finite nonnegative probe value required')
    return value


def same(left,right):
    if not math.isclose(left,right,rel_tol=1e-7,abs_tol=1e-12):raise ContractError('inconsistent probe aggregate')


def public_measurement(report,expected):
    """Expected identities/counts come from host inputs, never from the report."""
    exact(report,{'schema','blocks','scope','limitations','source_hash','probe_hash','initialization','settings','unchanged_state'}|({'history_hash'} if 'history_hash' in expected else set()))
    if report['schema']!='masked-residual-probe-v1' or report['unchanged_state'] is not True:
        raise ContractError('probe schema or model-state claim invalid')
    for key in ['source_hash','probe_hash','settings','initialization']:
        if report[key]!=expected[key]:raise ContractError('probe input identity differs: '+key)
    if 'history_hash' in expected and report['history_hash']!=expected['history_hash']:
        raise ContractError('probe history input identity differs')
    blocks=report['blocks'];count=expected['settings']['NUM_BLOCKS'];slices=expected['slice_counts']
    if type(count) is not int or not 1<=count<=256 or not isinstance(blocks,list) or len(blocks)!=count:
        raise ContractError('probe block coverage differs')
    if set(slices)-{'all','short','long'} or 'all' not in slices or any(type(n) is not int or n<=0 for n in slices.values()):
        raise ContractError('invalid host slice protocol')
    if sum(n for k,n in slices.items() if k!='all')!=slices['all']:raise ContractError('host slices do not partition histories')
    if type(expected['valid_positions']) is not int or expected['valid_positions']<slices['all']:
        raise ContractError('invalid host valid-position count')
    measurements={};references={}
    for index,row in enumerate(blocks):
        exact(row,{'block_index','activation_rms','residual_rms','zero_fraction','valid_positions','bypass'})
        if type(row['block_index']) is not int or row['block_index']!=index:raise ContractError('probe block ordering differs')
        if type(row['valid_positions']) is not int or row['valid_positions']!=expected['valid_positions']:
            raise ContractError('masked activation count differs')
        prefix='block_'+str(index)+'.'
        for key in ['activation_rms','residual_rms','zero_fraction']:
            measurements[prefix+key]=nonnegative(row[key])
        if row['zero_fraction']>1:raise ContractError('invalid zero fraction')
        exact(row['bypass'],slices)
        for name,n in slices.items():
            effect=row['bypass'][name]
            exact(effect,{'histories','tail_change_rms','tail_reference_rms','relative_tail_change'})
            if type(effect['histories']) is not int or effect['histories']!=n:raise ContractError('probe slice population differs')
            change=nonnegative(effect['tail_change_rms']);reference=nonnegative(effect['tail_reference_rms'])
            if name in references:same(reference,references[name])
            else:references[name]=reference
            ratio=effect['relative_tail_change']
            if reference==0:
                if ratio is not None:raise ContractError('zero-reference ratio must be undefined')
            else:same(nonnegative(ratio),change/reference)
            for key in ['tail_change_rms','tail_reference_rms']:
                measurements[prefix+name+'.'+key]=effect[key]
            measurements[prefix+name+'.relative_defined']=int(reference>0)
            if reference>0:measurements[prefix+name+'.relative_tail_change']=ratio
        for key in ['tail_change_rms','tail_reference_rms']:
            # Compare scaled RMS to avoid overflow for otherwise finite values.
            values=[row['bypass'][name][key] for name in slices]
            scale=max(values)
            if scale:
                aggregate=sum((row['bypass'][name][key]/scale)**2*n/slices['all'] for name,n in slices.items() if name!='all')
                same((row['bypass']['all'][key]/scale)**2,aggregate)
    # Never propagate free-form text or extra data from a model-controlled worker.
    return {'measurements':measurements,'limitations':[
        ('Sampled training-history' if 'history_hash' in expected else 'Synthetic-history')+' activation and representation sensitivity; not recommendation accuracy.',
        'Untrusted model-process observations passed host schema and arithmetic checks, not independent scientific verification.',
        'Block bypass can be out of distribution; a missing relative value means zero reference RMS.']}
