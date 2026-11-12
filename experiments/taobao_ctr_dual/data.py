"""Task-owned chronological data construction; labels never enter model features."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import duckdb
import numpy as np

from common import read, write, sha, identity

PROFILE = ['cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
           'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']
FIELDS = ['user', 'adgroup_id', 'cate_id', 'campaign_id', 'customer', 'brand', 'pid',
          *PROFILE, 'hour', 'weekday', 'price_bucket']
BUCKETS = [262145, 262145, 16385, 262145, 131073, 131073, 129,
           2049, 129, 5, 17, 9, 9, 5, 17, 25, 8, 12]
LOOKBACK = 14 * 86400


def literal(path):
    return "'" + str(path).replace("'", "''") + "'"


def numeric_hash(expression, buckets):
    return f'CASE WHEN TRY_CAST({expression} AS BIGINT) IS NULL THEN 0 ELSE CAST(((TRY_CAST({expression} AS BIGINT) % {buckets-1})+{buckets-1})%{buckets-1}+1 AS UINTEGER) END'


def memmap(root, name, shape, dtype):
    return np.lib.format.open_memmap(root / (name + '.npy'), mode='w+', shape=shape, dtype=dtype)


def key(users, times):
    return (np.asarray(users, dtype=np.uint64) << np.uint64(32)) | np.asarray(times, dtype=np.uint64)


def prepare(raw, output, fixture=False, resume=False):
    started = time.time()
    if resume:
        if (output/'metadata.json').exists() or read(output/'progress.json')['phase']!='sort_behavior_log':
            raise ValueError('Only the committed pre-sort stage can be resumed')
        acquisition=read(output.parent/'assets/acquisition.json')
        expected={entry['name']:entry['expected_sha256'] for entry in acquisition['reused_tables']['files']}
        expected['behavior_log.csv']=acquisition['behavior_log']['sha256']
        for name,digest in expected.items():
            if sha(raw/name)!=digest:raise ValueError('Resume source identity mismatch: '+name)
        write(output/'resume-receipt.json',{'resumed_unix':started,'source_hashes':expected,'reason':'retain committed static arrays and full behavior table; exclude out-of-window anomalous timestamps before key checks'})
    else:
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'progress.json', {'phase': 'audit_impressions', 'started_unix': started})
    con = duckdb.connect(str(output / 'prepare.duckdb'))
    con.execute("SET threads=24")
    con.execute("SET memory_limit='180GB'")
    con.execute('SET preserve_insertion_order=true')
    required = ['raw_sample', 'ad_feature', 'user_profile', 'behavior_log']
    for stem in required:
        if not (raw / (stem + '.csv')).is_file():
            raise FileNotFoundError(stem + '.csv')
    if not resume:
        # Serial CSV scan preserves physical source row identity before any join.
        con.execute(f'''CREATE TABLE samples AS SELECT row_number() OVER ()-1 AS sample_id,
            CAST("user" AS BIGINT) AS uid, CAST(time_stamp AS BIGINT) AS ts,
            CAST(adgroup_id AS BIGINT) AS aid, pid, CAST(clk AS INTEGER) AS label,
            CAST(nonclk AS INTEGER) AS complement, CAST(floor((CAST(time_stamp AS BIGINT)+28800)/86400) AS BIGINT) AS day
            FROM read_csv({literal(raw/'raw_sample.csv')}, header=true, all_varchar=true, parallel=false)''')
        invalid = con.execute('SELECT count(*) FROM samples WHERE label IS NULL OR label NOT IN (0,1) OR complement IS NULL OR label+complement<>1 OR uid<0 OR uid>=4294967296 OR ts<0 OR ts>=4294967296').fetchone()[0]
        if invalid:
            raise ValueError('Invalid exposure labels or compound-key range: ' + str(invalid))
        days = con.execute('SELECT day, count(*), sum(label) FROM samples GROUP BY day ORDER BY day').fetchall()
        if len(days) != 8 or [d[0] for d in days] != list(range(days[0][0], days[0][0]+8)):
            raise ValueError('Require eight contiguous exposure days: ' + str(days))
        if any(p <= 0 or p >= n for _, n, p in days):
            raise ValueError('A daily partition has only one class')
        first_day = days[0][0]
        n = sum(row[1] for row in days)
        for stem, name, id_col in [('ad_feature', 'ads', 'adgroup_id'), ('user_profile', 'users', 'userid')]:
            # Strip publisher header whitespace before constructing a view.
            relation = con.read_csv(str(raw/(stem+'.csv')), header=True, all_varchar=True, parallel=False)
            original = relation.columns
            relation.project(','.join('"'+c.replace('"','""')+'" AS "'+c.strip()+'"' for c in original)).create_view(name)
            count, distinct = con.execute(f'SELECT count(*),count(DISTINCT {id_col}) FROM {name}').fetchone()
            if count != distinct:
                raise ValueError('Nonunique static join key: ' + stem)
        expressions = []
        source_fields = ['s.uid', 's.aid', 'a.cate_id', 'a.campaign_id', 'a.customer', 'a.brand', 's.pid',
                         *['u.'+field for field in PROFILE]]
        for field, expression, buckets in zip(FIELDS, source_fields, BUCKETS):
            encoded = f'CAST(hash({expression})%{buckets-1}+1 AS UINTEGER)' if field == 'pid' else numeric_hash(expression, buckets)
            expressions.append(encoded + ' AS f' + str(len(expressions)))
        expressions += ['CAST(floor((s.ts+28800)%86400/3600)+1 AS UINTEGER) AS hour',
                        'CAST((s.day+3)%7+1 AS UINTEGER) AS weekday',
                        'CAST(CASE WHEN TRY_CAST(a.price AS DOUBLE) IS NULL THEN 0 WHEN TRY_CAST(a.price AS DOUBLE)<=0 THEN 1 WHEN TRY_CAST(a.price AS DOUBLE)<=10 THEN 2 WHEN TRY_CAST(a.price AS DOUBLE)<=30 THEN 3 WHEN TRY_CAST(a.price AS DOUBLE)<=50 THEN 4 WHEN TRY_CAST(a.price AS DOUBLE)<=100 THEN 5 WHEN TRY_CAST(a.price AS DOUBLE)<=200 THEN 6 WHEN TRY_CAST(a.price AS DOUBLE)<=500 THEN 7 WHEN TRY_CAST(a.price AS DOUBLE)<=1000 THEN 8 WHEN TRY_CAST(a.price AS DOUBLE)<=5000 THEN 9 ELSE 10 END AS UINTEGER) AS price_bucket']
        static = memmap(output, 'static', (n,len(FIELDS)), np.uint32)
        labels = memmap(output, 'labels', (n,), np.uint8)
        users = memmap(output, 'users', (n,), np.uint32)
        times = memmap(output, 'times', (n,), np.uint32)
        day_indices = memmap(output, 'days', (n,), np.uint8)
        query = 'SELECT s.uid,s.ts,s.label,s.day,' + ','.join(expressions) + ' FROM samples s LEFT JOIN ads a ON s.aid=TRY_CAST(a.adgroup_id AS BIGINT) LEFT JOIN users u ON s.uid=TRY_CAST(u.userid AS BIGINT) ORDER BY s.sample_id'
        reader = con.execute(query).fetch_record_batch(262144)
        offset = 0
        for batch in reader:
            count = batch.num_rows
            values = [column.to_numpy(zero_copy_only=False) for column in batch.columns]
            users[offset:offset+count], times[offset:offset+count], labels[offset:offset+count] = values[:3]
            day_indices[offset:offset+count] = values[3] - first_day
            static[offset:offset+count] = np.column_stack(values[4:])
            offset += count
        if offset != n:
            raise ValueError('Static joins changed the exposure population')
        for array in [static,labels,users,times,day_indices]:
            array.flush()
        train_frequencies=np.bincount(np.asarray(users[day_indices<6],dtype=np.int64),minlength=int(users.max())+1)
        np.save(output/'user_train_frequency.npy',train_frequencies)
    else:
        days=con.execute('SELECT day,count(*),sum(label) FROM samples GROUP BY day ORDER BY day').fetchall()
        first_day=days[0][0];n=sum(row[1] for row in days)
        static=np.load(output/'static.npy',mmap_mode='r')
        labels=np.load(output/'labels.npy',mmap_mode='r')
        users=np.load(output/'users.npy',mmap_mode='r')
        times=np.load(output/'times.npy',mmap_mode='r')
        day_indices=np.load(output/'days.npy',mmap_mode='r')
        if static.shape!=(n,len(FIELDS)) or any(len(array)!=n for array in [labels,users,times,day_indices]):
            raise ValueError('Incomplete pre-sort arrays')
        if not (output/'user_train_frequency.npy').exists():raise ValueError('Missing training-frequency diagnostic')
    write(output/'progress.json', {'phase':'sort_behavior_log','impressions':n,'elapsed_seconds':time.time()-started})
    lo = first_day*86400-28800-LOOKBACK
    hi = (first_day+8)*86400-28800
    if not resume:
        con.execute(f'''CREATE TABLE behaviors AS SELECT row_number() OVER ()-1 AS event_id, CAST("user" AS BIGINT) AS uid,
            CAST(time_stamp AS BIGINT) AS ts, TRY_CAST(cate AS BIGINT) AS cate,
            TRY_CAST(brand AS BIGINT) AS brand, btag
            FROM read_csv({literal(raw/'behavior_log.csv')},header=true,all_varchar=true,parallel=false)''')
    behavior_counts = con.execute('SELECT btag,count(*) FROM behaviors GROUP BY btag').fetchall()
    allowed = {'pv','ipv','cart','fav','buy'}
    if not {x[0] for x in behavior_counts} <= allowed:
        raise ValueError('Unknown behavior types: ' + str(behavior_counts))
    anomalous_times=con.execute('SELECT count(*) FROM behaviors WHERE ts IS NULL OR ts<0 OR ts>=4294967296').fetchone()[0]
    if con.execute(f'SELECT count(*) FROM behaviors WHERE ts>={lo} AND ts<{hi} AND (uid IS NULL OR uid<0 OR uid>=4294967296)').fetchone()[0]:
        raise ValueError('Invalid user identity inside the eligible behavior window')
    m = con.execute(f'SELECT count(*) FROM behaviors WHERE ts>={lo} AND ts<{hi}').fetchone()[0]
    histories = memmap(output,'history',(m,4),np.uint32)
    keys = memmap(output,'history_keys',(m,),np.uint64)
    query = f'''SELECT uid,ts,{numeric_hash('cate',BUCKETS[2])},{numeric_hash('brand',BUCKETS[5])},
        CASE btag WHEN 'pv' THEN 1 WHEN 'ipv' THEN 1 WHEN 'cart' THEN 2 WHEN 'fav' THEN 3 WHEN 'buy' THEN 4 END
        FROM behaviors WHERE ts>={lo} AND ts<{hi} ORDER BY uid,ts,event_id'''
    offset=0
    for batch in con.execute(query).fetch_record_batch(1048576):
        values=[column.to_numpy(zero_copy_only=False) for column in batch.columns]
        count=batch.num_rows
        histories[offset:offset+count]=np.column_stack([values[2],values[3],values[4],values[1]])
        keys[offset:offset+count]=key(values[0],values[1])
        offset+=count
    histories.flush();keys.flush()
    con.close()
    write(output/'progress.json',{'phase':'causal_prefix_index','events':m,'elapsed_seconds':time.time()-started})
    starts=memmap(output,'history_start',(n,),np.int64)
    ends=memmap(output,'history_end',(n,),np.int64)
    for offset in range(0,n,262144):
        stop=min(n,offset+262144)
        start_times=np.maximum(times[offset:stop].astype(np.int64)-LOOKBACK,0)
        starts[offset:stop]=np.searchsorted(keys,key(users[offset:stop],start_times),side='left')
        ends[offset:stop]=np.searchsorted(keys,key(users[offset:stop],times[offset:stop]),side='left')
    starts.flush();ends.flush()
    # Deterministic sentinel coverage includes all exposure days, empty histories and ties.
    checked=0
    for offset in range(0,n,262144):
        stop=min(n,offset+262144)
        selected=np.arange(offset,stop,97)
        valid=ends[selected]>starts[selected]
        rows=selected[valid]
        positions=ends[rows]-1
        if np.any((keys[positions]>>np.uint64(32))!=users[rows]) or np.any(histories[positions,3]>=times[rows]):
            raise ValueError('Causal-prefix audit failed')
        checked+=len(rows)
    manifests={stem: {'path':str((raw/(stem+'.csv')).resolve()),'bytes':(raw/(stem+'.csv')).stat().st_size,'sha256':sha(raw/(stem+'.csv'))} for stem in required}
    meta={'fixture':fixture,'rows':n,'events':m,'fields':FIELDS,'buckets':BUCKETS,
          'category_field_index':2,'brand_field_index':5,'history_columns':['category','brand','behavior_type','event_time'],
          'split':{'timezone':'Asia/Shanghai','train_days':list(range(6)),'validation_day':6,'test_day':7,
                   'dates':[datetime.fromtimestamp(d*86400,timezone.utc).date().isoformat() for d,_,_ in days]},
          'lookback_seconds':LOOKBACK,'feature_fitting':'predeclared numeric modulo hashes; pid uses the pinned DuckDB hash; missing token=0; no fitted held-out statistics',
          'hash_collision_limitation':'IDs share finite hash buckets; category/brand namespaces align through identical numeric mappings, not verified fine-grained item identities',
          'static_snapshot_assumption':'Publisher-provided user/ad snapshots treated as static benchmark attributes. Their as-of dates are undocumented; this is not verified production point-in-time data.',
          'source_files':manifests,'duckdb_version':duckdb.__version__,
          'out_of_window_anomalous_timestamps':anomalous_times,'anomaly_policy':'Retain the original source; histories use the declared lookback and exposure window, excluding out-of-window negative/future timestamps. Do not drop exposures or entire users.',
          'behavior_types':dict(behavior_counts),'pv_alias':{'ipv':'pv'},'causal_sentinel_exposures':checked,
          'elapsed_seconds':time.time()-started}
    meta['identity']=identity(meta)
    write(output/'metadata.json',meta)
    # Label-distribution audit stays on the host, outside the coding-agent export.
    write(output/'host-label-audit.json',{'daily_counts':[{'day_index':i,'samples':int(c),'positives':int(p)} for i,(_,c,p) in enumerate(days)]})
    write(output/'progress.json',{'phase':'complete','elapsed_seconds':time.time()-started,'identity':meta['identity']})
    return meta


class Data:
    def __init__(self,root,history_limit=128):
        self.root=Path(root)
        self.meta=read(self.root/'metadata.json')
        self.history_limit=history_limit
        for name in ['static','labels','users','times','days','history','history_start','history_end','user_train_frequency']:
            setattr(self,name,np.load(self.root/(name+'.npy'),mmap_mode='r'))

    def indices(self,split):
        if split=='train':return np.flatnonzero(self.days<6)
        if split=='validation':return np.flatnonzero(self.days==6)
        if split=='test':return np.flatnonzero(self.days==7)
        raise ValueError('Unknown split')

    def batch(self,indices,device):
        import torch
        indices=np.asarray(indices)
        ends=self.history_end[indices]
        positions=ends[:,None]-self.history_limit+np.arange(self.history_limit)[None,:]
        mask=positions>=self.history_start[indices,None]
        values=self.history[np.maximum(positions,0)]
        values[~mask]=0
        age=np.maximum(self.times[indices,None].astype(np.int64)-values[:,:,3],0).astype(np.float32)
        age[~mask]=0
        result={'static':np.asarray(self.static[indices],dtype=np.int64),
                'history_category':values[:,:,0].astype(np.int64),'history_brand':values[:,:,1].astype(np.int64),
                'history_type':values[:,:,2].astype(np.int64),'history_age_seconds':age,'history_mask':mask}
        return {name:torch.as_tensor(value,device=device) for name,value in result.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--fixture',action='store_true')
    parser.add_argument('--resume-behavior-stage',action='store_true')
    args=parser.parse_args()
    prepare(args.raw.resolve(),args.output.resolve(),args.fixture,args.resume_behavior_stage)
