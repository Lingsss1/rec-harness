import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event,get_ident
from rec_harness.gpu_scheduler import dispatch


class SchedulerTests(unittest.TestCase):
    def test_reuses_free_device_before_slow_peer_finishes(self):
        third_started=Event();launched=[];finished=[];controller=get_ident()
        def run(job):
            if job=='slow':
                if not third_started.wait(3):raise AssertionError('idle GPU was not refilled')
            if job=='third':third_started.set()
            return job
        with ThreadPoolExecutor(max_workers=2) as pool:
            def launch(job,device):
                self.assertEqual(get_ident(),controller);launched.append((job,device))
                return pool.submit(run,job)
            def complete(job,device,future):
                self.assertEqual(get_ident(),controller);finished.append(future.result())
            pending=dispatch(['slow','fast','third'],[0,1],lambda:True,launch,complete)
        self.assertEqual(pending,[]);self.assertEqual(launched,[('slow',0),('fast',1),('third',1)])
        self.assertCountEqual(finished,['slow','fast','third'])

    def test_budget_waits_for_settlement_and_never_overreserves(self):
        remaining=[1.];launches=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            def launch(job,device):
                self.assertGreaterEqual(remaining[0],1);remaining[0]-=1;launches.append(job)
                return pool.submit(lambda:job)
            def complete(job,device,future):
                future.result()
                # First reservation refunds half, too little for another full job.
                remaining[0]+=.5
            pending=dispatch(['first','second'],[0,1],lambda:remaining[0]>=1,launch,complete)
        self.assertEqual(launches,['first']);self.assertEqual(pending,['second'])

    def test_failed_job_can_be_accounted_and_slot_reused(self):
        finished=[]
        with ThreadPoolExecutor(max_workers=1) as pool:
            def work(job):
                if job=='fail':raise RuntimeError('training failed')
                return job
            def complete(job,device,future):
                try:future.result();finished.append('success')
                except RuntimeError:finished.append('failed')
            self.assertEqual(dispatch(['fail','next'],[0],lambda:True,
                lambda job,device:pool.submit(work,job),complete),[])
        self.assertEqual(finished,['failed','success'])
