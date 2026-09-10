import importlib.util, math, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'scripts/paired_probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
def receipt(ranks):
    return {'protocolId':'p','split':'validation','codeIdentity':'c','records':[{'user':str(i),'rank':r,'hr':float(r<5),'ndcg':1/math.log2(r+2) if r<5 else 0.} for i,r in enumerate(ranks)]}
class PairedProbeTest(unittest.TestCase):
    def test_known_gain_loss_and_rank_decomposition(self):
        x=p.paired(receipt([8,0,2,9]),receipt([0,7,1,8]));self.assertEqual(x['hrDelta'],0);self.assertEqual([g['count'] for g in x['groups'].values()],[1,1,1,1]);self.assertAlmostEqual(x['ndcgDelta'],(1/math.log2(3)-.5)/4)
    def test_null_and_symmetry(self):
        a,b=receipt([0,8,2]),receipt([5,1,4]);self.assertEqual(p.paired(a,a)['ndcgDelta'],0);self.assertAlmostEqual(p.paired(a,b)['ndcgDelta'],-p.paired(b,a)['ndcgDelta'])
    def test_reject_invalid_or_stale(self):
        a=receipt([0,2]);variants=[]
        for key,val in [('protocolId','old'),('split','test'),('records',[])]:
            b=receipt([0,2]);b[key]=val;variants.append(b)
        b=receipt([0]);variants.append(b)
        b=receipt([0,2]);b['records'][0]['ndcg']=float('nan');variants.append(b)
        b=receipt([0,2]);b['records'][1]['user']='0';variants.append(b)
        for b in variants:
            with self.assertRaises(ValueError):p.paired(a,b)
if __name__=='__main__':unittest.main()
