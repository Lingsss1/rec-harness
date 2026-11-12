"""Archive exact public reference versions for the external task knowledge index."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import time
import urllib.request

from common import read,write,sha

PROJECT=Path(__file__).resolve().parents[2]
DESIGN=PROJECT/'task_designs/autonomous_research_v1'


class TextParser(HTMLParser):
    def __init__(self):super().__init__();self.parts=[];self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in {'script','style'}:self.skip+=1
        if tag in {'p','div','section','h1','h2','h3','h4','tr','li','br'}:self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in {'script','style'}:self.skip=max(0,self.skip-1)
        if tag in {'p','section','h1','h2','h3','h4','tr','li'}:self.parts.append('\n')
    def handle_data(self,data):
        if not self.skip:self.parts.append(data)
    def text(self):return '\n'.join(line.strip() for line in ''.join(self.parts).splitlines() if line.strip())+'\n'


def cache(root):
    output=root/'knowledge-assets-v2';output.mkdir(exist_ok=True)
    task=DESIGN/'tasks/taobao_display_ctr'
    industry=read(task/'industry/sources.json');papers=read(task/'paper_sources.json')
    entries=industry['sources']+papers['sources']
    urls={'paper:din':'https://arxiv.org/html/1706.06978v4',
          'paper:dmr':'https://raw.githubusercontent.com/lvze92/DMR/2c6e76804281954a3c969010a31712cd5054aca9/README.md'}
    def acquire(entry):
        entry=dict(entry);identifier=entry['id'];stem=identifier.replace(':','_')
        descriptor={**entry,'scientific_status':'external_reference','local_reproduction':False,'cached_unix':time.time()}
        if identifier=='taac2026-organizer-overview':
            source=DESIGN/'source_cache/TAAC_2026.pdf'
            if sha(source)!=entry['sha256']:raise ValueError('Organizer source integrity changed')
            text=output/(stem+'.txt');shutil.copyfile(DESIGN/'source_cache/TAAC_2026.txt',text)
            descriptor.update(text_path=text.name,text_sha256=sha(text),content_sha256=entry['sha256'],
                              access_scope='archived organizer overview with PDF page markers')
        elif entry.get('url'):
            url=urls.get(identifier,entry['url']);text=output/(stem+'.txt');raw=output/(stem+'.source')
            try:
                if not raw.exists():
                    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research-reference-archive/1.0'}),timeout=45) as response:
                        content=response.read(8<<20)
                    raw.write_bytes(content)
                else:content=raw.read_bytes()
                decoded=content.decode('utf-8',errors='replace')
                if '<html' in decoded[:5000].lower() or '<!doctype html' in decoded[:5000].lower():
                    parser=TextParser();parser.feed(decoded);decoded=parser.text()
                if len(decoded)<200:raise ValueError('Source text unexpectedly short')
                text.write_text(decoded)
                descriptor.update(text_path=text.name,text_sha256=sha(text),content_sha256=sha(raw),retrieved_url=url,
                                  access_scope='author README only, not the blocked publisher paper' if identifier=='paper:dmr' else 'archived exact-version primary text; caching does not establish a local mechanism result')
            except Exception as error:
                descriptor.update(cache_error=repr(error),text_path=None,access_scope='URL metadata only; primary source remains externally accessible')
        else:
            descriptor.update(text_path=None,fallback_ref='taac2026-organizer-overview',access_scope='detailed report not located; organizer summary only')
        descriptor.pop('local_cache',None)
        write(output/(stem+'.json'),descriptor)
        return descriptor
    with ThreadPoolExecutor(4) as executor:result=list(executor.map(acquire,entries))
    write(output/'manifest.json',{'revision':'knowledge-access-v2','sources':result,'immutable_versions':True,
                                'industry_card_count':5,'not_local_experimental_evidence':True})
    print(json.dumps({'sources':len(result),'cached_texts':sum(bool(row.get('text_path')) for row in result),
                      'missing_detailed_reports':[row['id'] for row in result if row.get('fallback_ref')]}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();cache(args.root.resolve())
