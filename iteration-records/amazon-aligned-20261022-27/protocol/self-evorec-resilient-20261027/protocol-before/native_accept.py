"""Import and initialize the actual upstream method components, without training."""
import argparse
import ast
from pathlib import Path
import sys

from .bootstrap import bootstrap
bootstrap()
from .core import read, save
from .prepare import ROOT, WORKSPACE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['mlevolve', 'self_evolverec'])
    args = parser.parse_args()
    config = read(ROOT / 'config.json')
    test_config = {**config, 'root': str(ROOT / 'preflight')}
    if args.method == 'mlevolve':
        from .mlevolve import NativeController, build
        controller = NativeController(test_config, 'native-mlevolve')
        controller.best = {'id': 'not-trained-acceptance-only', 'status': 'not_trained', 'score': None}
        agent = build(controller)
        assert agent.global_memory is not None
        assert agent.scfg.parallel_search_num == 4
        assert len(agent.journal.nodes) == 1
        save(ROOT / 'preflight/native-mlevolve.json', {'passed': True,
            'engine': type(agent).__name__, 'memory': type(agent.global_memory).__name__,
            'metric_maximize': agent.metric_maximize})
    else:
        upstream = WORKSPACE / 'artifacts/Self-EvolveRec-code'
        sys.path.insert(0, str(upstream))
        from omegaconf import OmegaConf
        from agents import Agent
        from coder_rec import CoderAgent
        from coder_rec_diagnosis import CoderAgent_Diagnosis
        from researcher_rec import ResearcherAgent
        from researcher_rec_diagnosis import ResearcherAgent_Diagnosis
        from database2 import ProgramDatabase
        from utils.datatypes import IdeaData, ReportData, WebSearchPlan, ReflectionPlan
        from .self_evolverec import parse_files, INTERFACE, create_engine
        from self_evolverec import Self_EvolveRec
        from .core import Controller, strict_schema
        c = Controller(test_config, 'native-self')
        joined = '# === main_code.py ===\n' + c.baseline() + '\n# === diagnosis_tools.py ===\n'
        joined += (upstream / 'examples/sasrec_cds/initial_code/diagnosis_tools.py').read_text()
        joined += '\n# === self_evolverec_interface.py ===\n' + INTERFACE
        files = parse_files(joined)
        for text in files.values():
            ast.parse(text)
        model = config['llm_model']
        effort = config['llm_effort']
        components = [CoderAgent(model, model, effort), CoderAgent_Diagnosis(model, model, effort),
                      ResearcherAgent(model, model, model, effort),
                      ResearcherAgent_Diagnosis(model, model, model, effort)]
        cfg = OmegaConf.load(upstream / 'configs/config_baseline.yaml')
        database = ProgramDatabase(cfg.database)
        assert len(database.islands) == 3
        Self_EvolveRec._setup_logging = lambda self: None
        engine = create_engine(c, 0, (upstream / 'examples/sasrec_cds/initial_code/diagnosis_tools.py').read_text())
        assert parse_files(engine.initial_code).keys() == files.keys()
        assert engine.config.coder.reasoning_effort == config['llm_effort']
        assert engine.config.researcher.reasoning_effort == config['llm_effort']
        for role in [engine.researcher, engine.diagnosis_researcher, engine.coder,
                     engine.diagnosis_coder, engine.debugger]:
            role.update_topic('Shared Amazon task', 'amazon', 'Fixed external validation contract')
        for output_type in [IdeaData, ReportData, WebSearchPlan, ReflectionPlan]:
            normalized = strict_schema(output_type.model_json_schema())
            assert normalized['additionalProperties'] is False
            assert all(item['additionalProperties'] is False for item in normalized.get('$defs', {}).values())
        save(ROOT / 'preflight/native-self.json', {'passed': True,
             'components': [type(x).__name__ for x in components], 'database': type(database).__name__,
             'islands': len(database.islands), 'source_files': sorted(files),
             'initialized_engine': type(engine).__name__, 'reasoning_effort': config['llm_effort']})
    print('NATIVE_' + args.method.upper() + '_ACCEPTED', flush=True)


if __name__ == '__main__':
    main()
