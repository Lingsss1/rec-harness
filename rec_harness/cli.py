"""Local CLI. Campaign input paths are relative to the configuration file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agents import CodexAgent
from .core import ContractError, canonical, read_json, write_json
from .evolution import Evolution
from .migration import export_memory, import_legacy, import_memory
from .replay import Replay, build_world, world_from_campaign
from .runtime import Research
from .store import Store


def inspection_controller(directory):
    root=Path(directory)
    if (root/'project.json').exists():
        from .project import ResearchProject
        return ResearchProject(root)
    if (root/'evolution-auto.json').exists():return Evolution(root)
    if (root/'amazon-auto.json').exists():
        from .adapters.amazon_autonomous import AmazonAutoresearch
        return AmazonAutoresearch(root,load_agent=False)
    return Research(root)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rec-harness")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser('amazon-final-seal',help='freeze validation-selected checkpoints and close research before host-only final evaluation')
    p.add_argument('study');p.add_argument('output');p.add_argument('--model',action='append',required=True)
    p.add_argument('--test-directory',required=True);p.add_argument('--eval-seed',type=int,required=True)
    p.add_argument('--eval-users',type=int,default=1000)
    p = commands.add_parser('amazon-final-run',help='run host-only final checkpoint scoring from a sealed selection')
    p.add_argument('output');p.add_argument('--recipe',required=True);p.add_argument('--uid-registry',required=True)
    p = commands.add_parser('amazon-restore-archive',help='restore archived job files and expand frozen runtime references; does not execute a job')
    p.add_argument('snapshot');p.add_argument('objects');p.add_argument('output')
    p = commands.add_parser('amazon-retry-source',help='reopen a failed pre-admission round after a host repair; retain costs and outputs')
    p.add_argument('directory');p.add_argument('--reason',required=True)
    p = commands.add_parser('autoresearch',help='create or resume autonomous research through its frozen round/budget stop')
    p.add_argument('directory');p.add_argument('--config',help='initial public task configuration; omit on resume')
    p = commands.add_parser('activation-probe',help='run or recover an isolated local NextItNet activation probe')
    p.add_argument('config');p.add_argument('directory')
    p = commands.add_parser("init", help="freeze public workspace, evaluator and protocol")
    p.add_argument("config"); p.add_argument("directory")
    p = commands.add_parser("run", help="run research roles via codex exec")
    p.add_argument("directory"); p.add_argument("--rounds", type=int)
    p = commands.add_parser("consolidate", help="retry graph knowledge without repeating training")
    p.add_argument("directory"); p.add_argument("attempt")
    for name in ("status", "verify", "recover"):
        p = commands.add_parser(name); p.add_argument("directory")
    p = commands.add_parser("graph")
    p.add_argument("directory"); p.add_argument("action", choices=["snapshot", "patch", "rollback", "export", "rebuild", "bundle", "bundle-pair"])
    p.add_argument("--file"); p.add_argument("--version", type=int); p.add_argument("--task"); p.add_argument("--protocol")
    p.add_argument("--role", default="evidence"); p.add_argument("--query", default="")
    p = commands.add_parser("migrate")
    p.add_argument("directory"); p.add_argument("source"); p.add_argument("--task", required=True); p.add_argument("--protocol", default="unknown")
    p = commands.add_parser("memory")
    p.add_argument("directory"); p.add_argument("action", choices=["export", "import"]); p.add_argument("file")
    p.add_argument("--task", required=True); p.add_argument("--protocol", required=True); p.add_argument("--version", type=int)
    p.add_argument("--cutoff-seq", type=int); p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--max-bytes", type=int, default=1048576)
    p = commands.add_parser("world-from-campaign")
    p.add_argument("directory"); p.add_argument("output"); p.add_argument("--workers", type=int, default=1)
    p = commands.add_parser('amazon-run-phase', help='run a registered phase and reuse audited existing comparators')
    p.add_argument('directory'); p.add_argument('proposal'); p.add_argument('phase', choices=['development','confirmation']); p.add_argument('--execution')
    p = commands.add_parser('amazon-run-development',help='run or resume admitted matched development, including controls and independent scoring; no Codex calls')
    p.add_argument('directory');p.add_argument('proposal');p.add_argument('output')
    p.add_argument('--execution',required=True);p.add_argument('--recipe',required=True);p.add_argument('--uid-registry',required=True)
    p = commands.add_parser('amazon-review-development',help='resume evaluation, lead and knowledge for a recovered preregistered development matrix')
    p.add_argument('directory');p.add_argument('proposal');p.add_argument('source_round');p.add_argument('rescores')
    p.add_argument('--config',help='initial autonomous review configuration; omit on resume')
    p = commands.add_parser('amazon-review-retry',help='archive an exhausted review role after a host repair without resetting charges')
    p.add_argument('directory');p.add_argument('--reason',required=True)
    p = commands.add_parser('amazon-call-report',help='read-only role submissions, outcomes and retained costs by round')
    p.add_argument('directory')
    p.add_argument('--format',choices=['json','markdown'],default='json')
    p = commands.add_parser('amazon-continue', help='create an authorized successor campaign with inherited costs')
    p.add_argument('directory'); p.add_argument('predecessor'); p.add_argument('project')
    p.add_argument('--additional-budget',type=float,required=True); p.add_argument('--authorization-file',required=True)
    p = commands.add_parser('amazon-activate', help='recover a prepared continuation after predecessor sealing')
    p.add_argument('directory')
    p = commands.add_parser('amazon-development', help='audit the complete development matrix of a registered research plan')
    p.add_argument('directory'); p.add_argument('proposal'); p.add_argument('output')
    p = commands.add_parser('amazon-memory-export',help='export development-only memory for a subsequent campaign')
    p.add_argument('directory');p.add_argument('output')
    p = commands.add_parser('amazon-confirm', help='freeze an eligible development selection before confirmation runs')
    p.add_argument('directory'); p.add_argument('proposal'); p.add_argument('decision'); p.add_argument('--readout-hash', required=True)
    p = commands.add_parser('amazon-plan', help='freeze research handoffs and candidate/control sources without executing code')
    p.add_argument('directory'); p.add_argument('handoff')
    p = commands.add_parser('amazon-summary', help='publish audited validation contrasts, counterevidence and complete budget')
    p.add_argument('directory'); p.add_argument('output')
    p = commands.add_parser('amazon-archive', help='audit real GPU receipts and resource ledger without inventing replay transitions')
    p.add_argument('directory'); p.add_argument('output')
    p = commands.add_parser('amazon-replay-world', help='freeze one observed development cycle for bounded historical replay')
    p.add_argument('directory');p.add_argument('output')
    p.add_argument('--gpu-second-weight',type=float,required=True);p.add_argument('--role-call-weight',type=float,required=True)
    p = commands.add_parser('amazon-replay-chain', help='freeze ordered observed cycles with committed cost lineage')
    p.add_argument('output');p.add_argument('campaigns',nargs='+')
    p.add_argument('--gpu-second-weight',type=float,required=True);p.add_argument('--role-call-weight',type=float,required=True)
    p = commands.add_parser('amazon-history-diagnostic', help='aggregate paired validation gains by training-history length')
    p.add_argument('directory'); p.add_argument('output')
    p.add_argument('--baseline',default='legacy_no_prefix');p.add_argument('--candidate',default='legacy_incumbent')
    p.add_argument('--partition',action='append',choices=['development','confirmation'])
    p = commands.add_parser('amazon-block-score', help='host-score a reference or one/multiple-block bypass on development data')
    p.add_argument('directory');p.add_argument('model_version');p.add_argument('dataset');p.add_argument('seed',type=int);p.add_argument('output')
    p.add_argument('--execution',required=True)
    group=p.add_mutually_exclusive_group();group.add_argument('--bypass-block',type=int)
    group.add_argument('--bypass-blocks',type=int,nargs='+')
    p.add_argument('--resume',action='store_true',help='reuse completed inference and recover successful process collection')
    p = commands.add_parser('amazon-block-interaction',help='recompute a paired reference/single/single/double block contrast')
    p.add_argument('directories',nargs=4);p.add_argument('--output',required=True)
    p = commands.add_parser("world-build")
    p.add_argument("nodes"); p.add_argument("output"); p.add_argument("--workers", type=int, default=1)
    p = commands.add_parser("replay")
    p.add_argument("world"); p.add_argument("actions"); p.add_argument("--budget", type=float, required=True)
    p.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    p = commands.add_parser("replay-policy", help="run a prefix-only policy in a Linux filesystem/syscall sandbox")
    p.add_argument("world"); p.add_argument("policy"); p.add_argument("output")
    p.add_argument("--runtime", required=True); p.add_argument("--jails", required=True)
    p.add_argument("--budget", type=float, required=True); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    p = commands.add_parser("evolve")
    p.add_argument("directory"); p.add_argument("action", choices=["propose", "validate", "promote", "rollback"])
    p.add_argument("--id"); p.add_argument("--kind", choices=["tool", "harness"]); p.add_argument("--context"); p.add_argument("--plan"); p.add_argument("--stage")
    p = commands.add_parser("codex-smoke", help="one real read-only structured Codex call")
    p.add_argument("directory"); p.add_argument("--model"); p.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        if args.command == 'amazon-final-seal':
            from .adapters.amazon_final_selection import seal_selection
            plan=seal_selection(args.study,args.output,args.model,args.test_directory,eval_seed=args.eval_seed,eval_users=args.eval_users)
            result={'selection_hash':plan['hash'],'status':'sealed','test_data_opened':False}
        elif args.command == 'amazon-final-run':
            from .adapters.amazon_final import run_final
            report=run_final(args.output,read_json(args.recipe),args.uid_registry)
            result={'status':'completed','report_hash':report['hash'],'report':str(Path(args.output).resolve()/'final-report.json')}
        elif args.command == 'amazon-restore-archive':
            from .adapters.amazon_evidence import restore_job_archive
            from .core import Artifacts
            result=restore_job_archive(read_json(args.snapshot),Artifacts(args.objects),args.output)
        elif args.command == 'amazon-retry-source':
            from .adapters.amazon_autonomous import AmazonAutoresearch
            controller=AmazonAutoresearch(args.directory,load_agent=False)
            try:result=controller.reopen_source_failure(args.reason)
            finally:controller.close()
        elif args.command == 'autoresearch':
            from .autonomous import autoresearch
            result=autoresearch(args.directory,args.config)
        elif args.command == 'activation-probe':
            from .activation_runtime import run_probe
            report=run_probe(read_json(args.config),args.directory)
            result={key:report[key] for key in ['status','budget']}
            if report.get('reason'):result['reason']=report['reason']
            if report['status']=='success':result['measurement_file']=str(Path(args.directory).resolve()/'measurement.json')
        elif args.command == "init":
            config_path = Path(args.config).resolve()
            config = read_json(config_path)
            config["source"] = str((config_path.parent / config["source"]).resolve())
            if 'harness_registry' in config:
                config['harness_registry'] = str((config_path.parent / config['harness_registry']).resolve())
            result = {"campaign": str(Research.init(args.directory, config))}
        elif args.command in {"run", "consolidate"}:
            research = Research(args.directory)
            try:
                result = research.run(args.rounds) if args.command == "run" else research.consolidate(args.attempt)
            finally:
                research.store.close()
        elif args.command == 'status':
            from .status import inspect_status
            result=inspect_status(args.directory)
        elif args.command in {"verify", "recover", "graph", "migrate", "memory"}:
            store = Store(Path(args.directory) / "memory")
            try:
                if args.command == "verify":
                    research = inspection_controller(args.directory)
                    try:
                        result = research.verify()
                    finally:
                        research.store.close()
                elif args.command == "recover":
                    # Controller lock prevents marking live tasks interrupted.
                    research = inspection_controller(args.directory)
                    try:
                        with research.lock():
                            result = {"interrupted": store.recover()}
                    finally:
                        research.store.close()
                elif args.command == "migrate":
                    result = import_legacy(store, args.source, args.task, args.protocol)
                elif args.command == "memory":
                    result = export_memory(store, args.file, task_id=args.task, protocol_hash=args.protocol, version=args.version,
                                           cutoff_seq=args.cutoff_seq, limit=args.limit, max_bytes=args.max_bytes) if args.action == "export" else import_memory(store, args.file, args.task, args.protocol)
                elif args.action == "snapshot":
                    result = store.snapshot(args.version)
                elif args.action == "patch":
                    result = store.patch(read_json(args.file))
                elif args.action == "rollback":
                    if args.version is None:
                        raise ContractError("--version required")
                    result = store.rollback(args.version)
                elif args.action == "rebuild":
                    result = store.rebuild()
                elif args.action == "export":
                    store.export_compat(args.file); result = {"exported": args.file}
                else:
                    result = store.bundle(args.task, args.protocol, args.role, version=args.version, query=args.query)
                    if args.action == 'bundle-pair':
                        from .memory_view import paired_views
                        result = paired_views(result)
            finally:
                store.close()
        elif args.command == 'amazon-continue':
            from .adapters.amazon_campaign import continue_campaign
            result=continue_campaign(args.directory,args.predecessor,args.project,args.additional_budget,
                                     Path(args.authorization_file).read_text())
        elif args.command == 'amazon-activate':
            from .adapters.amazon_campaign import activate_campaign
            result=activate_campaign(args.directory)
        elif args.command == 'amazon-run-phase':
            from .adapters.amazon_phases import run_phase
            result = run_phase(args.directory, args.proposal, args.phase, args.execution)
        elif args.command == 'amazon-run-development':
            from .adapters.amazon_matched import run_matched_development
            result=run_matched_development(args.directory,args.proposal,args.output,
                execution_id=args.execution,recipe=read_json(args.recipe),uid_registry=args.uid_registry)
        elif args.command == 'amazon-review-development':
            from .adapters.amazon_autonomous import AmazonAutoresearch
            from .adapters.amazon_review_recovery import review_recovered
            if args.config:AmazonAutoresearch.init(args.directory,read_json(args.config),review_only=True)
            result=review_recovered(args.directory,args.proposal,args.source_round,args.rescores)
        elif args.command == 'amazon-review-retry':
            from .adapters.amazon_review_recovery import retry_review
            result=retry_review(args.directory,args.reason)
        elif args.command == 'amazon-call-report':
            from .adapters.amazon_call_report import call_report,markdown_report
            result=call_report(args.directory)
            if args.format=='markdown':print(markdown_report(result));return 0
        elif args.command == 'amazon-replay-chain':
            from .adapters.amazon_replay_chain import archive_chain
            result=archive_chain(args.campaigns,args.output,gpu_weight=args.gpu_second_weight,role_weight=args.role_call_weight)
        elif args.command == 'amazon-memory-export':
            from .adapters.amazon_memory import export_development_memory
            result=export_development_memory(args.directory,args.output)
        elif args.command == 'amazon-development':
            from .adapters.amazon_selection import development_readout
            result = development_readout(args.directory, args.proposal)
            if Path(args.output).exists():
                raise ContractError('development output must be new')
            write_json(args.output, result)
        elif args.command == 'amazon-confirm':
            from .adapters.amazon_selection import release_confirmation
            result = release_confirmation(args.directory, args.proposal, read_json(args.decision), args.readout_hash)
        elif args.command == 'amazon-plan':
            from .adapters.amazon_plan import register_plan
            handoff = read_json(args.handoff)
            result = register_plan(args.directory, handoff['cards'], handoff['versions'])
        elif args.command == 'amazon-block-score':
            from .adapters.amazon_controls import load_config
            from .adapters.amazon_rescore import rescore_job
            if args.seed not in load_config(args.directory)['development_seeds']:
                raise ContractError('block score diagnostics are development-only')
            execution=read_json(args.execution)
            result=rescore_job(args.directory,args.model_version,args.dataset,args.seed,args.output,
                execution['recipe'],execution['uid_registry'],bypass_block=args.bypass_block,bypass_blocks=args.bypass_blocks,resume=args.resume)
        elif args.command == 'amazon-block-interaction':
            from .adapters.block_interaction import summarize_interaction
            result=summarize_interaction(args.directories);write_json(args.output,result)
        elif args.command == 'amazon-summary':
            from .adapters.amazon_summary import build_summary
            report = build_summary(args.directory, args.output)
            result = {'hash': report['hash'], 'qualification': report['qualification'], 'output': args.output}
        elif args.command == 'amazon-history-diagnostic':
            from .adapters.history_diagnostic import diagnose
            report = diagnose(args.directory, args.output,args.baseline,args.candidate,
                              partitions=args.partition if args.partition is not None else ('development','confirmation'))
            result = {'hash': report['hash'], 'rows': len(report['rows']), 'scope': report['scope']}
        elif args.command == 'amazon-archive':
            from .adapters.amazon_archive import archive_study
            archive = archive_study(args.directory, args.output)
            result = {key: archive[key] for key in ['hash', 'qualification', 'charged_or_reserved_gpu_seconds', 'reasons']}
            result['record_count'] = len(archive['records'])
            result['output'] = args.output
        elif args.command == 'amazon-replay-world':
            from .adapters.amazon_replay import archive_cycle
            world=archive_cycle(args.directory,args.output,gpu_weight=args.gpu_second_weight,role_weight=args.role_call_weight)
            result={key:world[key] for key in ['hash','qualification','resources','support']}
            result['output']=args.output
        elif args.command == "world-from-campaign":
            result = world_from_campaign(args.directory, args.workers)
            write_json(args.output, result)
        elif args.command == "world-build":
            result = build_world(read_json(args.nodes), workers=args.workers)
            write_json(args.output, result)
        elif args.command == "replay":
            replay = Replay(read_json(args.world), args.budget, args.direction)
            for action in read_json(args.actions):
                if replay.stopped:
                    break
                replay.step(action)
            result = {"report": replay.report(), "visible_state": replay.view(), "history": replay.history}
        elif args.command == "replay-policy":
            from .policy_sandbox import PolicySandbox
            from .replay_policy import run_policy
            result = run_policy(read_json(args.world), args.policy, PolicySandbox(args.runtime, args.jails),
                                args.output, args.budget, seed=args.seed, max_steps=args.max_steps, direction=args.direction)
        elif args.command == "evolve":
            evolution = Evolution(args.directory)
            try:
                if args.action == "propose":
                    result = evolution.propose(args.id, args.kind, read_json(args.context), read_json(args.plan))
                elif args.action == "validate":
                    result = evolution.validate(args.id, args.stage)
                elif args.action == "promote":
                    result = evolution.promote(args.id)
                else:
                    result = evolution.rollback(args.kind)
            finally:
                evolution.store.close()
        else:
            result = CodexAgent(model=args.model, timeout=args.timeout).run("evaluation", {
                "task": "Transport smoke test only. No training. Return selection provisional, conclusion_status unknown, empty evidence_refs and unresolved_explanations; rationale must say transport test."}, args.directory)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ContractError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"rec-harness: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
