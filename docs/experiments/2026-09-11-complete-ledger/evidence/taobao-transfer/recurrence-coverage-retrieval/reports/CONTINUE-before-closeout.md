# Coverage retrieval: formal/quality/latency complete; paired analysis running

Use `workspace/ai4ai-runtime/opd-native-env/bin/python -B` for runtime scripts. System Python lacks hashlib.file_digest. All CUDA work requires the already-authorized escalated environment. Goal remains active: improve second-cycle retrieval with rec-harness memory/tools. This continuation is progress, not completion.

## Live handle — poll, never duplicate

**Paired analysis session10337 is RUNNING.** Command: `paired_analysis.py --run`. Mandatory16 comparisons, conditional0 because no per-parent winner beats best single. Last verified50s write_stdin returned live10337; reports/paired-analysis/progress.json reached pair3/16,375/500draws at about280s elapsed. Follow the same handle; observation timeout/silence is not termination. Fixed3600wall/14400CPU/16GiB,4threads,500shared user draws(seed20260906),2871767rows/268201users. All4AUC/BCE,macroAUC,6directtask contrasts. The two completed first parent comparisons have negative macro intervals; do not infer the remaining14.

## Terminal — do not rerun

- Formal session49697 EXIT0, all8 workers EXIT0 and2193updates each, completed observed2026-09-09T11:51:21Z. All artifacts saved. No extra training seed/test.
- Full quality audit session16421 EXIT0. reports/quality-summary.json SHAab3e9cad65194b7d34656032a6f18736c72826ecb77cb106797a588eb558bb05;329file identities incl122sharedtrain/validinputs,wall4.6865755s.
- Latency session6254 EXIT0: all11models, sameGPU0/shared1024batch/BF16,10warmups+40timed=550forwards. reports/latency/completion.json SHA99875d4e84ac6262cc4594bde6f006e127f5b6cac0d67f1dcf0c471b91cabb69;recordedwall29.9128459s. Do not repeatforreview.
- Tiny native46593 EXIT0(16stageforward/backward+2PV);8realnative two-update smokes98773 EXIT0. All gates reviewed. ModelCPU9/formula8/runner7/postprocess25tests passed. Do not repeat.
- Previous directional V round completely closed/consolidated: formal35368,audit2003,paired4422(all12),latency2586(all11), all EXIT0;no rerun.

## Actual current result — all eight fail retention

The eight coverage models all have negative macro point deltas below-.0001 and are8.13%–9.46%slower than their own original. Noquality ornear candidate regardless pending interval. Currentbest remains optimizedsingleCross8 .8286235325196747;originalq4 .8284725654262436;originalq8 .8285212133742255.

| Candidate | Macro AUC | Delta own parent |
|---|---:|---:|
| cov_q4_mean_positive | .8261778138431993 | -.0022947515830442633 |
| cov_q4_mean_negative | .8262218231384939 | -.002250742287749685 |
| cov_q4_loo_positive | .8235546443409760 | -.0049179210852675626 |
| cov_q4_loo_negative | .8257093643740046 | -.0027632010522390393 |
| cov_q8_mean_positive | .8247431500095996 | -.0037780633646258277 |
| cov_q8_mean_negative | .8250707110166375 | -.0034505023575879257 |
| cov_q8_loo_positive | .8274520920161917 | -.0010691213580337200 |
| cov_q8_loo_negative | .8270354398345265 | -.0014857735396989202 |

POINT_RESULTS.md records all8macro/task points andlatencies withpending-CI scope. No claim all macro CIsnegative until10337actuallycompletes. No posthoconly-buy/clicktargetswitch. Keep all16comparisons/signs.

## Immediate next steps

1. Poll live10337 until actuallyEXIT0. Never recreate paired folder/restart. Once complete, delegate independent saved-result review of smallJSON and500draw arrays only; no real bootstrap/forward repeat. Write reports/paired-result-independent-review.json with completion_sha256,quality_sha256,protocol_sha256 atminimum and exact21intervals percomparison.
2. Training/latency independent review is COMPLETE: reports/training-latency-independent-review.json SHA4fd6d3b8871308e1344e30bcfd1e5dc28ff97e6dad057825e9e46f9719893b0e,statepassed,blockers[]. Rootcheckedquality/latencybindings.185smallsource/artifacts checked,8fullbudgets,11samebatch/GPU/550forwards;all8retainfalse. NoCIorrealarraysread. Latencycompletionidentities legitimatelyaddsfeature_map.json beyondplan,withsamequality/commonhash;do notrequireexactplankeyset. Do notrepeatthisreview.
3. Run new `render_results.py --check`, inspect then `--write` after complete results. It emits RESULTS.md and reports/interpretation.json, no resampling/largearray/CP reads. It does not promoteincumbent. Independent staticreview already passed. Do not overwrite partialoutput blindly.
4. Rootprepare reports/memory-decisions.json fromactualreviewedresults, newscopednegativecoverage lesson(s) onlysupportedbyvalidoutcomes, explicit incumbentretain reason, meaningful nexthypothesisqueue. Bind current5memorybeforehashes and exact consolidator inputs. `consolidate_completed.py` defaults read-only; inspect its concrete readyreceipt, then `--apply` ONCE. Newcounts should133training/203episodes/203TSV. Rootsolememorywriter; preserveallhistoricalepisodes/TSVprefixandfailuresbytes.
5. Rootupdatecontroller/finalRESULTS/memory andclose currentround. Goalactiveuntilactuallyimprovedretrievalverified, notmerelyall8negative. No automaticstrengthsweep oroldprobe repeats.

## Memory authoritative

Globalmemory still125completedtraining/195episodes/195TSV/24lessons. Eight newlycompletedformaloutcomes are notyetledgered, waitingpaired interpretation. pending_coverage_retrieval.state=formal_and_latency_completed_paired_running,completed_training_runs8,training_completed_not_yet_in_ledger8,paired_tool_session10337. Root operational registrations reports/memory-posttraining-registration.json and memory-analysis-registration.json include full before/afterhashes andbackups;onlystrategychanged,episodes/TSV/failuresunchanged. Lastglobalepisode is Vdir_q8_centered_key. No Cepisode appended. Do notreuse Vconsolidator hardcodes.

## Fixed structure/protocol

C QFormerCross8CoverageRetrieval changes onlysecondcall of stage.layers[1].cross_attn SDPA: sign*tanh(valid_length*C-1),strength1,mean/leave-own-slot-outcoverage. P1reference=FP32maskedsoftmax(firstnormalizedQK*scale),autocastdisabled,allow_tf32false/highest;notfusedBF16hiddenP. NativeQ2/K2/V2/directinteractioninputretained;2workreads,no predictionfeedback,onefinalpostattention. FirstworkSDPAoutputunchanged,zero newparameters/buffers. q4/q8params51724228/81616324. Fullgradientcoverage,floatmaskmaychangebackend. Slotsarenottasks;coveredhistoryisnotnecessarilyirrelevant;noown-slot-onlysametransformcontrol.

Frozen2193updates,epoch1,batch8192,seed20262027,PSGDKron every-steppreconditioner beta1.9 denseLR1e-4,sparseAdagrad.05,clip1,BF16. Trainingfreeze SHA338681a79c6a7c68fcc52943b1c51d393b0095afac03e8391d46d5eba77d63d3;postprocessfreeze SHA7d9d18a7596bd382bd8e2c2db51a090ededf5a5bd277040ecbff77cac57906d8. No frozen training/numericalpostprocesssourcewasmodifiedwhiletraining.

## New closeout helpers / reviews

- render_results.py SHA20b08169244ae0a9db52917693fd97b1bb0ba75579511d463d179d144c207c6f.14temporarysyntheticchecks passed;authorreport render-results-offline.json SHA f1ae1594bbc21a7a16a044ff30b6c905a349b330296ff1d6fa967ca6b188bbdc preserves3invocations inclsystemPythonfailure,nativepass,actualpending. No rerunneeded.
- render-results-independent-review.json SHA524dd948c7b8a5e3cd8b57fd0670a1a66835ac1e42af711afd96a57f68a95628 passedstatic.
- consolidate_completed.py SHA0f1f102750b07d80c1d82cf9a4f1b4b64b75dc940bd3a1229851d4bb510fb6e9;memory-consolidator-static-review.json SHA28d79df1eea137adb369312f45feb1f3d29e6cb448d029e30cb711b0b5765266 passedstatic. No realapplyyet.
- Consolidator needs rootdecisions.state=reviewed_by_root,memory_before_sha256,result_sha256 mapping RESULT_NAMES plus RESULTS.md/protocol/model/consolidatorsource,next_question,new_lessons,incumbent{action:retain|replace,reason},phase,active_hypothesis,next_queue,exhausted_families_append. Supportsactualkeep/discardandconditionalpairs,noautopromotion. Checks review→completion→resultbinding andexactselection/order,allinputhashesbeforewrite. Backup/preparedexclusiveguards rejectblindpartialretry;multiplememoryfilesnotjointlyatomic. Oninterruptionreconcilebefore/afterhashes/rootexclusivewrite.

## Diagnostic followup / agent work

DIAGNOSTIC_FOLLOWUP.md checked actualrec-harness registry and memory. Handoffs directoryabsent;no readyedit. Oldsecond-read acceptedadapter lackscoveragebias andonlybindsoriginalmodel;cannotdirectlyreuse. Samecheckpoint native/format_zero/identity actualbiasintervention is conditionalproposal only,notdispatched/validated. Allnegativebranchdefaultsnoextraprobetounchangedretentiondecision.

NEXT_STRUCTURE_AUDIT.md is COMPLETE,SH A279caee1214fe7651cc8dae86dc264048fb284b20b639d2dbcb8b2c5419ebd77;rootreadit. KuaiRec alreadyhad8r13query→history/self-historywritebacks,allbelowr7,so generic H+=Attn(H,z1,z1)isnotnew. Taobaoearliermemory-source8onlyconcatenatedH/e/z1/z2,alsofailed. Two explicitlyuntrainedhypotheses: jointper-eventH/firstworkstateV-onlynonlinearresidual(add18432paramswithPSGDKronlayoutconcern);secondKitem-onlywithfullV(reuseexistingfieldprojectioncolumns,zero newparams). Noactualhistoricaltimestampsavailable,so do notinventtimegaps. No newpapers/proposalsimplyefficacy.

Rootconditionallyprioritizeszero-new-parameterfieldroles overnewparamVresidual,notyetqueued/dispatched. correction_route_probe is doing ONLY feasibility/source audit in FIELD_ROLE_FEASIBILITY.md: perq4/q8 fourpossibleroles Kitem/Vfull,Kaction/Vfull,Kfull/Vitem,Kfull/Vaction;firstread/Q2/w_qi/residual/twocycles/no-feedbackunchanged. Mustcheckvalidprojectioncolumns,bias/pos/LN,actionleakage/padding/nativeBF16andnondegeneracy. Eightmodelsiflaterchosen;no implementation/training/probeorprotocolfreezeauthorizedbythisdocumentitself. User'songoingresearchauthorizationpersists,butcompletecurrent16pairedandmemoryconsolidationbeforeselectingthenextround.

Latest10337poll stayedlive throughpair6/16,450/500draws at589.8s. Allq4parentfourpairresultsnowexist;otherintervalsremainpending. Controller/memoryactivegoalkeepcurrentcoverageuntilconsolidated. All source/runner/nativechecks remain frozenandterminal; no GPUworkleftthisround.
