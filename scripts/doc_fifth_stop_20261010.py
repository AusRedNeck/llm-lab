"""Append the FIFTH STOP + the honest verdict to the p160-pile-parity-full arm note."""
import json
import os

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "experiments.json") if False else "D:/Projects/llm-lab/experiments.json"

ADD = (
    " ||| FIFTH STOP = A TRUE POSITIVE, AND THE ARM DID NOT ANSWER ITS QUESTION "
    "(2026-10-10 ~09:55, step 12,500/24,694 = 50.6%). Stopped by the SATURATION guard "
    "again -- 'flat stop: per-round envelope stalled at bpb=1.5788 for 25% of the "
    "schedule (1-eval cycle x 0.005 bpb); best=1.3080' -- but this time the firing is "
    "correct and the two-eval persistence fix (030b762) is what validated it: peaks AND "
    "troughs both climbed for 3,400 steps with no new best after 1.3080@9100 "
    "(troughs 1.4242@11300 -> 1.4571@11800 -> 1.4857@12300; peaks 1.5128@11400 -> "
    "1.5494@11900 -> 1.5788@12500, last two evals 1.5767/1.5788 in agreement). That is "
    "the ROADMAP's documented 'val bottoms 1300-2400 then climbs while train descends' "
    "divergence, not a spike: lr was still 2.73e-4 (over half peak) at the moment val "
    "hit 1.5788. First guard firing all run that deserves to stand. "
    "VERDICT, stated so nobody quotes the flattering version: best_bpb 1.3080 is NOT a "
    "claimable win against the 1.4614 bar, for three independently verified reasons. "
    "(a) SEED-0 REPLAY INFLATES IT: step 9100 measured 1.41535 in the segment resumed "
    "from 8500 and 1.30800 in the segment resumed from 9000 -- same step, 0.107 apart, "
    "because a resume resets torch.manual_seed(0) so eval N draws the windows a fresh "
    "run had at step N-minus-resume-point; post-resume evals land 0.15-0.20 optimistic "
    "(recorded 2026-10-08) and the best checkpoint IS a post-resume number. (b) TWO "
    "INSTRUMENTS: 1.4614 came from the HF reference harness (reports/"
    "pythia160m_step1000_reference.json: seed 1234, 60 batches, 245,760 non-overlapping "
    "tokens) while 1.3080 came from the trainer's in-loop eval_three_way (20 random "
    "batches off the global RNG); scripts/score_ours_vs_reference.py says outright "
    "'CANNOT SETTLE whether our trainer's in-loop val_bpb matches this harness -- quote "
    "them as two instruments, not one'. (c) TOKEN MISMATCH: 12,500 steps = 1.638B "
    "tokens vs pythia@step1000's 2.097B = 78%, and the anchor (16,000) was never "
    "reached, so no at-anchor measurement exists. The 70M got its '6.0% ahead' verdict "
    "only after both sides went through the SAME harness at matched tokens. Even the "
    "pessimistic reading (~1.42) is in the bar's neighbourhood -- neighbourhood is not "
    "parity, and the run was diverging, so more steps at 5e-4 could not have closed it. "
    "THE ACTUAL FINDING: THE LR PROBE SELECTED ON TOO SHORT A HORIZON. The probe gave "
    "5e-4 its 0.159 bpb win by running its OWN 2,000-step cosine to 10%, so by step "
    "2,000 the probe's lr was already 5e-5 -- it never tested 5e-4 SUSTAINED at high lr "
    "for thousands of steps, which is precisely where this run diverged (val peaked "
    "1.3797@2300 while the full run's 24,694-step cosine still had lr ~4.9e-4 there). "
    "'Compare final 4-eval block means' cannot detect divergence appearing at 5,000+. "
    "5e-4 won a sprint and lost the marathon; three separate segments say it diverges by "
    "~9,000. DO NOT relaunch at 5e-4. "
    "HUMAN COST, on the record: between the four FALSE stops and this run's end, the GPU "
    "sat dead ~8.5h then ~22h = ~30h idle on a job whose entire plan was ~30h. Roughly "
    "half the wall-clock was a stopped run nobody reported -- guards killed it four times "
    "wrongly, and the COS cron delivery was broken (403 then standalone_sender_fn) so "
    "every death went unreported. Both halves fixed 2026-10-10: delivery now goes "
    "write-file + relay with real message ids, and cron_health_guard.py gained a check "
    "for enabled jobs whose last_status is delivery_failed -- delivery failures leave "
    "failure_streak at 0 and never reach cron_incidents, so the old guard was blind to "
    "the one failure mode destroying output. "
    "KEPT: checkpoints/exp002_pythia160_tokenizer_202610072312_best.pt (1.95GB, the "
    "1.3080 model, keeper-protected). NEXT DECISION (Shane's): (1) score that checkpoint "
    "through score_ours_vs_reference.py to settle parity with one harness -- cheap, "
    "caveated as 78% tokens; or (2) re-bracket lr at 3e-4/2e-4 on a 6,000-step horizon "
    "long enough to see the divergence the 2,000-step probe structurally cannot. "
    "Recommendation: (1) first -- it converts 'neighbourhood' into a real number. "
    "GPU deliberately left untouched 2026-10-10 while Shane was sim racing; scoring must "
    "not share the card with iRacing."
)

with open(EXP, encoding="utf-8") as f:
    data = json.load(f)

arms = data["arms"] if isinstance(data, dict) and "arms" in data else data
lst = arms if isinstance(arms, list) else list(arms.values())
target = next(a for a in lst if a.get("id") == "p160-pile-parity-full")

if "FIFTH STOP" in target.get("note", ""):
    print("already documented, skipping")
else:
    target["note"] = target["note"].rstrip() + ADD
    target["verdict"] = "stopped at 50.6% (step 12500/24694); saturation guard TRUE positive; parity NOT established - best_bpb 1.3080 not comparable to the 1.4614 bar (replay inflation + two instruments + 78% tokens). LR probe selected on too short a horizon."
    with open(EXP, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("appended; new note length:", len(target["note"]))
    print("verdict:", target["verdict"][:80], "...")
