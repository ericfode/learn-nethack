# Fitness Objective Analysis

Date: 2026-06-18

Scope: choose the next useful fitness function for the current `learn-nethack`
pipeline. This is an objective-design note, not an implementation plan.

## Current State

The pipeline can now build SFT rows, train LoRA adapters on Modal, evaluate
policy and next-frame rows, run watch comparisons, and log reports locally plus
to W&B. The important empirical constraint is that current adapters can improve
some offline or teacher-forced metrics without producing better live NetHack
behavior. Recent watch artifacts show repeated wall actions, action collapse,
zero reward, and no depth progress in short live rollouts.

Therefore the primary objective must be live-environment anchored. Offline
losses remain useful as auxiliary objectives and gates, not as the definition of
success.

## Research Constraints

NetHack is procedurally generated, stochastic, entity-rich, partially
observable, and long-horizon. NLE was designed as a fast RL environment for
exploration, planning, skill acquisition, and generalization research. The
NeurIPS 2021 NetHack Challenge used ascensions, median score, and mean score as
the official ranking tuple, but the report also notes that in-game score is an
imperfect proxy for ascension progress. Challenge analysis found strong role
variance, top-level restriction, starvation deaths, and a large gap between
symbolic and neural agents. NLD adds human and symbolic-bot trajectory data, but
the dataset paper still treats human-level NetHack as open, and explicitly
frames progress as requiring offline data plus environment interaction.

Local implication: score-only reward is too sparse and too exploitable for this
repo's next step. The objective needs dense anti-failure terms, but those terms
must be measured from NLE observations and reports rather than from hand-rolled
NetHack legality.

## Weighted Options

| Rank | Objective option | Utility weight | What it trains | Main strength | Main defect |
| --- | --- | ---: | --- | --- | --- |
| 1 | Live rollout utility composite | 34 | Policies that score, survive, descend, and avoid known bad loops | Best alignment with watchable NetHack behavior | Needs careful normalization and enough seeds |
| 2 | Demonstration likelihood plus preference ranking | 22 | Policies that imitate true keypress or strong-bot actions, with pairwise preference for better outcomes | Dense, stable, available before strong RL | Can clone weak or local behavior; pseudo labels are narrow |
| 3 | Exploration and skill-acquisition curriculum | 18 | Policies that leave start rooms, find stairs, pick/eat/use, fight safely, and recover from prompts | Attacks sparse-reward and hierarchy problem directly | Requires task definitions and stage gates |
| 4 | Dynamics/world-model objective | 14 | Models that predict NetHack transitions and support planning or consistency checks | Improves state understanding and auditability | Generated-frame validity is not live competence |
| 5 | Pure NLE score/reward objective | 12 | Direct score maximization | Canonical and externally comparable | Sparse, high variance, exploitable, misses role/failure modes |

Weights sum to 100. They are utility weights for the next training objective,
not permanent scientific importance. Pure score matters more later, after the
agent can move, recover, and avoid obvious traps.

## Recommended Primary Objective

Use a live rollout utility composite as the scalar RL/update fitness:

```text
U_live =
  + 1.00 * normalized_score_delta
  + 0.25 * normalized_cumulative_reward
  + 3.00 * depth_delta
  + 0.10 * min(meaningful_event_count, 5)
  + 0.50 * clean_live_progress_event_count
  - 0.50 * hp_damage
  - 8.00 * death
  - 1.50 * starvation_or_faint
  - 0.10 * wall_or_solid_stone_message_count
  - 2.50 * wall_message_rate
  - 0.10 * bad_message_count
  - 1.50 * bad_message_rate
  - 0.20 * non_advancing_step_count
  - 1.00 * non_advancing_step_rate
  - 0.05 * action_collapse_excess
  - 2.00 * max(0, action_repeat_rate - 0.50)
  - 1.50 * menu_or_prompt_step_rate
  - 0.50 * stuck_menu_or_prompt_loop_count
  - 1.00 * dirty_live_progress_event_count
  - 3.00 * zero_progress_episode
```

Implementation note: after reading the 2026-06-17 ML-analysis evidence, the
2026-06-18 watch/preference artifacts, and the follow-up concern that the
fitness was still too proxy-friendly, this is tracked as
`live_rollout_utility_v7`. The v6 change kept v5's wall-message,
action-collapse, bad-message, prompt/menu, true score/reward/depth, and
zero-progress penalties, then removed visible-map novelty from the scalar
fitness entirely. V7 keeps that removal and further requires live-progress
bonuses to be clean: a progress event earns the live-progress bonus only when it
is not a wall, menu/prompt, bad-message, non-advancing, or death event. Dirty
progress remains logged as real reward/score/depth movement, but it is taxed by
`dirty_live_progress_event_count` rather than treated as high-quality policy
evidence. V7 also keeps the v6 zero-progress episode penalty of `-3.00` and the
proof-gating rule that a run must show absolute current score, reward, or depth
progress, not merely a favorable delta against a weak baseline.

Implementation hardening: v7 scoring must inspect rendered terminal frames, not
only NLE `message` strings. Some bad states in the watch artifacts show up as
`--More--`, extended-command pages, or solid-stone/wall text inside the rendered
frame while the message field is empty or `<missing>`. Watch scoring and
preference scoring therefore treat frame-visible wall/menu/prompt screens as bad
signals.

Preference hardening: map novelty is not by itself a high-quality positive
label. Preference rows may use novelty as a small utility component, but a
chosen action is only positive training evidence when it produces reward, score,
or depth progress, or when it cleanly avoids a bad rejected action. Do not train
preference stages from two merely different zero-progress actions. A clean
zero-progress action may be preferred over a wall/menu failure, but it must still
carry negative absolute utility.

Proof hardening: relative improvement is not enough. A trained policy must also
satisfy absolute current-policy ceilings for wall-message rate, bad-message
rate, non-advancing rate, action-repeat rate, menu/prompt rate, and
zero-progress episodes, plus a positive absolute v7 fitness score, no dirty
progress regression, and absolute current score/reward/depth progress. This
prevents the gate from certifying "less bad than a bad baseline" or "progress
inside a broken interaction loop" as useful NetHack progress.

Use this under a KL penalty to the best current SFT/reference policy:

```text
L_rl = -advantage(U_live) * log pi_theta(a_t | obs_t)
       + beta_kl * KL(pi_theta || pi_ref)
       - entropy_coef * H(pi_theta)
```

The policy still scores exact JSON candidates only:

```json
{"action_id": 7}
```

Do not use next-frame task outputs for action selection.

## Required Evaluation Ledger

Keep the scalar objective small enough to train, but never evaluate with only
the scalar. Every RL/eval report should continue to log:

- cumulative reward and in-game score
- max depth and staircase events
- HP damage, death, death cause
- hunger/starvation/faint counters
- wall or solid-stone messages
- non-advancing steps and prompt/menu loops
- action histogram, action repeat rate, action collapse excess
- role/race/alignment breakdown
- terminal frames and replay artifacts

The scalar trains. The ledger prevents self-deception.

## Phased Use

Phase 1: make the live composite the watch/RL score-to-beat metric. Use 16-64
short seeded episodes, report medians by role, and require improvement over base
Gemma and the current best adapter. The proof gate should reject smaller sweeps
as smoke evidence rather than scientific proof.

Phase 2: train with composite REINFORCE/KL, while keeping supervised
demonstration loss as a small anchor. Do not let pseudo labels dominate the
policy anchor; true keypress data gets higher trust.

Phase 3: add skill-curriculum heads or tags only after the composite shows basic
movement and recovery. Early skills should be low-level and auditable: leave
room, avoid wall loop, attack adjacent weak monster, pick up food, eat when
hungry, use stairs.

Phase 4: use dynamics loss as auxiliary representation pressure and planning
diagnostic. It should remain evaluated against ground truth next frames and
never be treated as proof of reachable generated game states.

Phase 5: increase the pure-score weight once the agent beats baseline on
movement, loop avoidance, survival, and shallow-depth progress.

## Decision

Adopt the composite live rollout utility as the primary fitness function now.
Use demonstration likelihood/preference ranking as the main auxiliary objective,
exploration/skill curriculum as the next curriculum layer, dynamics as an
auxiliary model-understanding objective, and pure NLE score as the canonical
external evaluation target rather than the sole early training reward.

This objective matches the current evidence: the immediate problem is not JSON
validity or teacher-forced likelihood. It is live behavior collapse.

## Sources

- NLE repository: https://github.com/facebookresearch/nle
- NLE paper: https://arxiv.org/pdf/2006.13760
- NetHack Challenge analysis: https://arxiv.org/pdf/2203.11889
- NLD paper: https://arxiv.org/pdf/2211.00539
- RND exploration paper: https://arxiv.org/abs/1810.12894
