## GTFORMER: LONG-CONTEXT EVALUATION SUITE

=========================================


Official PyTorch implementation and evaluation suite for the GraphTree architecture. 

This repository benchmarks scale-invariant hierarchical graph message-passing against 

tree-based ablations and flat Transformer baselines across 8 synthetic long-context 

reasoning tasks, evaluating zero-shot length extrapolation from N <= 512 up to 

N = 4,194,304 (8192x OOD extrapolation).





## KEY FEATURES

------------

- 100% Vectorized Data Generation: On-the-fly GPU batch generation for 8 non-mathematical 

  sequence reasoning tasks with zero disk I/O overhead.

- 7 Model Architectures: Complete ablation suite comparing bidirectional GraphTree against 

  bottom-up variants, pure binary trees, unbounded MLPs, and Sinusoidal/RoPE Transformers.

- Extreme OOD Extrapolation: Automated evaluation across 13 sequence lengths 

  (N in [64 ... 4,194,304]) with FP16 autocast, latency tracking (ms/seq), and peak VRAM monitoring.

- Fault-Tolerant Orchestration: Automatic checkpointing and resumption via results_incremental.json, 

  alongside real-time export of CSV summaries and publication-ready LaTeX booktabs tables.





## REQUIREMENTS & INSTALLATION


The benchmark suite relies strictly on standard Python libraries and PyTorch.



  pip install torch





## QUICK START



### 1. Run the Full Multi-Tier Evaluation Suite:

   By default, the script runs a prioritized 4-tier experiment queue for up to 18 hours, 

   automatically saving progress after every model evaluation:



   python test_suite.py --target_hours 18.0 --out_dir ./paper_results



### 2. Fast Verification / Lower VRAM Run:

   To run a quick test or cap maximum sequence length to fit smaller GPUs (e.g., capping at N = 131,072):



   python test_suite.py \

       --target_hours 2.0 \

       --steps 1000 \

       --base_seeds 1 \

       --max_eval_len 131072 \

       --out_dir ./quick_results



### 3. Resuming an Interrupted Run:

   If a run is interrupted or hits its time budget, re-run the exact same command pointing 

   to the same --out_dir. The script reads results_incremental.json and skips completed jobs automatically.





## CLI ARGUMENTS REFERENCE

-----------------------

--target_hours  : float (default: 18.0) - Maximum wall-clock runtime budget in hours.

--steps         : int   (default: 2200) - Training steps per individual model run.

--base_seeds    : int   (default: 5)    - Number of initial random seeds per configuration in Tier 1.

--max_eval_len  : int   (default: 4194304) - Maximum sequence length N evaluated during OOD testing.

--out_dir       : str   (default: ./paper_results) - Output directory for JSON checkpoints, CSVs, and LaTeX tables.





## BENCHMARK TASKS (8 NON-MATH TASKS)

----------------------------------

All tasks are formulated over a unified 64-class exact-match output space (4^3 joint combinations across 3 output slots):



1. override_qa        : Chronological multi-register state tracking under random register arrival orders.

2. unaligned_kv       : Multi-token [KEY_r, VAL_v] binding placed at odd binary-tree boundaries (2m+1, 2m+2).

3. pointer_hop        : 2-hop relational pointer dereferencing via [VAR_k, VAL_v] and [PTR_p, VAR_k] pairs.

4. intrusion_fsm      : Security log finite-state tracking with mid-stream GLOBAL_RESET events.

5. code_scope_shadow  : Lexical block scoping where inner-scope overrides either revert (SCOPE_EXIT) or persist (SCOPE_COMMIT).

6. multi_needle_rag   : Multi-needle fact synthesis joining scattered entity attributes [ENT_e, ATTR_v] with queries [QRY_p, ENT_e].

7. xml_bracket_check  : Hierarchical nested tag closure and corruption detection across 3 independent subsystems.

8. adversarial_decoys : Unaligned key-value recall under high-density syntactic decoy noise ([DECOY_KEY_r, VAL_v]).





## EVALUATED ARCHITECTURES (7 MODEL VARIANTS)

------------------------------------------

1. GraphTree_Bidirectional     : Full model with bottom-up reduction and top-down root broadcast.

2. GraphTree_BottomUp          : Bottom-up hierarchical reduction with local shift and expander edges.

3. GraphTree_NoExpander        : Ablates cross-subtree horizontal expander shortcuts.

4. Pure_Binary_Tree            : Standard binary tree GNN without horizontal shift or expander edges.

5. Unbounded_Tree_MLP          : Replaces shared-salience convex gating with an unconstrained MLP merge.

6. Flat_Transformer_Sinusoidal : 2-layer Transformer Encoder with sinusoidal positional encodings and mean pooling.

7. Flat_Transformer_RoPE       : 2-layer Transformer Encoder with Rotary Position Embeddings and attention pooling.





## EXPERIMENTAL TIERS

------------------

The orchestrator executes runs in four prioritized tiers:



- Tier 1 (tier1_main): Full evaluation matrix across all 8 tasks x 7 architectures x base_seeds (d_model = 128, N_train <= 512).

- Tier 2 (tier2_width_scaling): Width and capacity scaling laws sweeping d_model in [64, 96, 192, 256] across core relational tasks.

- Tier 3 (tier3_train_horizon): Training context horizon ablation sweeping N_train in [64, 128, 256, 1024].

- Tier 4 (tier4_deep_seeds): Deep seed expansion (up to 25 seeds per task/architecture pair) for high-confidence interval estimation.





## OUTPUT ARTIFACTS

----------------

After every completed model run, the suite updates four files in --out_dir:



- results_incremental.json : Raw per-seed metrics, training durations, latencies, and VRAM usage (used for checkpoint resumption).

- results_summary.csv      : Aggregated mean/std accuracy, latency (ms), and peak VRAM (MB) for the main evaluation matrix.

- scaling_laws_summary.csv : Aggregated metrics for Tier 2 (width scaling) and Tier 3 (training horizon) ablations.

- paper_tables.tex         : Auto-generated, publication-ready LaTeX table* environments formatted with booktabs.





## LICENSE

-------

This project is licensed under the GNU Affero General Public License (AGPL-3.0).
