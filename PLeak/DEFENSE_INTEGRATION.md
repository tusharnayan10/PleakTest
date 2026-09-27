# PLeak defense integration (`PleakD`)

PLeak remains the benchmark. The adapters change victim protection, not attack
optimization or leakage metrics. This is an implementation, **not a completed
GPU benchmark or a claim that either defense is effective**.

## Inspected code and scope

Base PLeak commit: `822a3edce94b34a82d965e0c67d40e81b2601ec1`.

| Responsibility | Existing implementation | Change |
| --- | --- | --- |
| AQ initialization, gradients, search, progressive loss | `Attack.py`: `HotFlip.init_triggers`, `compute_loss`, `hotflip_attack`, `replace_triggers` | None; file is byte-for-byte unchanged |
| Victim loading | `ModelFactory.py`: `get_model`, `get_tokenizer` | None; adapters share these exact objects |
| Victim generation | `Sampler.py`: `sample_sequence` | Route the two new names through adapters |
| Output defenses | `Defense.py`: `_creator`, `defend`, `filter_based` | None; `None`/`Filter` keep their existing path |
| Postprocessing | `Sampler.py`: `postprocess` | None; applies to the actual defended response |
| Metrics | `Sampler.py`: `evaluate`, normalization helpers | None |
| CSV schema/writer | `Sampler.py`: `save_to_csv` | None |
| CLI/results paths | `sample.py`, `main.py`, new `DefenseCLI.py` | Optional artifact manifest/output flags; defense-specific directories |

This checkout has no `Sandwich` implementation. It also has no victim-response
adaptive attack loop: HotFlip adapts to **shadow-model losses**, and sampling
happens afterward. No response-feedback attack or new query budget was invented.
`main.py --defense ...` protects evaluation after unchanged attack optimization.
Both entry points still use the original seeds (0), 1,000 requested test contexts,
AQ formatting, three beams, and the original Llama sampling settings. Original
model loading, including the unused extra model in `Defense.__init__`, is retained.

Fixes necessary to run the existing entry points: import `SentenceTransformer`
in `Sampler.py`; replace `main.py`'s filename expression containing undefined
`model`. Result paths now use `results/DEFENSE/DATASET_MODEL_AQHASH.csv` to avoid
cross-defense overwrites and arbitrary AQ characters in paths. `--output` allows
an explicit path. CSV columns remain `context` and the exact AQ string.

The original `substring`, `em`, `edit`, `semantic` evaluation calls and printed
formats are unchanged; the optional existing BLEU implementation is unchanged too.
There is no additional defense-specific leakage metric. New-defense runtime
failures stop the run instead of silently excluding trials.

## Official code reused

Bundled sources were checked against the official repositories:

- [Prompt Obfuscation](https://github.com/Dormant-Neurons/prompt_obfuscation),
  inspected upstream commit `e8e62c758ece2fc211952d86d935cde90d6b397a`.
  `generate_output.py:get_sys_prompt` loads the saved tensor;
  `src/prompt_utils.py:replace_sys_prompt_batch` and
  `update_attention_mask_batch` replace only the context span;
  `src/model.py:Model.get_embeddings` and `Model.generate_output` provide the
  embedding and generation paths. `prepare_defenses.py` invokes official
  `obfuscate.py` and `evaluate_obfuscation.py` to obtain `best_candidate.pt`.
- [PromptKeeper](https://github.com/SamuelGong/PromptKeeper), inspected upstream
  commit `c3b75286512f0d325e791bdea4897bbd5b3f9869`.
  `promptkeeper/defense.py:get_defended_model_response` runs the complete
  `mll_test_regen` decision and mitigation;
  `hypothesis_test.py:fit_distribution`, `get_likelihood_ratio`, and
  `get_mean_likelihood_conditioned_on_input` perform official calibration and
  scoring; `likelihood_test.py:get_likelihood_ratio_given_significance` computes
  the threshold. The mitigation in this source regenerates **without a system
  prompt**, and its actual response is returned to PLeak.

The upstream defense source changes are compatibility shims in
`src/model.py`: pass `mean_resizing=False` only when the installed Transformers
resize method supports it. Older versions retain their existing random
initialization. `return_legacy_cache=False` is also passed only when supported
by the loaded model generation configuration. No defense algorithm is modified. Scoped, locked backend bindings
adapt PromptKeeper's model access and prompt serialization to PLeak; they are
restored even on exceptions. Offline fit loading replaces its fixed relative
cache directory. Calibrations must be produced with this same PLeak backend;
stock chat-template fits are not interchangeable.

Obfuscation deliberately retains PLeak's plain concatenation rather than adding
upstream chat-template delimiters. It reuses the official replacement and
embedding-generation methods, with explicit PLeak context-span indices. It does
not use the upstream span finder, which assumes a chat template and added pad
vocabulary. The original query tokens and BOS are preserved. A context/query
boundary merged by the tokenizer is rejected explicitly rather than silently
retokenizing the attack. The response allowance stays `original_input_length +
length` (`length=50`), even when the protected prefix changes length.

**Comparability limit:** the official obfuscation optimizer trains with its own
chat template, quantization settings, and added pad vocabulary. Applying its
saved tensors to PLeak's unchanged formatting/model loading is a transfer
setting; it is not identical to the paper's native serving setup. Hard tokens
outside PLeak's original vocabulary are rejected. Soft tensors never become
text. Among PLeak's existing aliases, official artifact preparation supports
`llama-chat`; it never silently switches `llama` or another alias to that model.
Keep the same alias across every defense in a comparison. Defense preparation
and PromptKeeper's internal scoring/regeneration incur extra work, but introduce
no extra attacker queries.

## Environment

Use a CUDA machine with access to the selected Hugging Face model and sufficient
memory for PLeak's existing model loading. Keep **one identical dependency
environment** for all defense comparisons. Original PLeak documents Transformers
4.32.1, whereas the bundled obfuscation code was written for 4.52.2 / PyTorch 2.7.0.
Use a separate environment from historical experiments and rerun `None` there;
results across library versions should not be treated as bit-identical.

A starting dependency set (not GPU-validated here):

```bash
python -m pip install torch==2.7.0 transformers==4.52.2 \
  torchmetrics==0.11.4 torcheval sentence-transformers==4.1.0 \
  datasets==3.6.0 bitsandbytes==0.45.5 accelerate==1.7.0 \
  sentencepiece numpy scipy nltk evaluate pandas munch openai \
  google-generativeai pyyaml matplotlib nvidia-ml-py tqdm \
  rouge_score sacrebleu bert-score textdistance Levenshtein cer
```

See each bundled defense's requirements for additional optional preparation
metrics. Dataset availability and Hugging Face credentials remain the original
PLeak responsibilities. Run commands from `PLeak/` because its datasets refer to
relative `util/` paths. No model weights or trained artifacts are included in
this change.

## Prepare artifacts separately

Every original PLeak context needs its own matching artifact. The precomputed
pirate/robot prompts in the repository do **not** protect unrelated PLeak contexts.
The preparation script reproduces test dataset construction with seed 0 and
writes a manifest keyed by SHA256 of the exact UTF-8 context (including whitespace).
Preparation for 1,000 contexts can be expensive; use `--limit 1` for a smoke check.

```bash
cd PLeak
python prepare_defenses.py Roles llama-chat PromptObfuscation \
  --method soft --output-dir artifacts/obf-soft
# For hard prompts, use --method hard and a separate directory.
python prepare_defenses.py Roles llama-chat PromptKeeper \
  --output-dir artifacts/keeper
```

For evaluation via `main.py`, add `--training-size 16` (matching its final
positional argument) when preparing artifacts, since building the training set
consumes Python RNG state before constructing the test set. Keep separate
artifact directories for sample.py and main.py datasets. The official obfuscation
preparation uses its own default training seed, separate from PLeak evaluation.

Manifests have this shape (paths are relative to the manifest):

```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "significance": 0.05,
  "contexts": {
    "SHA256_OF_EXACT_CONTEXT": {
      "params": "CONTEXT_HASH/params.json",
      "tensor": "CONTEXT_HASH/best_candidate.pt"
    }
  }
}
```

For PromptKeeper, each context entry instead contains:

```json
{"fit": "CONTEXT_HASH/fit_result.pkl", "format": "pleak-concat-v1"}
```

Only use trusted locally prepared PromptKeeper pickle files. Missing context,
wrong model/prompt, incompatible tensor dimensions/vocabulary, invalid fits, or
failed threshold solving produce explicit errors. Degenerate calibrations need
more samples in a **fresh output directory**; upstream caches existing fits.

## Run the same AQ and metrics

Use the same AQ generated by unchanged `main.py` or the same fixed AQ for every
run. For example, replace this placeholder with your exact original AQ:

```bash
AQ='YOUR_UNCHANGED_PLEAK_AQ'
python sample.py Roles llama-chat None "$AQ"
python sample.py Roles llama-chat Filter "$AQ"
python sample.py Roles llama-chat PromptObfuscation "$AQ" \
  --defense-config artifacts/obf-soft/manifest.json
python sample.py Roles llama-chat PromptKeeper "$AQ" \
  --defense-config artifacts/keeper/manifest.json
```

Or retain the original attack-generation entry point:

```bash
python main.py Roles 12 llama-chat llama-chat 16 \
  --defense PromptKeeper --defense-config artifacts/keeper-main/manifest.json
```

Do not include a `Sandwich` row in a comparison until an actual implementation
and configuration are supplied; this branch rejects unsupported names.

## Quick sanity checks

Dependency-free checks from the repository root:

```bash
python3 -m unittest discover -s PLeak/tests -v
python3 -m compileall -q PLeak
```

These check unchanged attack/metrics/CSV methods, legacy CLI parsing, manifest
validation, original `None` generation, defended response routing, the actual
upstream PromptKeeper safe/leaking/equality branches, and official tensor
replacement functions for hard and soft inputs. Model/likelihood boundaries use
stubs; they do not establish numerical GPU correctness or defense effectiveness.

For a real-model check, prepare the first context for each defense with the same
commands above plus `--limit 1`, then run from `PLeak/`:

```bash
python smoke_defenses.py Roles llama-chat None "$AQ"
python smoke_defenses.py Roles llama-chat PromptObfuscation "$AQ" \
  --defense-config artifacts/obf-soft/manifest.json
python smoke_defenses.py Roles llama-chat PromptKeeper "$AQ" \
  --defense-config artifacts/keeper/manifest.json
```

The smoke tool builds the original 1,000-context dataset and tests its first
context only. It invokes the original four evaluation methods. One-sample
standard deviation may be NaN under the original metric implementation. A
partial manifest is intentionally insufficient for full `sample.py`; prepare all
contexts before a full comparison. Full inference was not run in the editing
environment, which lacks PyTorch, CUDA weights, and context-specific artifacts.

## Update an existing checkout

From the downloaded `PleakD` branch, run:

```bash
bash update_pleak.sh /path/to/PleakTest
```

The script clones `PleakD` if the destination is absent, or fetches and switches
a clean checkout to that branch, allowing only a fast-forward. It refuses dirty
checkouts and unexpected origins; it does not reset or discard work.

### Older Transformers: `mean_resizing` error

Pull the latest `PleakD` update and rerun preparation. The compatibility shim
omits the unsupported resize keyword on older versions. Preparation now checks
that each subprocess actually produced its required files, even when upstream
logs an error and exits with status zero. An incomplete run missing `params.json`
or its candidate tensors is retried; no artifact-directory deletion is needed.
This fixes the reported resize error, not a guarantee that every other API in
the bundled defense supports historical Transformers versions.

### Older Transformers: `return_legacy_cache` error

The generation bridge now omits this cache-format option when it is absent from
the model generation configuration. Logits, decoding settings, and the
obfuscation objective are unchanged. Preparation also passes the official
`--task_hints` flag: task instructions go into obfuscation-training user queries
instead of being prepended to the protected PLeak context. Existing artifacts
with extra system instructions are rejected; retrain those in a new directory.
Failed runs with no params/candidate artifacts can simply be retried.

### CPU Adam: `addcdiv_cpu_out` not implemented for Half

The official embedding accessor returns CPU tensors in the victim's dtype. On
older PyTorch, Adam cannot update these CPU float16 tensors. Soft obfuscation
now keeps its trainable prompt and Adam state in float32, and casts the prompt
back to the input-embedding dtype for each differentiable replacement. The
victim weights/dtype, objective, learning rate, epsilon, and iteration schedule
remain unchanged. This is an optimizer-precision fix in `obfuscate.py`; it does
not change PLeak. The failed run did not save precomputed outputs, so retrying
will repeat precomputation.

### Older sentence-transformers: missing `SimilarityFunction`

The obfuscation candidate-selection cosine metric uses the longstanding
`sentence_transformers.util.cos_sim` API with the same `all-mpnet-base-v2`
encoder, all-pairs cosine matrix, and averaging. It no longer imports the newer
`SimilarityFunction` enum or calls `model.similarity`. The standard encoder is
loaded without the newer `trust_remote_code` constructor argument. No PLeak
metric is modified, and saved obfuscation candidates can be reused.

### Saved soft-prompt evaluation dtype

Both official replacement-generation paths (`generate_model_responses_replace`
and `precompute_model_outputs_replace`) cast a local copy of the saved prompt
to the victim input embeddings' dtype/device before concatenation. This permits
evaluation of FP32 optimizer checkpoints with FP16/BF16 models without changing
the saved tensors or promoting model inputs to FP32. Hard token IDs are unchanged.
Existing saved candidates can be evaluated without repeating training.
