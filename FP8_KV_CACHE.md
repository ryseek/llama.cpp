# Experimental FP8 KV Cache for CUDA SM120

## Status and scope

This private-fork implementation adds calibration-free FP8 KV cache modes for NVIDIA Blackwell SM120 GPUs. It was developed and validated on an RTX 5090 with CUDA 13.3 and a `120a-real` build.

"Native FP8" in this document refers to the persistent KV cache storage format. The current MXFP8 attention path reconstructs a logical F16 K/V cache before invoking the existing F16 Flash Attention kernel. It does not yet execute production attention directly with Blackwell FP8 tensor-core instructions.

The default KV cache behavior is unchanged unless an MXFP8 mode is explicitly selected.

## Cache formats and modes

| Mode | Payload | Scale | Layout |
|---|---|---|---|
| `default` with `f8_e4m3` | E4M3 | F32 per token and head | Row-scaled FP8 |
| `mxfp8` | E4M3 | UE8M0 per 32 values | Pure MXFP8-32 |
| `mxfp8-hybrid` | Cold E4M3, hot F16 | UE8M0 per 32 cold values | True-replacement hybrid |

Each block scale is selected dynamically from the block maximum absolute value. No calibration dataset, calibration pass, or stored calibration table is required.

The MXFP8 tensors use a structure-of-arrays layout:

- E4M3 payload: `[head_dim, cache_row, kv_head, stream]`
- UE8M0 scales: `[head_dim / 32, kv_head, cache_row, stream]`
- F16 hot payload: `[head_dim, hot_row, kv_head, stream]`

The scale tensors use raw one-byte UE8M0 values. A scale byte `0x7f` represents 1.0.

## Hybrid cache architecture

The hybrid mode stores recent K/V rows in F16 and older rows in MXFP8. It is a replacement design, not a shadow cache: a row exists in either the F16 hot tier or the MXFP8 cold tier.

The current policy is:

```text
hot_size  = context_size / 32
sink_size = 0
cold_size = context_size - hot_size
```

This produces a 4096-token F16 hot tier at 128K context and a 1024-token tier at 32K context.

For each append:

1. Resolve the destination F16 ring slot from the logical token position.
2. If the slot already contains a row that is leaving the hot window, quantize that F16 row into its MXFP8 cold row.
3. Store the incoming K or V row in the F16 slot.
4. Preserve logical token order for the attention mask and graph.

At full capacity, the approximate storage per value is:

| Format | Bytes per value |
|---|---:|
| F16 | 2.00000 |
| Pure MXFP8-32 | 1.03125 |
| MXFP8 hybrid with `hot_size = N / 32` | 1.06152 |
| Q8_0 | 1.06250 |

For the tested Qwen3.8-27B model at 128K context, total attention KV storage is 4348 MiB for hybrid MXFP8, 4352 MiB for Q8_0, and 8192 MiB for F16.

## CUDA write and attention paths

The CUDA implementation supports K/V head widths of 128 and 256.

### Write path

- Pure MXFP8 uses a block-32 E4M3 writer with one UE8M0 scale per block.
- Hybrid mode fuses F16 victim demotion and incoming F16 hot-row storage.
- Programmatic dependent launch synchronization protects cache and sidecar dependencies.
- Side-effecting scaled writes are excluded from generic RoPE and RMS-Norm/RoPE fusion paths.
- Unsupported CUDA builds and CPU backends reject the custom operations instead of falling back to generic `SET_ROWS` handling.

### Read path

The current hybrid attention bridge:

1. Allocates a padded logical F16 K/V scratch region.
2. Decodes the active cold MXFP8 prefix into logical positions.
3. Overlays the F16 hot rows in logical order.
4. Leaves padded rows zeroed and applies the existing attention mask.
5. Runs the mature F16 Flash Attention implementation.

This is a correctness and quality bridge. It preserves the optimized F16 attention kernel but adds global-memory traffic for the reconstructed cache.

## Build

Use a CUDA toolkit with SM120a support. The validated configuration used CUDA 13.3:

```bash
cmake -S . -B build-fp8 \
    -DGGML_CUDA=ON \
    -DGGML_FLASH_ATTN=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120a-real \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build-fp8 -j
```

## Usage

Pure MXFP8-32:

```bash
./build-fp8/bin/llama-cli \
    -m /path/to/model.gguf \
    -c 131072 -ngl 999 -sm none -fa on \
    -ctk f8_e4m3 -ctv f8_e4m3 \
    --kv-cache-mode mxfp8
```

MXFP8 plus an F16 hot tier:

```bash
./build-fp8/bin/llama-cli \
    -m /path/to/model.gguf \
    -c 131072 -ngl 999 -sm none -fa on \
    -ctk f8_e4m3 -ctv f8_e4m3 \
    --kv-cache-mode mxfp8-hybrid
```

Both MX modes require K and V to use `f8_e4m3`. Explicit MX modes fail instead of silently selecting another cache format.

## Current constraints

The initial implementation deliberately uses strict guardrails:

- CUDA SM120 only; validated on RTX 5090.
- One CUDA device, with every attention layer offloaded.
- Flash Attention enabled.
- Matching K and V head widths of 128 or 256.
- No MLA, DSV4, custom sparse cache, or shared assistant cache layout.
- No sliding-window attention in hybrid mode.
- One sequence in hybrid mode.
- Monotonic, contiguous, append-only logical positions in hybrid mode.
- No hybrid cache sharing, sequence copies, position shifts, partial removals, recurrent rollback, or MTP context.
- No hybrid state serialization or restoration.
- Full cache clear is supported and resets hybrid append state.
- Pure `mxfp8` state serialization includes the block-scale sidecars.

Unsupported operations are rejected rather than approximated.

## Verification

Validation hardware and workload:

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0
- CUDA: 13.3
- Build architecture: `120a-real`
- Model: `Qwen3.8-27B-UD-Q4_K_XL.gguf`
- Dataset: WikiText-2 raw test split

Completed checks:

- Release CUDA build.
- CUDA-disabled CPU build.
- Scalar E4M3 and UE8M0 reference tests.
- D=128 and D=256 block-scaled writes.
- Q=1, Q=2, and Q=8 attention paths.
- GQA ratio 6.
- `H-1`, `H`, and `H+1` hot-to-cold transitions.
- Padded logical widths 255, 256, and 257.
- Persistent two-compute prefix and nonzero-base append.
- Byte-exact hot, cold, and scale verification.
- CUDA Compute Sanitizer memcheck with zero errors.
- Real-model prompt-processing and token-generation smoke tests.

## Quality results

One WikiText-2 chunk at 128K context:

| KV cache | PPL, lower is better | KV memory | Inference pass time |
|---|---:|---:|---:|
| F16 | 5.2353 | 8192 MiB | 88.05 s |
| Q8_0 | 5.2367 | 4352 MiB | 91.02 s |
| MXFP8 plus F16 hot | 5.2361 | 4348 MiB | 79.18 s |

The PPL estimates were printed to four decimal places and came from one corpus chunk. They show the recorded run, not a universal ranking across models or datasets.

Paired 32K comparison against an F16-cache logits baseline:

| KV cache | Mean KL, lower is better | RMS probability error | Top-1 agreement |
|---|---:|---:|---:|
| Q8_0 | 0.001716 | 1.179% | 97.949% |
| Pure MXFP8 | 0.001911 | 1.183% | 97.784% |
| MXFP8 plus F16 hot | 0.001783 | 1.174% | 98.059% |

The F16 hot tier reduced pure-MX mean KL by 6.7%. Q8_0 retained slightly lower KL at 32K, while hybrid MXFP8 had slightly lower RMS probability error and higher top-1 agreement.

## Performance results

Qwen3.8-27B prompt processing at 32K, with one warmup and three measured runs:

| KV cache | Prompt throughput |
|---|---:|
| Q8_0 | 3313.5 tok/s |
| MXFP8 plus F16 hot | 3296.6 tok/s |
| Pure MXFP8 | 3163.4 tok/s |

Hybrid MXFP8 was 0.51% behind Q8_0 at 32K in this run. The 128K perplexity inference pass was faster with hybrid MXFP8, but it was not a repeated `llama-bench` measurement.

Results are specific to this model, GPU, build, clocks, and workload.

## Known bottleneck and next optimization

The F16 reconstruction bridge is the main format-specific bottleneck. For every attention call it reads MXFP8 K/V, writes a full F16 logical cache, and then reads the F16 cache again. The overhead becomes particularly important for long-context token generation because the active cache is reconstructed for each generated token.

The next Blackwell optimization should:

1. Consume cold MXFP8 tiles directly with block-scaled `QMMA.SF` instructions.
2. Consume the F16 hot tail directly with the existing F16 MMA path.
3. Combine cold and hot regions through one online-softmax state.
4. Eliminate the global F16 logical-cache scratch buffer.

A lower-risk intermediate step is native MXFP8 QK combined with V decoding into shared memory and the existing F16 probability-times-V calculation. Full native FP8 QK and PV can follow after numerical validation at long context.
