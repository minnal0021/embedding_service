# Spike: llama.cpp ROCm Docker image on WSL2 (Radeon 890M)

**Goal:** prove that the prebuilt `ghcr.io/ggml-org/llama.cpp:server-rocm`
image can run both embedding models on this machine's AMD iGPU inside Docker on
WSL2, with correct output, acceptable memory/speed, stable under load, and
reachable from other machines on the LAN. The result decides whether we move to
the Docker architecture (FastAPI gateway + one `llama-server` container per
model) planned in the previous discussion.

Machine facts (checked 2026-10-02):

| | |
| --- | --- |
| CPU / iGPU | AMD Ryzen AI 9 HX 375 / Radeon 890M (gfx1150, PCI ID `0x150E`) |
| WSL | Ubuntu 24.04.4, kernel 6.6.87.2, mirrored networking, 12 GB RAM cap |
| Docker | Docker Engine 29.6.1 inside the WSL distro (not Docker Desktop) |
| GPU node | `/dev/dxg` present; no `/dev/kfd`, no ROCm installed |
| LAN IP | 192.168.10.212 (shared with Windows via mirrored networking) |

Why it should work: AMD's WSL bridge library `librocdxg` lists the Ryzen AI 9
HX 375 as supported from librocdxg 1.2.0 + ROCm 7.2.x; the llama.cpp ROCm image
is built on ROCm 7.2.1 with `gfx1150` among its GPU targets. Known risks:
llama.cpp marks its Docker images "not verified on WSL", and there is an open
report of a hang on WSL2 + ROCm 7.2 (ggml-org/llama.cpp#20545).

> Steps marked **(you)** need admin/sudo or Windows access. Everything else can
> be run by Claude once those are done. Record each step's result in the table
> at the end.

---

## Step 1 (you) — Windows: driver, WSL kernel, firewall

In **Windows**:

1. Install the latest **AMD Software: Adrenalin Edition** for the Ryzen AI 9 HX
   375 from <https://www.amd.com/en/support/download/drivers.html> (a recent
   driver is what exposes ROCm to WSL). Reboot if asked.
2. In an **admin PowerShell**:

   ```powershell
   wsl --update
   # Inbound port for the embedding service through Windows Defender Firewall…
   New-NetFirewallRule -DisplayName "Embedding service 8001" -Direction Inbound `
     -Protocol TCP -LocalPort 8001 -Action Allow
   # …and through the Hyper-V firewall that guards WSL in mirrored mode
   # ({40E0AC32-…} is WSL's VM creator id).
   New-NetFirewallHyperVRule -Name "EmbeddingService8001" `
     -DisplayName "Embedding service 8001 (WSL)" -Direction Inbound `
     -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' `
     -Protocol TCP -LocalPorts 8001 -Action Allow
   wsl --shutdown
   ```

   (If the rules already exist from earlier LAN testing, skip them —
   `Get-NetFirewallRule -DisplayName "Embedding service 8001"` shows it.)

3. Reopen the WSL terminal.

## Step 2 (you) — WSL: install the ROCm-on-WSL bridge

Only `librocdxg` is needed on the host: the container brings ROCm itself and
mounts these files. The package has no dependencies and installs just
`/opt/rocm/lib/librocdxg.so*` and `/opt/rocm/share/rocdxg/dids.conf`.

```bash
cd /tmp
curl -LO https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
sudo dpkg -i rocdxg-roct_1.2.2_amd64.deb
ls -l /opt/rocm/lib/librocdxg.so /opt/rocm/share/rocdxg/dids.conf /usr/lib/wsl/lib/libdxcore.so
```

All three files must exist.

---

From here on, run from the repo root. These shell variables are reused by the
later steps:

```bash
cd ~/minnal_dev/embedding_service
# Flags that give a container the GPU on WSL (instead of /dev/kfd + /dev/dri).
# HSA_ENABLE_DXG_DETECTION is required for ROCm < 7.13 (the image has 7.2.1).
ROCM_WSL=(--device /dev/dxg
  -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so
  -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so
  -v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf
  -e HSA_ENABLE_DXG_DETECTION=1)
# GGUFs already downloaded by the service (HF cache layout; symlinks resolve
# inside the mount).
GEMMA=/models/$(cd model_cache && find . -name 'embeddinggemma-300M-Q8_0.gguf' | head -n 1 | sed 's|^\./||')
QWEN=/models/$(cd model_cache && find . -name 'Qwen3-Embedding-8B-Q4_K_M.gguf' | head -n 1 | sed 's|^\./||')
echo "$GEMMA"; echo "$QWEN"
```

## Step 3 — The GPU is visible to ROCm inside a container

```bash
docker run --rm "${ROCM_WSL[@]}" rocm/dev-ubuntu-24.04:7.2.1 rocminfo \
  | grep -E 'Marketing Name|Name: +gfx'
```

**Pass:** an agent `gfx1150` / `AMD Radeon 890M Graphics` is listed (alongside
the CPU agent).

If no GPU agent appears, try in order (re-run the command after each):

1. Tell librocdxg the device ID explicitly:
   `echo '0x150E,11,5,0' | sudo tee -a /opt/rocm/share/rocdxg/dids.conf`
2. Add `-e HSA_OVERRIDE_GFX_VERSION=11.5.0` to the `docker run`.
3. Check the Windows driver version (Step 1) and `wsl --update` again.

If it still fails, stop here: the ROCm path isn't viable on this machine yet
(fallbacks are in *Decision* below).

## Step 4 — Gemma on the GPU via `server-rocm`

First confirm llama.cpp itself sees the GPU, and note its free memory:

```bash
docker run --rm "${ROCM_WSL[@]}" ghcr.io/ggml-org/llama.cpp:server-rocm --list-devices
```

**Pass:** a `ROCm0: AMD Radeon 890M Graphics (… MiB, … MiB free)` line, not
`(none)`. Record the free MiB — it bounds what Step 6 can load.

Then run Gemma with all layers offloaded:

```bash
docker run -d --name spike-gemma "${ROCM_WSL[@]}" \
  -v "$PWD/model_cache:/models:ro" -p 127.0.0.1:8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-rocm \
  -m "$GEMMA" --embedding --pooling mean -ngl 99 \
  -c 2048 -b 2048 -ub 2048 --host 0.0.0.0 --port 8080

sleep 15; docker logs spike-gemma 2>&1 | grep -iE 'no usable GPU|ROCm|error|fail|out of memory'
curl -s localhost:8080/health
```

**Pass:** `/health` returns `{"status":"ok"}` and the log has **no**
`no usable GPU found` warning (that warning means it silently fell back to the
CPU). For the full device/offload detail, re-run with `-lv 4` added to the
llama-server arguments.

## Step 5 — Gemma output matches the current (CPU) service

Start the current service on the CPU for reference, then compare the same
prompted texts from both:

```bash
./embedding_service.sh start --models gemma

python3 - <<'EOF'
import json, math, urllib.request

def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))

def unit(v):
    n = math.sqrt(sum(x * x for x in v)); return [x / n for x in v]

texts = ["The court filed the complaint on Tuesday.",
         "Why is the sky blue?", "Photosynthesis converts light into chemical energy."]
ref = post("http://127.0.0.1:8001/embedding/gemma/document", {"payloads": texts})["embeddings"]
gpu = post("http://127.0.0.1:8080/v1/embeddings",
           {"input": ["title: none | text: " + t for t in texts]})["data"]
for t, r, g in zip(texts, ref, gpu):
    cos = sum(a * b for a, b in zip(r, unit(g["embedding"][:768])))
    print(f"{cos:.5f}  {t}")
EOF

./embedding_service.sh stop
```

**Pass:** every cosine ≥ 0.998. (Dry run with the CPU `server` image gave
0.9998 — same GGUF, different llama.cpp build; the GPU adds only float noise.)

## Step 6 — Qwen (Q4_K_M) fits and runs on the iGPU

The 890M's memory is Windows shared memory, not the WSL VM's 12 GB, so watch
the logs for allocation failures.

```bash
docker run -d --name spike-qwen "${ROCM_WSL[@]}" \
  -v "$PWD/model_cache:/models:ro" -p 127.0.0.1:8081:8080 \
  ghcr.io/ggml-org/llama.cpp:server-rocm \
  -m "$QWEN" --embedding --pooling last -ngl 99 \
  -c 8192 -b 8192 -ub 8192 --host 0.0.0.0 --port 8080

sleep 60; docker logs spike-qwen 2>&1 | grep -iE 'no usable GPU|out of memory|failed to allocate|error|fail'
curl -s localhost:8081/health
curl -s localhost:8081/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input":["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: why is the sky blue?"]}' \
  | python3 -c "import sys,json; print(len(json.load(sys.stdin)['data'][0]['embedding']), 'dims')"
```

**Pass:** health ok, `4096 dims`, and no `no usable GPU` / allocation errors in
the log. Compare `--list-devices` free memory (Step 4) before and after to see
what Qwen uses. If it runs out of memory, retry with
`-c 4096 -b 4096 -ub 4096`; if that also fails, record it (Qwen would stay on
the CPU or need a smaller quantization).

While both containers run, check overall memory in Windows Task Manager →
Performance → GPU ("Shared GPU memory").

## Step 7 — Speed vs CPU

```bash
python3 - <<'EOF'
import json, time, urllib.request
docs = []
with open("sample_data/eli5_question_answer.jsonl") as f:
    for line in f:
        q, a = json.loads(line); docs.append(q + "\n" + a[:1500])
        if len(docs) == 64: break

def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))

for name, port in (("gemma", 8080), ("qwen", 8081)):
    t = time.time()
    for i in range(0, len(docs), 16):
        post(f"http://127.0.0.1:{port}/v1/embeddings", {"input": docs[i:i + 16]})
    print(f"{name} GPU: {len(docs) / (time.time() - t):.1f} docs/s")
EOF
```

CPU baselines on this machine (from the current service): Gemma ≈ 12 docs/s,
Qwen Q4_K_M ≈ 0.8 docs/s.

**Pass:** the GPU is clearly faster than the CPU for Qwen (≥ 2×). Gemma may be
close to the CPU — it is small.

## Step 8 — Stability (watch for the WSL hang)

```bash
python3 - <<'EOF'
import json, time, urllib.request
def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))
t = time.time()
for i in range(300):
    post("http://127.0.0.1:8080/v1/embeddings", {"input": [f"stability test {i}"] * 8})
    if i % 20 == 0:
        post("http://127.0.0.1:8081/v1/embeddings", {"input": [f"stability test {i}"]})
print(f"300 + 15 requests OK in {time.time() - t:.0f}s")
EOF
```

**Pass:** finishes with no timeout (a hang raises `TimeoutError` after 120 s).
Also restart both containers once (`docker restart spike-gemma spike-qwen`) and
re-check `/health` — GPU re-initialisation must work.

## Step 9 — Reachable from the LAN through a published Docker port

```bash
docker rm -f spike-gemma
docker run -d --name spike-gemma "${ROCM_WSL[@]}" \
  -v "$PWD/model_cache:/models:ro" -p 0.0.0.0:8001:8080 \
  ghcr.io/ggml-org/llama.cpp:server-rocm \
  -m "$GEMMA" --embedding --pooling mean -ngl 99 \
  -c 2048 -b 2048 -ub 2048 --host 0.0.0.0 --port 8080
```

**(you)** From **another machine** on the network:

```bash
curl http://192.168.10.212:8001/health
```

**Pass:** `{"status":"ok"}`. If it times out:

1. Retry the container with `--network host` instead of `-p …` (and
   `--port 8001` for llama-server) — Docker's port publishing has had issues
   under WSL mirrored networking.
2. If that also fails, the firewall rules from Step 1 are the suspect
   (`Get-NetFirewallHyperVRule -Name EmbeddingService8001`).

Record which variant worked; the compose file will use it.

## Cleanup

```bash
docker rm -f spike-gemma spike-qwen
# Optional, frees ~10+ GB:
docker image rm rocm/dev-ubuntu-24.04:7.2.1 ghcr.io/ggml-org/llama.cpp:server-rocm
```

---

## Results

| Step | Check | Result | Notes |
| --- | --- | --- | --- |
| 3 | `rocminfo` in container shows gfx1150 | ✅ Pass | No workarounds needed (no dids.conf entry, no GFX override). GPU pool ≈ 16 GB (Windows shared memory). |
| 4 | `--list-devices` shows ROCm0; Gemma runs on GPU | ✅ Pass | free MiB: 15176 of 16369. Gemma: 25/25 layers offloaded, 312 MiB weights + 63 MiB compute on ROCm0. Default log level hides offload info — use `-lv 4` to see it. |
| 5 | Gemma cosine vs CPU ≥ 0.998 | ✅ Pass | min cosine: 0.99974 (0.99974 / 0.99977 / 0.99979) |
| 6 | Qwen Q4_K_M fits on iGPU | ✅ Pass | ctx 8192 (`-b/-ub 8192`): 37/37 layers, ROCm0 ≈ 10.6 GB = 4454 MiB weights + 1152 MiB KV + 4996 MiB compute; ~3.9 GB left with both loaded. 4096 dims, ready in ~23 s. |
| 7 | Speed (docs/s) | ✅ Pass | gemma: 27.0 (≈2.3× CPU) · qwen: 1.9 (≈2.4× CPU). 64 ELI5 docs, batches of 16, after one warm-up request. |
| 8 | 300-request run, restart | ✅ Pass | 300 + 15 requests in 12 s; `docker restart` of both → healthy (gemma 4 s, qwen 40 s), full offload again. Extra 5-min concurrent soak (2 threads per model, real docs): 5152 gemma + 440 qwen docs, no errors/hangs. |
| 9 | LAN access | ✅ Pass | `-p 0.0.0.0:8001:8080` (no `--network host` needed): `curl http://192.168.10.212:8001/health` from another LAN machine → `{"status":"ok"}`. |

## Decision

**Outcome (2026-10-02):** all steps passed. The Docker architecture below is
built (`docker-compose.yml`, `docker/compose.rocm-wsl.yml`).

- **All pass** → build the Docker architecture: `docker-compose.yml` with the
  gateway (port 8001, LAN-facing) plus `llama-gemma` / `llama-qwen` (internal
  only) using `server-rocm`; `./embedding_service.sh start|stop` drives
  `docker compose up -d` / `down`. A `vulkan` / `cpu` profile selects other
  images on other machines; macOS uses a native `llama-server` (Homebrew, Metal)
  since Docker on macOS has no GPU access.
- **Steps 3–4 fail** (no GPU in the container) → keep the same Docker design but
  run the `server` (CPU) image here for now; revisit when librocdxg/ROCm add or
  fix gfx1150 support. Optionally try `server-vulkan` (Mesa's D3D12-backed
  Vulkan on WSL — expect it to be slower than ROCm).
- **Step 6 fails only** → Gemma on the GPU, Qwen on the CPU image (or a smaller
  quantization).
- **Step 8 fails** (hangs) → not production-safe; fall back as for 3–4 and track
  ggml-org/llama.cpp#20545.

## References

- ROCm on WSL bridge, supported GPUs and Docker flags: <https://github.com/ROCm/librocdxg>
- llama.cpp ROCm image build (ROCm 7.2.1, GPU targets incl. gfx1150): <https://github.com/ggml-org/llama.cpp/blob/master/.devops/rocm.Dockerfile>
- llama.cpp Docker images: <https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md>
- WSL2 + ROCm 7.2 hang report: <https://github.com/ggml-org/llama.cpp/issues/20545>
- WSL mirrored networking & Hyper-V firewall: <https://learn.microsoft.com/en-us/windows/wsl/networking>
