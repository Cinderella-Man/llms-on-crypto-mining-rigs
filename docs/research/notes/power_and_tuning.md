# Power on the P106 rig: what the cards draw, when the 75 W cap costs speed, and which documented settings help (2026-10-24)

Scope: how much power the nine P106-090 cards of a Manli M-P106L9-N6G ([hardware spec](../../mining_rig.md)) draw in
each of this project's workloads, whether they are power-limited, and which settings make them faster or more
efficient. The new question is v2's modes in which every card works at once: the pipelined executor (Q87) and its
races (Q93), several conversations staggered through the cards (Q104, Q120/Q121), the staircase drafter on idle cards
(Q20) and long prompts with a full micro-batch pipeline. "Safe" means NVIDIA's documented management interfaces only
(`nvidia-smi`, NVML, and NV-CONTROL through `nvidia-settings`), every change reversible at runtime without a reboot, no
firmware or driver modification, and an exact output check after any change. This note builds on the
[hardware tuning note](hardware_tuning.md) (memory offset, PCIe, ASPM, host settings) and does not repeat it.
Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (vendor documentation, headers, measurements in a repository) and
  secondary.
- "Top bin" is a card's highest boost clock, reached when nothing limits it: 1,860–1,911 MHz depending on the card
  (GPU0 1,860; GPU1 1,873; GPU4 and GPU6 1,885; GPU8 1,898; the others 1,911). "Base clock" is 1,354 MHz.
- "Duty" is the share of time a card runs kernels. In a layer split each card works on its own layers of every token
  in turn, so one stream keeps each of N cards busy about 1/N of the time.
- Power figures are board power as `nvidia-smi` reports it (`power.draw`). On Pascal this is an instantaneous reading,
  not a one-second average (NVML: averaging over 1 s applies to Ampere and newer).

Starting point, measured. The figures come from the `nvidia-smi` logs (every card every ~0.54 s: P-state, clocks,
power, temperature, utilization, clock event reasons) recorded with the calibration (2026-10-22), the Flint and Qwen
sweeps (2026-10-22), Q38/Q46 and the rig step-1 runs (2026-10-23): 266,202 card samples, cut into llama-bench tests by
each test's start time and duration. Limits, thresholds and cumulative counters come from the rig inventory (E0.1)
and an `nvidia-smi -q` power and temperature query of all nine idle cards taken between timed runs on 2026-10-24.
- **Limits, every card:** power limit 75 W by default, adjustable from 60 to 75 W, enforced 75 W; thermal limits
  83 °C target, 98 °C slowdown, 101 °C shutdown; `fan.speed` reports 0 %.
- **Decode on the fork, layer split, the tg128 that follows a prompt test:** each card averages 35.5–39.0 W (peaks
  45–68 W) at its top bin, and the cap shows in 0–1 % of samples. The rig draws 227–251 W of GPU power on six cards
  (three idle cards included) and 317–329 W on nine: 6.2–7.2 J per token on six cards (S4-L0, UD-Q4_K_XL at
  34–37 tok/s), 8.9 J on nine (Q8_0 at 35.9 tok/s).
- **One card running a whole model alone (stock llama.cpp, Llama-2-7B):** 68–72 W average in tg128 with the cap
  active in 45–94 % of samples and the clock at 1,640–1,780 MHz; 58–65 W in pp512 with the cap in 42–92 % of samples.
  The 75 W cap binds when a card is busy all the time.
- **Idle:** 5.9 W per card in P8 (5.6–6.3 W; also with a model loaded), 26 W in P0 at base clock, 33 W in P0 at the
  top bin.

## Verdict

- **The cards are not power-limited in today's layer split, and in all-busy modes they will be, but the cap costs
  little decode speed.** Power follows duty: a card draws ~33 W waiting in P0 at its top bin plus ~40 W times its duty
  (est. fit: 35–39 W measured at duty 1/6–1/9, ~70 W at duty 1). Nine busy cards would draw est. **630–675 W**
  (9 × 70–75 W, the cap holding the average) against ~320 W in nine-card decode today; with the host, ten fans and the
  power supply's losses the rig would draw est. **~800 W at the wall**, Manli's own figure for the system mining
  Ethereum, and half the 1,600 W supply. Under the cap a busy card's clock falls ~6–9 % (to ~1,700–1,780 MHz), which
  costs **est. 2–4 % on decode stages** (weight streaming hardly depends on the core clock) and **est. 6–9 % on
  prompt processing** (MMQ runs on the core clock). The notes on Q87, Q104 and Q20 charged a 10 % stage slowdown for
  the cap; for decode that is a conservative bound.
- **The larger loss is the other way round: lightly loaded cards drop to base clock.** NVIDIA's default PowerMizer
  policy on these cards "adjust[s] GPU clocks based on GPU utilization" (NV-CONTROL header). A layer-split stream keeps
  each card only 11–17 % busy, and a decode that starts from idle runs at **1,354 MHz with no clock event reason
  reported** (seen directly in the stock build's logs). It stays there until a heavy phase, such as a long prompt,
  lifts the cards to their top bin, where the following light decode kept them in every logged run. On the fork this
  is est. **14–18 % of decode speed**: tg128 run on its own gave 29.9 tok/s on six cards and 31.0 on nine, against
  34.7 and 36.1–36.5 when tg followed a prompt test on the same build and cards. A served chat request with a short
  prompt ran at 31.4–33.6 tok/s on nine cards. This is most likely the ~5 ms per token "state effect" of the
  [v2 synthesis](v2_synthesis.md) (24.6 ms of compute per token after prompt tests against 29.5 ms alone): a 0.45
  sensitivity to a 1,354 → ~1,870 MHz clock gives 4–5 ms (est.). The one documented control is the PowerMizer mode
  "Prefer Maximum Performance" (`GPUPowerMizerMode=1` through NV-CONTROL). It is available only if the running X server
  exposes the cards as NV-CONTROL GPU targets, and a one-minute read-only query answers that. It changes no clock
  beyond the stock boost table and no output bit.
- **Energy per token is set by waiting, not by work.** In single-stream decode ~80–85 % of the GPU energy goes to
  cards waiting in P0 between their turns (6 × 33 W of ~230–250 W on six cards). Every all-busy mode amortizes that:
  est. 2.5–2.9 J per token on nine cards at Q104's 235–256 tok/s and 2.8–3.4 J on six cards, against 8.9 J (nine
  cards) and 6.2–7.2 J (six) today, i.e. 1.8–3.6× less energy per token even with the cap binding. Single-stream
  decode on six cards instead of nine already saves 20–30 % per token (measured: 6.2–7.2 against 8.9 J).
- **Settings, ranked (details in the table):** (1) PowerMizer "Prefer Maximum Performance": measured in Q141, it gives
  nothing and costs ~20 W per idle card (section 3.1), so it is not a setting to use; (2) the memory offset (Q133), est. +3–5 % decode in the layer split and +2.5–4.5 % under the cap;
  (3) a core offset (Q135), est. +1–2.5 % decode and, under the cap, ~2/3 of the offset in clock, est. +3–4 % prefill
  per +100 MHz; (4) a 60 W power limit in all-busy modes for efficiency, est. +10–20 % tokens per joule (GPU) for
  2.5–5.5 % less decode speed. Persistence mode matters only headless. Fans, temperature targets, locked clocks and
  limits above 75 W have nothing to offer on these cards.
- **Throttle reasons in every log so far:** idle (0x1), none (0x0) and the software power cap (0x4) only, in 1 % of
  samples. No thermal slowdown (0x20, 0x40), no hardware slowdown or power brake (0x8, 0x80); the hottest card reached
  57 °C. NVML has no "reliability voltage" reason: a busy card at its top bin reports 0x0, as does a card held at base
  clock by the utilization policy, so clocks must be logged next to the bitmask.
- **First measurement (no setting changed, 10 min):** six cards, the v2 build, `llama-bench -p 0 -n 128` and then a
  second run with `-p 2048 -n 128` (its tg follows the prompt test), logging clocks and power every 200 ms. It confirms
  or refutes the base-clock cause of the cold-decode gap, and decides whether lever 1 is worth the owner's approval.

## Ranked settings

Gains are Flint decode or prefill on the fork, est. unless measured. "Layer split" is one stream today; "all-busy" is
Q87/Q93/Q104/Q20 or a full prefill pipeline with every card near full duty.

| # | Setting (interface) | Expected effect per mode (est.) | Risk | Reversible at runtime? | Owner approval | First test | Kill criterion |
|---|---|---|---|---|---|---|---|
| 1 | **PowerMizer "Prefer Maximum Performance"** per card (`nvidia-settings -a '[gpu:N]/GPUPowerMizerMode=1'`, NV-CONTROL attribute 334) | Measured (Q141): **no gain** (tg128 from idle 29.79 against 29.75 tok/s, a served reply after a minute idle 33.30 against 33.25, six cards); idle power +~20 W per card while set (P0 instead of P8) | Low: stock boost table, no overclock; higher idle power | Yes: set the default mode back (read first) | Yes (changes card behaviour; idle power) | R0 query, R1 cause check, R2 A/B (section 5) | The X server does not expose the cards (then the fallbacks of section 3.1); or tg alone gains < 3 % with the clocks at top bin |
| 2 | **Memory clock offset** (NVML `nvmlDeviceSetMemClkVfOffset`, Q133) | Layer split +3–5 % decode at +10 %; all-busy decode +2.5–4.5 % (1–3 W of memory power comes out of the core clock under the cap); prefill 0–3 % | Silent weight corruption; a hang needs a reboot at the console | Yes (offset 0) unless a card hangs | Approved for GPU0 | Q133's kit | Q133's criteria |
| 3 | **Core clock offset** (NVML `nvmlDeviceSetGpcClkVfOffset` or `nvmlDeviceSetClockOffsets` for P0, Q135) | Per +100 MHz: layer split decode +1–2.5 %, prefill +4–5 %; under the cap ~2/3 of the offset survives as clock: all-busy decode +1–2 %, prefill +3–4 %. On Pascal the offset applies to every performance level, so base-clock decode gains too (1,354 → 1,454 MHz, +7 %) | Core instability: wrong results or Xid errors; a hang needs a reboot | Yes (offset 0) unless a card hangs | Yes; owner at the console | R6: the Q133 kit with `--core`, +25 MHz steps | < 2 % on a full prefill pipeline at +100 MHz, or any exact-check failure at the last clean step minus one |
| 4 | **Power limit 60 W in all-busy modes** (`nvidia-smi -i N -pl 60`, NVML `nvmlDeviceSetPowerManagementLimit`) | All-busy decode −2.5–5.5 % speed for −14 to −20 % card power (70–75 → 60 W): **tokens per joule +10–20 %** (GPU), +7–17 % at the wall; all-busy prefill −7–10 % speed, +5–15 % tokens per joule (GPU); layer split: no effect | None: clocks stay on the stock curve | Yes (`-pl 75`; any reboot or driver reload resets it) | Yes (a speed-for-energy trade) | R4: 75 against 60 W during the all-busy oracle and pp8192 | GPU tokens per joule +8 % or less, or decode slower by more than 5 % |
| 5 | **60–65 W plus a validated core offset** (the miners' tune) | Recovers ~2/3 of the offset at equal power: at 60 W + 100 MHz the clocks of ~66–68 W at stock; efficiency of row 4 at about half its speed cost | As rows 3 and 4 | Yes | Yes | After rows 3 and 4 | As rows 3 and 4 |
| 6 | **Card count and mode** (operational, not a driver setting) | Single stream on six cards instead of nine: 20–30 % fewer J per token at equal or better speed (measured); all-busy modes: 1.8–3.6× less energy per token | None | Yes | No | Already measured; R3 for all-busy | — |
| 7 | Persistence mode (`nvidia-smi -pm 1`) | 0 % with Xorg running; headless it keeps power limits and offsets between processes | None | Yes (`-pm 0`; reset to disabled at each boot) | With Q83 | With Q83 | — |
| 8 | Temperature target (`nvidia-smi -gtt`, NVML acoustic threshold), fan control, overvoltage, locked or application clocks, a limit above 75 W | 0 %: the cards never pass 57 °C against an 83 °C target; they report no fan of their own; the rest is unsupported or unsafe (section 3.7) | — | — | — | None | — |

## 1. The numbers (question 1)

### 1.1 Limits per card

All nine cards report the same values (`nvidia-smi -q -d POWER`, `--query-gpu=power.*`):

| Quantity | Value | Interface |
|---|---|---|
| Default power limit | 75 W | `power.default_limit`, `nvmlDeviceGetPowerManagementDefaultLimit` |
| Minimum / maximum settable | 60 W / 75 W | `power.min_limit`, `power.max_limit`, `nvmlDeviceGetPowerManagementLimitConstraints` |
| Current and enforced limit | 75 W | `power.limit`, `enforced.power.limit`, `nvmlDeviceGetEnforcedPowerLimit` |
| Temperature target / slowdown / shutdown | 83 / 98 / 101 °C; "Max Operating Temp", "T.Limit" and memory temperature N/A | `nvidia-smi -q -d TEMPERATURE`, `nvmlDeviceGetTemperatureThreshold` |
| Temperature at idle (2026-10-24 query) | 22–30 °C | `temperature.gpu` |
| Memory clock | 4,004 MHz in P0, 405 MHz in P8 (no other) | `clocks.mem`, supported clocks |
| Core clock | 139 MHz in P8; 1,354 MHz base; top bin 1,860–1,911 MHz | `clocks.sm`, `clocks.max.sm` reports 1,911 for every card |
| Fan | `fan.speed` 0 % on every card | NVML fan calls are for "products with dedicated fans" |
| Energy counter, one-second power average | Not available (Volta+ and Ampere+) | `nvmlDeviceGetTotalEnergyConsumption`, `power.draw.average` |

The power reading covers "the entire board" (`nvidia-smi`), i.e. the GPU, its memory and its regulators. The cards
report no memory temperature and no separate memory power.

### 1.2 Draw per card and per mode

| Mode | Duty per card | Card power, mean (peak) | Core clock while busy | Cap active | Rig GPU total (all nine cards) | Energy |
|---|---|---|---|---|---|---|
| Idle, P8 (model loaded or not) | 0 | 5.9 W | 139 MHz | – | ~53 W | – |
| Waiting in P0 (model loaded, between kernels) | 0 | 26 W at base clock, 33 W at the top bin | – | – | – | – |
| Decode, fork, layer split, six cards, after a prompt (Q38/Q46/step-1 runs) | ~16 % | 35.5–39.0 W (45–68 W) | top bin (1,780–1,883 MHz mean) | 0–1 % | 227–251 W | 6.2–7.2 J per token at 34–37 tok/s |
| Decode, fork, nine cards, after a prompt (Q38 and step-1 runs, Q8_0) | ~11 % | 35.3–36.5 W (44–47 W) | 1,859–1,876 MHz | 0 % | 317–329 W | 8.9 J per token at 35.9 tok/s |
| Decode starting cold, stock build, five cards (Flint depth run, depth 0) | 13–21 % | 26–33 W | **1,354 MHz** (base), reason 0x0 | 0 % | ~170 W | 11.7 J per token at 14.6 tok/s |
| Prompt, pp512, fork, six cards | ~13 % (one or two cards at a time) | 31.8–34.5 W; 41–55 W in busy samples (70–76 W) | 1,680–1,770 MHz | 5–28 % of busy samples | 201–224 W | 1.0–1.1 J per prompt token |
| Prompt, pp512, fork, nine cards (Q8_0) | ~8 % | 29.5–30.0 W (67–69 W) | 1,580–1,630 MHz | 0–6 % | 265–269 W | 1.4 J per prompt token |
| Long prompt, stock, five cards (8k–32k depth fill, `-ub 512`) | ~2.4 cards ≥ 80 % busy at once | 50–53 W in busy samples (69–76 W) | 1,820–1,840 MHz | 19–24 % | 249–263 W | – |
| One card does all the work (calibration, stock, Llama-2-7B) | ~100 % | tg 68–72 W; pp 58–65 W (peaks to 81 W) | tg 1,640–1,780; pp 1,690–1,790 MHz | 45–94 %; 42–92 % | 211–221 W (1 busy + 8 idle) | 11–12 J per token |
| Two cards, ~50 % each (calibration) | ~50 % | 57 W | 1,823–1,849 MHz | 15–24 % | ~155 W | – |
| **Every card busy (est.)** | ~100 % | 70–75 W, capped | ~1,700–1,780 MHz (calibration) | most samples | **630–675 W** on nine; 438–468 W on six + three idle | 2.5–2.9 J per token (nine, Q104 est. 235–256 tok/s); 2.8–3.4 J (six, 137–159 tok/s) |

- **The duty model** (est., fitted on the rows above): P ≈ 33 W + duty × ~40 W at the top bin, capped at 75 W. Duty
  1/6 gives ~40 W and duty 1/9 ~37 W (measured 35.5–39 and 35–36.5), duty 1 gives ~73 W (measured 68–72 W with the cap
  active). Six-card single-stream decode therefore spends ~200 W of its ~230–250 W keeping cards ready in P0.
- **The hardware spec and the layer-split numbers agree.** The spec's "a card generating text draws ~60–70 W …
  spends much of its time at the 75 W limit" is the calibration's single card, which runs the whole model and is busy
  all the time. In a layer split the same card is busy a sixth or a ninth of each token and averages 35–39 W.
- **Instantaneous peaks pass the limit.** 166 samples read above 75 W, the highest 86.9 W: the cap holds the average
  over the controller's window, not every millisecond. Sizing below uses 75 W per card for averages and ~87 W for
  short peaks.
- **The fork's kernels at full duty have not been logged.** They stream 120–165 GB/s against the stock kernels'
  73 GB/s on the calibration card, so their memory draws more (est. +3–6 W) and their demand at full duty is at least
  the stock 70 W: est. capped most of the time, at ~1,650–1,760 MHz. R3 measures it.

### 1.3 Nine cards plus the host against the 1,600 W supply

| State | GPU board power | Host, fans, board (est.) | At the wall (est.) |
|---|---|---|---|
| Idle, all cards in P8 | ~53 W | 45–85 W: Celeron 3865U (15 W TDP) mostly idle, memory, SSD, NIC ~5 W; ten 90 mm 4,800 rpm fans est. 3–6 W each | ~110–170 W (supply at low load, est. 80–88 % efficient) |
| Single-stream decode, nine cards | ~320 W | 60–95 W | ~420–460 W |
| Every card busy | 630–675 W (peaks to ~780 W if all cards peak together) | 60–95 W | **~750–850 W** (est. 90–92 % efficient at ~45 % load) |

- **The supply has 2× headroom.** At ~45–50 % load a 1,600 W unit also sits near its best efficiency (est.; the
  model and its 80 PLUS rating are not recorded). Manli rates the whole system at 800 W while mining Ethereum with all
  nine cards, the same envelope as every card busy at its cap.
- **A wall meter replaces the host column** with one reading per state (section 4.2).

### 1.4 The seven 6-pin inputs and nine cards

- **Ratings:** the PCI Express CEM specification rates a x16 slot at 75 W and a 2×3 (6-pin) auxiliary connector at
  75 W; NVIDIA's power application note states "the PCIe slot is rated for 75 W, and a PCIe 8-pin connector is rated
  for 150 W", and that an 8-pin "can carry up to 175 W" in practice. A 6-pin carries the same three 12 V pins as an
  8-pin (NVIDIA's pinout table), and NVIDIA's own 75 W GP106 card (Quadro P2000) takes all its power from the slot.
- **This board:** the nine cards take their power through the board's slots, and the board takes its 12 V for them
  from seven 6-pin inputs (est.: how the inputs are wired to the slots, and whether the cards have power connectors of
  their own, is not documented by Manli and was not inspected).
- **Load per connector (est.):** if all GPU power enters through the seven inputs, today's nine-card decode puts
  ~46 W on each and all-busy work 90–96 W on average (630–675 W / 7), i.e. 1.2–1.3× the 75 W CEM rating, and ~2.7 A
  per 12 V pin. If two slots share one input, that input carries up to ~150 W (4.2 A per pin). Both are inside what
  NVIDIA allows on the same three pins of an 8-pin (175 W, ~4.9 A per pin), and within the 800 W mining envelope the
  system was sold for. With a shared 12 V plane on the board the inputs split the load by cable and contact
  resistance, unevenly by an unknown margin.
- **What this asks of the owner before the first sustained all-busy run:** all seven inputs plugged, each on its own
  supply cable rather than a splitter (NVIDIA's note advises against Y-splitters), and the supply's 12 V rail rating
  checked against ~700 W. Today's workloads never drew more than ~330 W of GPU power.

### 1.5 Throttle reasons in the logs

| Bit (NVML / `clocks_event_reasons.active`) | Meaning (NVML documentation) | Samples |
|---|---|---|
| 0x0 none | "Clocks are as high as possible" | 62,295 (23 %) |
| 0x1 GPU idle | "Nothing is running on the GPU" | 201,207 (76 %) |
| 0x4 SW power cap | clocks lowered to stay under the power limit | 2,700 (1 %): single-card runs, long prompts, a few prompt micro-batches |
| 0x2 application clocks, 0x10 sync boost, 0x100 display clock | settings this rig does not use | 0 |
| 0x8 HW slowdown, 0x40 HW thermal, 0x80 HW power brake | clocks halved or worse: temperature, an external power-brake signal, or fast-trigger over-power protection | 0 |
| 0x20 SW thermal slowdown | above the maximum operating temperature | 0 |

- **No thermal limit or power brake has ever acted:** the hottest card reached 57 °C (one card alone at full duty),
  decode runs at 28–38 °C, and the cumulative counters of all nine cards read 0 µs of hardware thermal slowdown and
  0 µs of power braking since the driver loaded (2026-10-24 query; software thermal slowdown reads N/A).
- **The cumulative power-capping counter is not usable on these cards.** In the same query two cards report
  ~1.49 × 10^16 µs (centuries) and others 52 s to ~4,100 s since boot, the high values hard to reconcile with
  the sampled cap shares of today's work (not verified either way). Cap shares in this note come from the sampled
  bitmask.
- **"Reliability voltage" has no NVML bit.** A busy card that is limited by nothing else stops at the top of its
  voltage/frequency table (its top bin) and reports 0x0; that is what Windows tools label a voltage limit. Mask 0x0 is
  also what a lightly loaded card reports while the utilization policy holds it at base clock (section 3.1).
- **Clocks fall with temperature in the logs** (uncapped busy samples: ~1,870 MHz at 25–34 °C, ~1,830 at 35–39 °C,
  1,730–1,800 at 40–54 °C), but the hotter samples come from full-duty runs, so power and temperature cannot be
  separated here (not verified; GPU Boost is reported to drop a ~13 MHz bin at some temperature steps).

## 2. When the 75 W cap costs speed (question 2)

### 2.1 What bounds each all-busy mode

| Mode | Work on each card | Bound by | Core-clock sensitivity (est.) | Cap's clock loss (est.) | Cost of the cap (est.) |
|---|---|---|---|---|---|
| Pipelined executor (Q87), first-finish race (Q93) | One-token decode stages back to back | ~55 % weight streaming (memory), ~25 % small kernels and kernel boundaries (core), the rest hops and host | 0.25–0.45 | 6–9 % | **2–4 % per stage** |
| Staggered conversations (Q104, Q120/Q121) | The same one-token stages; at N ≥ S_eff every card busy | as above; groups of two on Q78's 2-column kernel do more arithmetic per byte | 0.25–0.45; 0.35–0.55 for groups | 6–9 % | 2–4 %; 2–5 % for groups |
| Staircase drafter (Q20) | 2 dense layers + a 32k head, ~0.19 GB and ~30 launches per guess | memory and launches; a spare card stays at 25–40 % duty, card 1 reaches full duty | 0.3–0.5 | 0 % on a spare card; 6–9 % on card 1 | 0 on a spare card; 2–4 % on card 1's stage |
| Long prompt, full pipeline (Q109, `-ub 256/512` with four copies, Q108's layout) | MMQ tiles | `dp4a` issue and shared memory, both on the core clock ([prefill note](prefill_pascal.md) §2.2) | 0.85–1.0 | 6–9 % | **6–9 %** |

- **Why decode loses so little.** The cap lowers only the core clock: every capped sample in the logs still shows the
  full 4,004 MHz memory clock (all 2,181 capped samples at ≥ 90 % utilization). Weight streaming is set by memory
  bandwidth and by loads in flight, not by core cycles; the fork's Q5_0 kernel uses about half of the issue rate at
  138 GB/s (est. in the [hardware tuning note](hardware_tuning.md) §3), so 6–9 % fewer core cycles leave it
  memory-bound. What slows is the GPU-side fixed part of a token: norms, router, attention at short context, kernel
  ramp-up and tails.
- **The sensitivity comes from the rig.** Base-clock decode (1,354 MHz) against top-bin decode (~1,870 MHz, +38 %) gave
  +16–18 % on six and nine cards (section 3.1), i.e. a sensitivity of ~0.45 over that wide range (est.: assumes the
  whole gap is clock); the hardware tuning note's model gives 0.1–0.4 near the top bin. A 6–9 % clock loss is then
  2–4 %.
- **Prefill loses nearly all of it.** The calibration's single card ran pp512 at 1,690–1,790 MHz under the cap
  against its 1,860–1,911 MHz top bin; MMQ time scales with the core clock. Today's pp8192 already keeps several cards
  busy: the stock long-prompt logs show ~2.4 cards at ≥ 80 % at once with the cap in ~20 % of busy samples and
  1,820–1,840 MHz, i.e. est. 2–3 % lost today, 6–9 % once the pipeline keeps every card busy.
- **The cap shows only at high duty.** A card at ≤ 50 % duty rarely meets it (two-card calibration: 15–24 % of
  samples, 1,823–1,849 MHz); the drafter on a spare card, prefill with one or two micro-batches in flight, and every
  single-stream mode stay below it.

### 2.2 What the cap costs per v2 mode

- **Q87/Q93 single stream with guesses:** the cards work only as much as guesses keep them busy (D ≤ S_eff − 1), so
  duty rises toward 1 only at high acceptance. Est. 0–3 % on the overall speed-up; the drafter note's "1.97× at p 0.7
  with every stage 10 % slower" becomes ~2.2× at a 2–4 % stage loss (est., interpolated between its 2.32× and 1.97×).
- **Q104 at N ≥ S_eff:** the 235–256 tok/s estimate on nine cards includes a 10 % slowdown at its lower end; with
  2–4 % the range narrows to ~245–250 (est.).
- **Prefill at 2:1 prompt tokens per generated token (agent turns):** total generated speed falls by ~5–7 % (est.:
  prefill's 6–9 % weighted by its ~75 % share of card time at ρ = 2, ~6 ms per prompt token against ~3.8 ms per
  generated token on nine cards, plus decode's 2–4 % on the rest).
- **Energy:** the cap is what keeps an all-busy rig at ≤ 675 W of GPU power.

## 3. Safe settings (question 3)

### 3.1 PowerMizer mode: "Prefer Maximum Performance" against the base-clock decode

- **What the driver does by default.** NV-CONTROL's PowerMizer modes (`NV_CTRL_GPU_POWER_MIZER_MODE`, attribute 334,
  per GPU): ADAPTIVE "adjust GPU clocks based on GPU utilization"; PREFER_MAXIMUM_PERFORMANCE "raise GPU clocks to
  favor maximum performance, to the extent that thermal and other constraints allow"; AUTO "let the driver choose";
  PREFER_CONSISTENT_PERFORMANCE "lock to GPU base clocks". The default for a GPU is readable
  (`GPUPowerMizerDefaultMode`, attribute 400).
- **What the logs show.** In the stock Flint depth run, the tg64 at depth 0 started after idle: all cards ran in P0 at
  1,354 MHz with the full memory clock, 13–21 % utilization, 26–33 W and event reason 0x0, and a card that had been
  idling at 1,873 MHz dropped to 1,468 and then 1,354 MHz 3–7 s after the decode started. The 8k prompt fill of the
  next test lifted every card to its top bin, and the tg64 at depth 8k ran faster than at depth 0 despite the longer
  context (17.1 against 14.6 tok/s for UD-Q4_K_XL, 25.7 against 20.4 for the makers' Q4_1). The nine-card calibration's
  average of 1,598 MHz in decode most likely has the same cause (est.).
- **The same gap on the fork (speeds measured, clocks not logged in these runs):**

| Cards, build | tg128 started cold (`-p 0`) | tg after a prompt test in the same process | Gap |
|---|---|---|---|
| Six, v2 base, UD-Q4_K_XL | 29.86 tok/s (Q117 follow-up, alternating builds, each a fresh process) | 34.71 (Q117 headline, after pp8192) | +16 % |
| Nine, v2 base, UD-Q4_K_XL | 30.95 (`-t 2`; 28.85 at `-t 1`) (Q118) | 36.09 (Q118's warm-up tg32 after pp512); 36.5 (Q117, after pp8192) | +17–18 % |
| Nine, llama-server, 28-token prompt, 1,000 tokens (Q118) | 31.4–33.6 | – | – |

  In the Q117 follow-up the six cards read 1,354–1,670 MHz in P0 at the start of each run, before the model load
  (one reading per run).
  The fork's earlier power logs (Q38, Q46, step 1) all come from tg128 runs that followed a pp512 in the same
  process, which is why they show busy cards at their top bin. In those runs light decode held the top bin for the
  whole tg128 (~15 s); how long it holds, and when the policy steps down (as the idle card above did), is not
  documented.
- **Measured (Q141, 2026-10-24): mode 1 gives nothing and costs ~20 W per idle card.** UD-Q4_K_XL on six cards,
  mode 2 (Auto, the default) against mode 1 on all nine: tg128 from idle 29.79 ± 1.61 against 29.75 ± 1.66 tok/s,
  tg128 after a 2048-token prompt 35.07 against 34.31, a served 500-token reply after 60 s idle 33.30 against 33.25.
  Clocks every ~0.23 s: busy cards run at a median 1,847–1,911 MHz in both modes, and in both a card's first busy
  sample of a step reads 1,354 or 1,493 MHz, so mode 1 does not hold the top bin through the layer split's gaps. What it
  does hold is P0 at idle: 1,354 MHz SM, the full memory clock and 26 W per card, against P8 at 139 MHz and 5.7–6.0 W
  under Auto (~+180 W for nine idle cards). The cold start is a `llama-bench` tg-only effect: a served reply after a
  minute idle runs within ~5 % of the warm tg128. Mode 2 restored on every card; benchmarks warm the cards first.
  Data: [docs/data/2026-10-24_rig_q141](../../data/2026-10-24_rig_q141/README.md).
- **Where it can be set.** Only through NV-CONTROL on a running X server that drives the cards, as the desktop user:

```bash
# read-only (R0): does the X server expose the cards, and what are the modes now?
DISPLAY=:0 XAUTHORITY=<the desktop session's Xauthority> nvidia-settings -q gpus
DISPLAY=:0 XAUTHORITY=<…> nvidia-settings -q '[gpu:0]/GPUPowerMizerMode' -q '[gpu:0]/GPUPowerMizerDefaultMode' \
    -q '[gpu:0]/GPUCurrentPerfLevel' -q '[gpu:0]/GPUPerfModes'
# set (R2), per card; revert with the default read above
DISPLAY=:0 XAUTHORITY=<…> nvidia-settings -a '[gpu:0]/GPUPowerMizerMode=1'
```

  This rig's desktop runs on the Intel graphics while Xorg holds 4 MiB on each P106, so the cards may or may not be
  NV-CONTROL GPU targets; Coolbits is not needed for this attribute. NVML has no equivalent for Pascal GeForce-class
  cards: locked clocks are Volta and newer, application clocks are not supported on these cards, and auto boost "on
  Pascal and newer hardware" is "controlled through application clocks".
- **If the cards are not exposed:**
  - Operational: measure and serve warm. A prompt of a few thousand tokens lifts the cards; benchmarks run their tg
    after a prompt test, log clocks, and never compare cold and warm decode.
  - A second, bare X server for the NVIDIA cards only (the [hardware tuning note](hardware_tuning.md) §1.3, route B)
    would give NV-CONTROL without touching the desktop's configuration (est.); it is a larger step and needs the owner.
  - Headless mode (Q83) removes NV-CONTROL altogether, so Q83 should wait for R2's result.

### 3.2 Core clock offset

- **Interfaces:** `nvmlDeviceSetGpcClkVfOffset` ("works on Maxwell onwards", to be replaced by
  `nvmlDeviceSetClockOffsets`, which takes a P-state); getters `nvmlDeviceGetGpcClkVfOffset` and
  `nvmlDeviceGetGpcClkMinMaxVfOffset` (the allowed range); NV-CONTROL `GPUGraphicsClockOffsetAllPerformanceLevels` with
  Coolbits 8. "On GPUs based on the Pascal architecture the offset is applied to all performance levels" (driver
  README). Whether the P106-090 and driver 580 accept the NVML call is answered by the Q133 kit's read-only `probe`;
  P106-090 owners report core offsets up to +300 MHz applied under Linux mining distributions (report, in the
  [hardware tuning note](hardware_tuning.md) §1.2).
- **What it does under the cap (est.).** The offset shifts the whole voltage/frequency curve up: the same voltage gives
  Δ more MHz, at a little more power. Under the cap the controller then settles one or two voltage steps lower. With
  core power ∝ f·V², a curve slope of ~1,700 MHz per volt (a generic GP106 curve, ~0.8 V near 1,500 MHz to ~1.05 V
  near 1,900 MHz; this card's curve is not readable) and the capped point at ~1,730 MHz and ~0.93 V, the clock
  that survives is Δ · (1 − 1,700 / (1,700 + 2 · 1,730 / 0.93)) ≈ 0.69 Δ. So +100 MHz gives ~+70 MHz (+4 %) at the
  same 75 W, and the full +100 MHz wherever the cap does not bind.
- **Gains (est.):** decode in the layer split +1–2.5 % per +100 MHz (sensitivity 0.25–0.45 on +5.3 %); all-busy
  decode +1–2 %; prefill +4–5 % uncapped and +3–4 % capped. Because the offset also raises the 1,354 MHz level, it
  softens the base-clock decode of section 3.1 by ~+3 % (est.) where mode 1 is unavailable.
- **Risk and proof:** an unstable core gives wrong results or Xid errors. The validation is the Q133 kit with
  `--core` (offsets in MHz), +25 MHz steps (one boost bin is ~13 MHz), the exact check at each clean step, then a
  soak under all-busy load, where the card runs at lower voltages than at its top bin: both regimes need to pass.
  Rollback: offset 0; a hang needs a reboot.

### 3.3 Memory offset under the cap (Q133)

- In the layer split the memory offset is worth est. +3–5 % decode at +10 %
  ([hardware tuning](hardware_tuning.md) §2.1).
- Under the cap its extra 1–3 W (est.) comes out of the core: with core power ∝ f^~3.2 near the capped point (f·V²
  along the curve of section 3.2), 1–3 W of ~60 W of core power is 0.5–1.6 % of core clock, which costs decode
  ~0.2–0.7 %. Net est. +2.5–4.5 % on all-busy decode, still the largest decode lever after mode 1.
- A core offset of +25–50 MHz pays back the core clock the memory offset takes under the cap (est.).

### 3.4 Power limit lower, for tokens per joule

- **Interface:** `nvidia-smi -pl` ("Only on supported devices from Kepler family", root), NVML
  `nvmlDeviceSetPowerManagementLimit` (Kepler or newer, root; "not persistent across reboots or driver unloads"). Range
  60–75 W.
- **In the layer split:** nothing. Cards average 35–39 W; 60 W clips only the rare peaks of a prompt micro-batch.
- **In all-busy modes (est.):** with core power ∝ f·V² along the stock curve (which rises as ~f^3.2 near the capped
  point with the curve of section 3.2) and ~12–15 W of the 75 W in memory and board, 60 W leaves ~45 W of ~60 W for
  the core, which settles at ~1,550–1,600 MHz (−8 to −11 % against ~1,730). Decode then loses 2.5–5.5 % and prefill
  7–10 %, for 14–20 % less card power (from 70–75 W, depending on what the fork's kernels draw at full duty): tokens
  per joule +10–20 % on decode and +5–15 % on prefill (GPU), +7–17 % on decode at the wall with est. 60–95 W of fixed
  host and fan power plus losses. Nine cards would draw ~540 W instead of 630–675 W.
- **Owner's call:** it trades speed for energy. If the owner wants it, the natural place is long unattended batch work
  (agent suites, the race at high N); interactive work stays at 75 W.

### 3.5 Persistence mode

- `nvidia-smi -pm 1` / `nvmlDeviceSetPersistenceMode` (Linux only, root): "the driver software state is not torn down
  when the last client disconnects"; reset to disabled at every boot. While Xorg holds every card the driver never
  unloads, so it changes no speed and no power here; power limits are documented to need it headless. It belongs to
  Q83, as in the [hardware tuning note](hardware_tuning.md) §1.4.

### 3.6 Fans, cooling and temperature targets

- **The cards:** every card reports `fan.speed` 0 %. NVML's fan calls (`nvmlDeviceGetNumFans`,
  `nvmlDeviceSetFanSpeed_v2`, `nvmlDeviceSetFanControlPolicy`) are for "products with dedicated fans", and
  NV-CONTROL's cooler control needs Coolbits and a cooler target. With no fan reported, the cards rely on the chassis
  airflow (est.: whether a card has a fan header at all is not documented; `nvmlDeviceGetNumFans` answers it in R0).
- **The chassis:** ten 90 mm 4,800 rpm fans, five intake and five exhaust, run at a fixed speed and are not visible to
  the OS ([hardware spec](../../mining_rig.md)). Nothing in NVIDIA's interfaces reaches them.
- **Headroom:** 57 °C was the hottest reading (one card alone at full duty), 28–38 °C in layer-split decode. Nine cards
  at 70–75 W share one airflow, so inner cards will run hotter than in any log so far: est. 55–70 °C, still below the
  83 °C target, the point from which the driver lowers clocks for temperature.
- **Temperature target:** `nvidia-smi -gtt` ("Set GPU Target Temperature … within limits supported by GPU", root) and
  NVML's acoustic threshold (`nvmlDeviceSetTemperatureThreshold`, ACOUSTIC_CURR between ACOUSTIC_MIN and ACOUSTIC_MAX).
  Raising it gains nothing below 83 °C; lowering it only adds throttling. No action; R0 reads the allowed range.

### 3.7 What cannot be done on these cards through documented interfaces

- A power limit above 75 W (`power.max_limit` is 75 W; the VBIOS is signed and editing it is out of scope).
- Locked core clocks (`-lgc`, `nvmlDeviceSetGpuLockedClocks`: Volta+), locked memory clocks (Ampere+), application
  clocks (not supported on these cards) and auto-boost control (through application clocks on Pascal).
- Fan control on the cards (no fan reported) or on the chassis (not an NVIDIA device).
- An energy counter (`nvmlDeviceGetTotalEnergyConsumption`: Volta+) or a one-second power average (Ampere+): power must
  be integrated from instantaneous samples.
- Memory temperature or memory power: not reported.
- Overvoltage (Coolbits 16): excluded as unsafe; under a power cap more voltage also means a lower clock.

## 4. Measuring it (question 4)

### 4.1 Power per token and tokens per joule from `nvidia-smi`

Per run, on the rig, alongside the benchmark (field names as on driver 580; the older `clocks_throttle_reasons.*`
names are accepted too):

```bash
Q=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active
C=index,clocks_event_reasons_counters.sw_power_cap,clocks_event_reasons_counters.hw_thermal_slowdown,clocks_event_reasons_counters.hw_power_brake_slowdown
nvidia-smi --query-gpu=$C --format=csv,noheader > counters_before.csv
nvidia-smi --query-gpu=$Q --format=csv,noheader -lms 200 > gpu.csv & SMI=$!
# … the benchmark (llama-bench -o jsonl, llama-pipespec, llama-server) …
kill $SMI; nvidia-smi --query-gpu=$C --format=csv,noheader > counters_after.csv
```

- **Window:** llama-bench's JSONL gives each test's start (`test_time`, UTC) and its repetitions (`samples_ns`, plus
  one warm-up); the executor's traces give each token's time; llama-server's `timings` give prompt and decode times.
  Drop the first 0.6 s of each window (the clocks settle).
- **Energy:** for every card, sum power × the time to the next sample over the window, then sum all nine cards, idle
  ones included (they draw 6–33 W too). J per token = energy / tokens; tokens per joule is its inverse. Report decode
  and prompt tokens separately.
- **Sampling:** Pascal's reading is instantaneous, so 200 ms is a sample, not an average. A token's power pattern
  (27–36 ms) does not lock to 200 ms, so means converge; ≥ 30 s per window (≥ 150 samples per card) gives est. ±2 %.
  One query round of nine cards takes a few to ~40 ms, so the effective rate is ~4–5 Hz.
- **Cap share:** from the sampled bitmask (share of busy samples with 0x4). `clocks_event_reasons_counters.sw_power_cap`
  is documented as cumulative microseconds of capping ("Counters, in microseconds, for the amount of time factors have
  been reducing the frequency of clocks"), and after minus before over a window would be exact, but on these cards it
  reads impossible values on two of nine (section 1.5). Record it as a cross-check only, and trust it on a card only
  if its increase matches the sampled share within a few points in R1.
- **Per mode, what to record:** tokens per second, J per token (GPU), card power mean and peak, core clock mean while
  busy, capped share, maximum temperature, and whether each card was at base clock, top bin or capped.

### 4.2 A wall-power meter

- If the owner has a plug-in meter, it goes between the wall and the supply. Readings: all cards in P8 (no model
  loaded); a model loaded and idle; single-stream decode (six and nine cards); an all-busy run (R3); 2 min each,
  noting the meter's average watts and the `nvidia-smi` GPU sum over the same 2 min. Wall minus GPU sum is the host,
  fans and supply losses at each load, which turns every GPU J per token above into a wall figure.
- Energy (kWh) registers are too coarse for single runs (0.01 kWh is 36 kJ, a minute at 600 W): use the watt display or
  a meter with logging.

### 4.3 Reading the bitmask

`clocks_event_reasons.active` is a 64-bit mask; several bits can be set at once (NVML values):

| Bit | Name | On this rig |
|---|---|---|
| 0x1 | GPU idle | every idle card |
| 0x2 | Applications clocks setting | never (not supported) |
| 0x4 | SW power cap | busy cards at the 75 W limit |
| 0x8 | HW slowdown (set with 0x40 or 0x80; "reducing the core clocks by a factor of 2 or more") | never; a kill criterion |
| 0x10 | Sync boost | never |
| 0x20 | SW thermal slowdown | never; a kill criterion |
| 0x40 | HW thermal slowdown | never; a kill criterion |
| 0x80 | HW power brake ("External Power Brake Assertion … e.g. by the system power supply") | never; a kill criterion |
| 0x100 | Display clock setting | never |

For example 0x5 is idle and power cap at once, 0x0 means none. A busy card with 0x0 is either at its top bin (the
voltage limit) or held at base clock by the utilization policy: compare `clocks.sm` with 1,354 MHz and with that
card's top bin.

### 4.4 The exact output check

- **After an offset (memory or core):** the Q133 kit's `exact_check.sh`: the one-token llama-perplexity statistics
  block against a stock base file must equal a stock rerun's digit for digit (mean KLD prints 0.000000; the maximum
  is ~1e-5 because the base file stores log-probabilities in 16 bits), "same top p" 100 %, and the greedy-decoding
  hash identical. ~10 min with the model loads.
- **After a clock-policy or power-limit change (mode 1, `-pl`):** the arithmetic and the voltage/frequency table are
  unchanged by construction, so one greedy hash against stock (~3 min) is a sufficient sanity check.
- **In service:** a one-token KLD canary after each model load while any offset is set, as in the
  [hardware tuning note](hardware_tuning.md) §5.2.

## 5. Plan (question 5)

### 5.1 Rig session, ranked

`B` is the v2 base build, `M` UD-Q4_K_XL with `LLAMA_NO_MOE_DOWN_RESCALE=tg`, cards 0–5 unless stated; every step logs
as in 4.1.

| Step | What | Minutes | Approval | Pass / kill |
|---|---|---|---|---|
| R0 | Read-only (limits, thresholds and counters are already recorded): `nvidia-settings -q gpus` and the PowerMizer queries of 3.1; the Q133 kit's `probe` (core and memory offset ranges, power constraints); `nvmlDeviceGetNumFans` | 2 | No | Records which levers exist |
| R1 | **Cause of the cold-decode gap:** `$B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 3` (cold), then a second invocation with `-p 2048 -n 128 -r 3`, whose tg128 follows its prompt test; both logged | 10 | No | Cold tg at ~1,354 MHz and ~30 tok/s, warm tg at the top bin and ~34.7: cause confirmed. Equal clocks in both: the gap is not clocks; R2 is dropped |
| R2 | **PowerMizer mode 1** on cards 0–5: R1's cold tg again; one llama-server request (short prompt, 1,000 tokens); 2 min idle logged; greedy hash; default mode back | 15 | Yes | Keep if cold tg gains ≥ 3 % with the clocks at the top bin; record idle power. Kill if the attribute is not exposed or the clocks do not move |
| R3 | **All-busy power baseline**, inside the executor block (no extra time): Q104 experiment 2 (all-right oracle, depths 0–5 on six cards, 0–8 on nine) and pp8192 on the combined build, with the 200 ms log and the counters | +0 (logging only) | Tell the owner first: first sustained ~600 W GPU load | Per-card power, clock, capped share and J per token per depth. Kill (stop the run): any 0x8/0x20/0x40/0x80 bit, any Xid, a card above 75 °C |
| R4 | **60 W against 75 W** during R3's depth-4 oracle (six cards) and pp8192: `nvidia-smi -i 0,1,2,3,4,5 -pl 60`, run, `-pl 75` | 20 | Yes | Keep as an efficiency option if GPU tokens per joule rise ≥ 8 % with decode ≤ 5 % slower |
| R5 | Q133 memory offset on GPU0 | 110 | Approved (GPU0); owner near the console | Q133's criteria |
| R6 | **Core offset on GPU0** (Q135): the kit with `--core`, +25 MHz steps to +150; at the last clean step minus one, R3's pp8192 and depth-4 oracle with GPU0 set | 60 | Yes; owner at the console | Kill: < 2 % on pp8192 per +100 MHz on the card, or any check failing at the kept step |
| R7 | Wall meter, if the owner has one: the five states of 4.2 | 10 | Owner's meter | Fills section 1.3's host column |

R0–R2 fit into any gap between timed blocks (~30 min). R3 rides on the executor block. R4 and R6 need approval and
come after R3, because they only matter in the regime R3 measures.

### 5.2 Kill criteria for the whole programme

- Any hardware slowdown, thermal or power-brake bit (0x8, 0x20, 0x40, 0x80), any Xid in the kernel log, any card above
  75 °C (8 °C under the target): stop the run, put every setting back to stock.
- Any exact-check difference after an offset: that step fails; the kit steps back.
- Mode 1 that does not move the clocks, or that costs idle power without a speed gain: revert, keep the default.
- A lower power limit that loses more decode speed (in %) than it saves GPU energy per token: revert to 75 W.
- Anything that needs a configuration file, a package, an Xorg restart or a reboot is outside this round's "safe" and
  goes back to the owner as a separate decision.

## What this changes in the plan

- **Decode comparisons must hold the clock state equal.** tg128 run on its own starts at base clock (est.), tg after a
  prompt test at the top bin. For the combined build, the combined build with Q108 off, and the v2 base on six cards,
  the Q117 follow-up (tg alone, alternating builds) gave 30.6 / 30.6 / 29.9 tok/s and Q117's runs (tg after pp8192)
  33.5 / 36.8 / 34.7: the two sets differ by clock state as well as by build and order, which gives the v2
  synthesis's "llama-bench state" a mechanism. Q118's conclusion stands (server and llama-bench were both cold), but
  its absolute speeds are cold-decode speeds. Every rig script should log `clocks.sm` with the results and state
  whether decode started cold or warm. The synthesis's clock-logged pair of runs is R1 here.
- **PowerMizer mode 1 is not a lever:** measured in Q141, it gives nothing (section 3.1).
- **The executor's power regime (Q87/Q104) needs R3's logging:** `bench/pipespec/rig_run.sh`, `rig_streams.sh` and
  `rig_hostloop.sh` log every second without P-state, memory clock or event reasons; the 200 ms query of 4.1 and the
  counters belong in them.
- **The 10 % cap slowdown in the Q87, Q104 and Q20 projections is conservative for decode** (est. 2–4 %) and about
  right for prefill (6–9 %).
- **E2.16 (power-limit sweep) becomes R4,** an efficiency option for all-busy batch work, not a speed setting.
- **The hardware spec's power line** can say that power follows duty: ~70 W and capped for a card doing all the work,
  ~35–39 W per card in a layer split, ~6 W idle in P8 and 26–33 W waiting in P0.
- Suggested registry rows (not added here): Q136 PowerMizer mode (R1, R2); Q137 all-busy power baseline and the 60 W
  efficiency test (R3, R4); Q135 gains the under-cap arithmetic and R6.

## Already covered

| Finding | Where it is |
|---|---|
| Power-limit range 60–75 W, no limit above 75 W; busy layer-split cards at 33–38 W and their top bin, cap in < 4 % of busy samples | [Hardware tuning](hardware_tuning.md) §3; Q86 |
| Memory offset through NVML, its decode gain, the GPU0 protocol and exact check | [Hardware tuning](hardware_tuning.md) §1, §2, §5; Q133; [bench/mem_offset](../../../bench/mem_offset/README.md) |
| Core offset as a lever, `-lgc` Volta+, application clocks N/A, persistence mode headless | [Hardware tuning](hardware_tuning.md) §3, §1.4; Q135, Q83 |
| PCIe 2.0 impossible; ASPM off | [Hardware tuning](hardware_tuning.md) §4, §6 |
| Single-card power, clocks and cap shares; split relieves the cap | [Single-card calibration](../../benchmarks/2026-10-22-single-card-calibration.md) |
| The ~5 ms per token between llama-bench after prompt tests and llama-bench alone or served, Q117's decode loss as llama-bench state, a clock-logged pair of runs proposed; power per token est. 9–11 J on nine cards | [v2 synthesis](v2_synthesis.md) |
| A 10 % stage slowdown from the cap in the all-busy projections | [Multi-conversation serving](multi_conversation_serving.md) §1.5; [pipeline drafter](pipeline_drafter.md) |
| Power and tokens-per-joule logging at 100–250 ms, wall meter for system power | [Operations and benchmarking](ops_and_benchmarking.md) §4 |
| Prefill is bound by `dp4a` issue and shared memory | [Prompt processing](prefill_pascal.md) §2.2 |

## Not verified

- That the cold-decode gap on the fork is all clock: the fork's cold and warm runs were not logged with clocks (R1).
  Whether the served chat benchmarks so far ran at base clock (their speeds stayed flat over five turns, which fits
  either state held throughout).
- That this rig's X server exposes the nine cards as NV-CONTROL GPU targets, the default PowerMizer mode, whether
  mode 1 holds the clocks during light decode, and its idle-power cost; that `nvidia-settings` is installed on the rig.
- The power draw, clock and capped share of the fork's kernels at full duty (R3); the clock sensitivities 0.25–0.45
  (decode) and 0.85–1.0 (prefill) are fitted or modelled, not measured per mode.
- The voltage/frequency curve of the P106-090 (slope, voltages); the 0.69 · Δ and 60 W clock estimates rest on a
  generic GP106 curve and core power ∝ f·V².
- That NVML's core offset is accepted on the P106-090 under driver 580, and its units (the kit's probe).
- Host, fan and supply-loss power (no wall meter); the supply's model and efficiency.
- How the board wires its seven 6-pin inputs to the nine slots, whether the cards have power connectors of their own,
  and the board's power input for the host.
- The CEM figures for the 6-pin connector and the slot were taken from NVIDIA's and Intel's documents and the
  specification's common citation; the specification itself (members only) was not read.
- Whether the cards have fan headers (`nvmlDeviceGetNumFans`), and the allowed range of the temperature target.
- How much of the clock drop with temperature in the logs is temperature rather than power.
- The `clocks_event_reasons_counters.*` query fields on driver 580's `nvidia-smi` (`--help-query-gpu`; `-q` shows
  the counters), and why the power-capping counter reads impossible values on two cards and very high values on
  others.

## Sources

NVIDIA documentation and headers (P)
- NVML API reference, device commands: `nvmlDeviceSetPowerManagementLimit` (Kepler or newer, root, not persistent),
  `nvmlDeviceSetPersistenceMode`, `nvmlDeviceSetGpcClkVfOffset` and `nvmlDeviceSetMemClkVfOffset` ("Maxwell onwards",
  to be replaced by `nvmlDeviceSetClockOffsets`), `nvmlDeviceSetGpuLockedClocks` (Volta+),
  `nvmlDeviceSetMemoryLockedClocks` (Ampere+), `nvmlDeviceSetFanSpeed_v2` and `nvmlDeviceSetFanControlPolicy`
  (products with fans), `nvmlDeviceSetTemperatureThreshold`, `nvmlDeviceSetAutoBoostedClocksEnabled` (Pascal: through
  application clocks), <https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceCommands.html>
- NVML API reference, device queries: `nvmlDeviceGetPowerUsage` (Ampere+ averages over 1 s), `nvmlDeviceGetTotalEnergyConsumption`
  (Volta+), `nvmlDeviceGetEnforcedPowerLimit`, `nvmlDeviceGetPowerManagementLimitConstraints`, `nvmlDeviceGetFanSpeed`,
  `nvmlDeviceGetNumFans`, `nvmlDeviceGetPersistenceMode`, `nvmlDeviceGetClockOffsets`,
  `nvmlDeviceGetCurrentClocksEventReasons`, <https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html>
- NVML clock event reasons (descriptions), <https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlClocksEventReasons.html>;
  bit values, <https://docs.nvidia.com/deploy/archive/R535/nvml-api/group__nvmlClocksEventReasons.html>;
  `nvmlTemperatureThresholds_t` in `nvml.h`, <https://github.com/NVIDIA/go-nvml/blob/main/gen/nvml/nvml.h>
- nvidia-smi documentation: `-pm`, `-pl`, `-lgc`/`-rgc` (Volta+), `-ac` (deprecated), `-gtt`, power fields, clock
  event reasons and their counters, `fan.speed` ("Many parts do not report fan speeds because they rely on cooling via
  fans in the surrounding enclosure"), temperature thresholds, <https://docs.nvidia.com/deploy/nvidia-smi/index.html>
- NV-CONTROL header `NVCtrl.h`: `NV_CTRL_GPU_POWER_MIZER_MODE` (334, modes 0–3), `NV_CTRL_GPU_POWER_MIZER_DEFAULT_MODE`
  (400), `NV_CTRL_GPU_CURRENT_PERFORMANCE_LEVEL` (265), cooler control (319, 320),
  <https://github.com/NVIDIA/nvidia-settings/blob/main/src/libXNVCtrl/NVCtrl.h>
- NVIDIA Linux driver README (580 series), X configuration options: Coolbits (bit 3 offsets, Pascal applies them to
  all performance levels; bit 4 overvoltage; per X screen),
  <https://download.nvidia.com/XFree86/Linux-x86_64/580.95.05/README/xconfigoptions.html>
- NVIDIA, "Power Guidelines for Workstation Products", application note DA-07261-001_v17 (September 2025): slot 75 W,
  8-pin 150 W and up to 175 W in practice, 6-pin and 8-pin pinouts, Quadro P2000 at 75 W without an auxiliary
  connector, Y-splitter caution; as hosted by PNY,
  <https://www.pny.com/en-eu//File%20Library/Professional/DATASHEET/DATACENTER%20CARDS/Quick%20Start%20Guide/gpus-power-guidelines.pdf>

Power connectors and slots (P, read through vendor documents)
- PCI-SIG, PCI Express Card Electromechanical Specification: x16 slot 75 W, 2×3 auxiliary connector 75 W, 2×4 150 W
  (members only; not read)
- Intel support article 000019188: "the maximum power supported directly from a x16 PCIe add-in card slot = 75W",
  <https://www.intel.com/content/www/us/en/support/articles/000019188.html>

Vendor (P)
- Manli M-P106L9-N6G product page (system power while mining: 800 W for Ethereum, 600 W for Monero; 1,600 W supply;
  7 × 6-pin), as summarized in the [hardware spec](../../mining_rig.md)

This project (P)
- `nvidia-smi` logs of the calibration, the Flint and Qwen sweeps (2026-10-22), Q38, Q46 and the rig step-1 runs
  (2026-10-23) (raw logs kept outside the repository); the rig inventory E0.1; the idle `nvidia-smi -q` power and
  temperature query of 2026-10-24
- [Quant registry](../../../quants/experiments.md) (Q20, Q38, Q46, Q83, Q86, Q87, Q93, Q104, Q108, Q109, Q117, Q118,
  Q120, Q121, Q133–Q135), [single-card calibration](../../benchmarks/2026-10-22-single-card-calibration.md),
  [hardware spec](../../mining_rig.md), [hardware tuning](hardware_tuning.md),
  [multi-conversation serving](multi_conversation_serving.md), [pipelined speculation](pipelined_speculation.md),
  [pipeline drafter](pipeline_drafter.md), [prompt processing](prefill_pascal.md),
  [host in the decode loop](host_decode_loop.md), [bench/mem_offset](../../../bench/mem_offset/README.md)

Reports (S)
- P106-090 core offsets under Linux mining distributions (HiveOS forum, 2019–2021), as cited in the
  [hardware tuning note](hardware_tuning.md) §1.2; GPU Boost temperature bins (general reports, not verified here)
