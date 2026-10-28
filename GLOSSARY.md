# LLMs on crypto-mining rigs

Running language models on old mining hardware such as the Manli rig with nine P106-090 cards. These are the words this
project uses for model files and for how fast those cards run them.

## Language

### Machines

**Rig**:
The mining machine whose speed this project measures: a Manli M-P106L9-N6G with nine P106-090 cards.
_Avoid_: box, server

**Quant machine**:
Any other computer used to make and score quants; speed is never measured on it.
_Avoid_: rig, lab

### Software

**Miner Llama**:
This project and its llama.cpp fork for GPUs of compute capability 6.1, one code line for every model; v1 and v2 are its
releases.
_Avoid_: Pascal engine, Flint fork, model branch

**Model page**:
The folder `models/<model>/` that holds one model's recommended file, command line, and measured speed and quality.
_Avoid_: profile

### Models

**Hybrid model**:
A model whose layers mix attention, which keeps a cache of every past token, with recurrent layers such as Gated
DeltaNet, which keep a fixed-size state instead; the Ornith-1.5 and Qwen3.6 models are hybrid.
_Avoid_: recurrent model, linear-attention model, SSM

**MTP head**:
A small extra network that predicts the next tokens from the model's own hidden states, for speculative decoding;
Ornith-1.5 ships one in its files, and for Flint this project trains one.
_Avoid_: guesser, drafter, draft head

**N-gram table**:
A model's lookup table of embeddings for hashed sequences of two and three tokens, read a few rows per token and kept
apart from the other weights; Qwen3.8-Flash-Next has one.
_Avoid_: PLE table, per-layer embedding, engram

### Model files

**Format**:
The block encoding a tensor's weights are stored in, such as Q4_0, Q4_1, Q5_0, Q8_0, Q4_K or IQ4_XS.
_Avoid_: quant type, quantization (for the encoding)

**Mix**:
The assignment of a format to every tensor of a model file.
_Avoid_: layout, allocation, quant scheme

**Recipe**:
The quantizer options that reproduce a mix from a named source file and importance matrix.
_Avoid_: config, settings

**Mirror**:
A mix that keeps another quant's per-tensor bit budget but replaces each format with a fast format.
_Avoid_: format change

### Speed

**Decode efficiency**:
Effective bandwidth during generation (weight bytes read per token × tokens per second) divided by the card's measured
peak memory bandwidth.
_Avoid_: unpacking cost (its inverse), bandwidth utilisation

**Fast format**:
A format the rig's cards decode at least as efficiently as Q4_0, established by measurement; today Q4_0, Q4_1, Q5_0 and
Q8_0.
_Avoid_: simple format

**Pascal-friendly**:
Of a model file: decoded and prefilled fast by the rig's cards, through fast formats or through fewer bytes in formats
those cards still decode well; its formats need not be made for Pascal.
_Avoid_: Pascal-specific, Pascal-optimised

### Quality

**Within noise**:
A quality difference smaller than about two standard errors on the standard quality test: same top token and mean KL
divergence against the reference, over 80 × 512 tokens of wikitext-2.
_Avoid_: lossless, same quality
