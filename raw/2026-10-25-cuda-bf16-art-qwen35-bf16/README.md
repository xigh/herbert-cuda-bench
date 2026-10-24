# 2026-10-25 · Qwen3.5 0.8B, 2B et 4B en bf16 sur la 4080 : herbert-cuda contre TensorRT-LLM, llama.cpp, vLLM et SGLang, prefill de 128 à 4 000 tokens et decode p3 (pour l'article #003)

Comparaison ajoutée le 2026-10-24 au soir pour l'article #003 : Qwen3.5 0.8B, 2B et 4B en bf16, herbert-cuda face à TensorRT-LLM, llama.cpp, vLLM et SGLang, à plusieurs tailles de prompt.

## Machine et versions (`runs/campaign/env.txt`)

**Machine** :
- rtx4080, RTX 4080, pilote 595.99.02 ;
- horloges verrouillées à 2 805 / 11 201 MHz, vérifiées avant chaque unité, sans alerte `!!` ;
- relevé à la seconde (`clocks.csv`) : 2 277 lignes sur 2 287 à 2 805 / 11 201 MHz en P0, les 10 autres entre 2 715 et 2 790 MHz ;
- profil `CudaNoStablePerfLimit` (P0 pour tous les moteurs CUDA) ;
- `resident-gpu.service` arrêté puis relancé, actif à la fin.

**Moteurs** :
- **herbert-cuda** : herbert-cuda-bf16 `31b86e1`, le code de `6701002` (seuls des fichiers de notes ont changé depuis ; son PTX par défaut est celui de `f0b646a`, le tag `v0.1.0` publié), dans `~/herbert-cuda-bf16-art`. Ses réglages par défaut : `sdpa.gqa=1 sdpa.split=64 sdpa.pipe=1` (depuis `f0b646a`) et `gemm.auto=3 gemm.layout=xor prefill.block=auto`.
- **llama.cpp** b11718 (`547208dca`), CUDA, GGUF BF16 de `$BENCH_DIR`. Le 2B a été converti pour cette campagne par `convert_hf_to_gguf.py` de b11718 (`runs/convert-2b.log`, empreinte dans ce fichier et dans `env.txt`).
- **vLLM** 0.31.0 (`~/venvs/bench-vllm-latest` ; la ligne de `env.txt` est vide, la version vient de ses journaux : « LLM engine (v0.31.0) »).
- **SGLang** 0.5.22.
- **TensorRT-LLM** 1.4.0rc1 (backend PyTorch).

Tout est en bf16 : poids, activations et cache K/V.

## Méthode (`_scripts/_art-003.sh`)

**herbert-cuda** :
- prefill : `--ask-file pp<N>.txt`, sans réflexion, 1 token, `--warmup 1`, la ligne `[turn 1]`. Prompts de 128, 256, 513, 1 024, 2 048 et 4 001 tokens.
- decode : `p3-long` (759 tokens) avec réflexion, 255 tokens générés à chaque fois.

**llama.cpp** :
- `llama-bench -p 128,256,512,1024,2048,4000 -n 0 -r 1`, deux réglages : défaut (`-ub 512 -b 2048`, `-fa` auto) et `-ub 1024 -b 4096 -fa on`, les meilleurs réglages de prefill de la phase 1 de #003 ;
- decode : `-p 0 -n 255 -d 759`, défaut et `-fa on`.

**Ordre** : herbert-cuda et llama.cpp alternés tour par tour, ordre inversé un tour sur deux, **N = 5**. Les colonnes donnent la médiane [min..max] des 5 tours.

**vLLM** :
- `bench latency`, 2 itérations de chauffe et 5 mesurées, `--max-model-len 4608 --max-num-seqs 8 -cc.max_cudagraph_capture_size=512` ;
- prefill = entrée / latence avec 1 token généré ;
- decode = 254 / (L(759 → 255) − L(759 → 1)).

**SGLang** : `one_batch`, chaque taille 3 fois avec 1 token généré (médiane de son `prefill_throughput`), puis 759 → 255 trois fois (médiane de son `median_decode_throughput`).

**TensorRT-LLM** : `_scripts/_trtllm_latency.py`, même méthode que vLLM. Prompt d'ids aléatoires (graine 0), `max_seq_len` 4608, graphe CUDA pour un lot de 1, `TLLM_WORKER_USE_SINGLE_PROCESS=1`, nvcc 13.0 de son environnement pour DeepGEMM.

vLLM, SGLang et TensorRT-LLM tournent chacun en un passage par modèle, avec leurs propres répétitions, après les tours alternés.

**Faisabilité** (`runs/feas/`, `_scripts/_feas-art.sh`) : TensorRT-LLM en bf16 sur les trois modèles, et chaque moteur sur le 2B (nouveau dans #003). Tout passe.

## Valeurs (`runs/campaign/summary.md`, `_scripts/_art-summary.py`)

Prefill et decode en tok/s ; une seule valeur pour vLLM et TensorRT-LLM, qui moyennent leurs 5 itérations.

### Qwen3.5-0.8B

| Moteur | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4000 | decode p3 |
|---|---|---|---|---|---|---|---|
| herbert-cuda | 28 181 [28100..28299] | 45 624 [45582..45706] | 50 013 [49957..50096] | 61 082 [60960..61092] | 63 677 [63561..63762] | 61 820 [61725..62321] | 404,6 [404,6..404,6] |
| llama.cpp défaut | 16 065 [15891..16116] | 26 817 [26665..26933] | 34 669 [34533..34724] | 34 462 [34383..34513] | 38 341 [38272..38401] | 38 475 [38457..38546] | 325,3 [325,0..325,3] |
| llama.cpp ub1024-b4096-fa | 16 049 [15572..16237] | 26 722 [26547..27056] | 34 730 [34648..34831] | 38 556 [38451..38632] | 39 434 [39317..39485] | 40 810 [40709..40864] | 325,3 (`-fa on`) |
| vLLM | 11 156 | 21 799 | 42 182 | 53 749 | 54 021 | 53 254 | 365,8 |
| TensorRT-LLM | 9 407 | 19 404 | 38 566 | 59 036 | 64 588 | 58 224 | 378,4 |
| SGLang | 14 242 [14198..14975] | 29 537 [28532..30118] | 47 235 [46525..47244] | 58 709 [57969..58925] | 64 187 [63728..64216] | 58 410 [49749..58529] | 332,8 |

### Qwen3.5-2B

| Moteur | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4000 | decode p3 |
|---|---|---|---|---|---|---|---|
| herbert-cuda | 15 170 [15051..15174] | 22 576 [22552..22591] | 22 783 [22750..22803] | 25 895 [25883..25933] | 26 130 [26103..26152] | 25 598 [25509..25628] | 174,7 |
| llama.cpp défaut | 10 352 [10149..10383] | 16 405 [16306..16443] | 19 492 [19482..19517] | 19 829 [19808..19902] | 21 305 [21277..21312] | 21 176 [21147..21190] | 158,3 |
| llama.cpp ub1024-b4096-fa | 10 365 [10121..10371] | 16 356 [16237..16447] | 19 526 [19416..19533] | 20 397 [20365..20441] | 20 673 [20650..20705] | 21 083 [21069..21122] | 158,3 (`-fa on`) |
| vLLM | 10 218 | 17 725 | 19 917 | 23 860 | 26 091 | 26 007 | 166,3 |
| TensorRT-LLM | 8 873 | 18 091 | 23 447 | 26 294 | 28 113 | 27 256 | 168,9 |
| SGLang | 12 050 [11980..12426] | 18 797 [18480..18888] | 23 416 [23363..23419] | 26 639 [26603..26644] | 28 255 [28211..28271] | 27 430 [26138..27433] | 159,2 |

### Qwen3.5-4B

| Moteur | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4000 | decode p3 |
|---|---|---|---|---|---|---|---|
| herbert-cuda | 6 522 [6518..6524] | 9 338 [9303..9385] | 9 678 [9653..9702] | 11 352 [11287..11467] | 11 690 [11675..11722] | 11 402 [11394..11406] | 78,7 [78,5..78,8] |
| llama.cpp défaut | 5 288 [5283..5304] | 7 937 [7913..7947] | 8 927 [8924..8932] | 9 002 [8971..9009] | 9 396 [9392..9403] | 9 256 [9246..9260] | 74,4 |
| llama.cpp ub1024-b4096-fa | 5 239 [5222..5283] | 7 918 [7911..7944] | 8 914 [8912..8932] | 8 817 [8808..8830] | 8 833 [8818..8853] | 8 858 [8857..8863] | 74,4 (`-fa on`) |
| vLLM | 5 329 | 8 176 | 9 885 | 10 382 | 10 984 | 10 886 | 76,2 |
| TensorRT-LLM | 6 172 | 9 314 | 10 614 | 10 921 | 11 455 | 11 172 | 77,0 |
| SGLang | 6 070 [6057..6171] | 9 255 [9219..9259] | 10 674 [10674..10687] | 11 018 [11003..11036] | 11 364 [11357..11376] | 11 324 [11036..11332] | 74,6 |

## Différences de mesure qui restent

- **Prompts** : texte réel tokenisé pour herbert (pp512 = 513 tokens, pp4000 = 4 001), prompt synthétique de `llama-bench`, ids aléatoires pour vLLM, TensorRT-LLM (graine 0) et SGLang.
- **Mesure du prefill** :
  - herbert : temps de son prefill dans le processus ;
  - llama.cpp : `pp` de `llama-bench` ;
  - vLLM et TensorRT-LLM : latence de bout en bout d'une requête qui génère un token, ordonnancement et échantillonnage compris ;
  - SGLang : son `prefill_latency`.
- **Mesure du decode** :
  - herbert : 255 tokens avec réflexion, à partir de la position 759 ;
  - llama.cpp : 255 tokens à la profondeur 759 ;
  - vLLM et TensorRT-LLM : différence de deux latences ;
  - SGLang : médiane de ses pas.

## Fichiers

- `runs/campaign/` : toutes les sorties, `env.txt`, `clocks.csv`, `summary.md`, `summary.json` ;
- `runs/feas/` : la faisabilité ;
- `runs/convert-2b.log`, `runs/build-herbert.log` ;
- `_scripts/` : identiques à ceux lancés.

Les valeurs seulement.
