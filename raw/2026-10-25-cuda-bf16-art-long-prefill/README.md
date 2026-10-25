# 2026-10-25 · Qwen3.5 0.8B, 2B et 4B en bf16 sur la 4080 : prefill long (2 000 à 40 000 tokens), herbert-cuda contre TensorRT-LLM, llama.cpp, vLLM et SGLang (pour l'article #003)

Mesure ajoutée le 2026-10-25 au matin pour l'article #003, à la suite de la mesure de 128 à 4 000 tokens : prefill à 2 000, 5 000, 10 000, 20 000 et 40 000 tokens, pour les cinq moteurs et les trois modèles. Même méthode que la mesure de 128 à 4 000 tokens (`../2026-10-25-cuda-bf16-art-qwen35-bf16/`), prefill seul : le decode p3 est celui de la mesure de 128 à 4 000 tokens.

## Machine et versions (`runs/campaign-long/env.txt`)

**Machine** :
- rtx4080, RTX 4080, pilote 595.99.02, profil `CudaNoStablePerfLimit` (P0) ;
- horloges verrouillées à 2 805 / 11 201 MHz, vérifiées avant chaque unité : aucune alerte `!!` ;
- **sous charge, l'horloge SM descend** : la limite de puissance de la carte est de 320 W, et les prefills longs l'atteignent (311 à 322 W relevés). Relevé à la seconde (`clocks.csv`, `clocks-trt-mntN.csv`) : 325 lignes sur 755 sous 2 805 MHz pendant les tours alternés herbert-cuda / llama.cpp (04:48–05:01), 167 sur 2 776 pendant vLLM, SGLang et TensorRT-LLM (05:01–05:47), 38 sur 455 pendant la passe complémentaire de TensorRT-LLM. Minimum 2 625 MHz. La mémoire est restée à 11 201 MHz sur toutes les lignes. Le relevé à la seconde ne dit pas quel moteur tournait à chaque ligne pendant les tours alternés ; la mesure de 128 à 4 000 tokens (prompts ≤ 4 000) avait 10 lignes sur 2 287 sous 2 805 MHz ;
- `resident-gpu.service` arrêté puis relancé à la fin de chaque script, actif.

**Moteurs** (les mêmes que la mesure de 128 à 4 000 tokens) :
- herbert-cuda-bf16 `31b86e1`, réglages par défaut (`sdpa.gqa=1 sdpa.split=64 sdpa.pipe=1 gemm.auto=3 gemm.layout=xor prefill.block=auto`), `--ctx 40960` ;
- llama.cpp b11718 CUDA, GGUF BF16 ;
- vLLM 0.31.0 ;
- SGLang 0.5.22 ;
- TensorRT-LLM 1.4.0rc1 (backend PyTorch).

## Prompts (`runs/prompts-long/`)

pp10000, pp20000 et pp40000 sont ceux de la phase 5 de #003 (`~/c003/long-prompts`, prose de Moby-Dick). pp2000 et pp5000 sont construits pour cette campagne par `_scripts/_build_pp_prompts.py`, même méthode, à partir du token 130 000 du livre (`runs/prompts-long/_build.log`). La longueur compte les 12 tokens du gabarit de herbert. Lignes `[turn 1]` de herbert : 2 001, 5 000, 10 000, 20 000 et 40 000 tokens. Les autres moteurs gardent leur propre prompt : synthétique pour `llama-bench`, ids aléatoires pour vLLM, TensorRT-LLM (graine 0) et SGLang.

## Méthode (`_scripts/_art-long-003.sh`, `_scripts/_art-long-trt-mntN.sh`)

- **herbert-cuda** : `--ctx 40960 --no-system --no-think --max-tokens 1 --warmup 1 --ask-file ppN.txt`, ligne `[turn 1]`.
- **llama.cpp** : `llama-bench -p 2000,5000,10000,20000,40000 -n 0 -r 1`, réglage par défaut et `-ub 1024 -b 4096 -fa on`.
- herbert-cuda et llama.cpp **alternés tour par tour, N = 5**, ordre inversé un tour sur deux : médiane [min..max].
- **vLLM** : `bench latency`, entrée N, 1 token, 2 chauffes et 5 itérations, `--max-model-len 40960 --max-num-seqs 8 -cc.max_cudagraph_capture_size=512`. Deux réglages : attention par défaut (FLASH_ATTN d'après ses journaux) et `--attention-backend FLASHINFER`, le meilleur de la phase 5a aux longueurs longues. À ce contexte, vLLM découpe le prefill par blocs de 8 192 tokens (`max_num_batched_tokens=8192`, journaux).
- **TensorRT-LLM** : `_trtllm_latency.py` (celui de la mesure de 128 à 4 000 tokens, avec une option `--chunked-prefill` ajoutée), entrée N, 1 token, 2 chauffes et 5 itérations, `max_seq_len` 40960. Trois réglages :
  - **un bloc, `max_num_tokens` 40960** : le moteur chauffe une passe de 40 960 tokens au démarrage, ce qui fait manquer de mémoire le 4B à toutes les longueurs (256 Mio) ;
  - **un bloc, `max_num_tokens` = N** (passe complémentaire) : le 4B passe jusqu'à 20 000 tokens, 40 000 manque de mémoire ;
  - **blocs de 8 192** (`enable_chunked_prefill`, `max_num_tokens` 8192, comme vLLM).
- **SGLang** : `one_batch`, entrée N ×3, 1 token, `--context-length 40960 --mem-fraction-static 0.7 --max-running-requests 1`, un processus par longueur : médiane [min..max] de ses 3 `prefill_throughput`. **4B à 40 000 : aucun réglage essayé ne passe** (`runs/feas-long*/`) : mf ≤ 0.73 laisse un pool KV sous 40 000 tokens (34 186 à 0.7, 39 436 à 0.72) ; de 0.74 à 0.78, avec ou sans graphes CUDA et avec `expandable_segments`, la mémoire manque dans le prefill DeltaNet de FLA (`recompute_w_u_fwd`). En f16, la phase 5b de #003 avait le même trou.
- vLLM, SGLang et TensorRT-LLM tournent en un passage par modèle, après les tours alternés.

**Valeurs de vLLM et TensorRT-LLM** : N / latence moyenne des 5 itérations, comme dans la mesure de 128 à 4 000 tokens, puis entre parenthèses N / latence médiane des mêmes 5. Elles diffèrent surtout sur TensorRT-LLM 0.8B à pp2000 : avec `mnt N` et avec les blocs de 8 192, 2 itérations sur 5 prennent 160 à 172 ms au lieu de 33 ms (`trtllm/0.8B-*-in2000-out1.json`).

## Valeurs (`runs/campaign-long/summary.md`, `_scripts/_art-long-summary.py`)

Prefill en tok/s.

### Qwen3.5-0.8B

| Moteur | pp2000 | pp5000 | pp10000 | pp20000 | pp40000 |
|---|---|---|---|---|---|
| herbert-cuda | 62 615 [62365..62785] | 59 227 [58787..59389] | 55 857 [55676..56197] | 48 708 [48510..48825] | 37 800 [37749..38005] |
| llama.cpp défaut | 37 037 [36858..37114] | 38 137 [38080..38203] | 36 907 [36809..36929] | 33 920 [33888..34015] | 29 161 [29093..29244] |
| llama.cpp ub1024-b4096-faon | 37 712 [37577..37824] | 40 862 [40787..40904] | 39 552 [39528..39561] | 36 367 [36294..36431] | 30 799 [30725..30860] |
| vLLM défaut | 54 184 (méd. 53 479) | 52 421 (méd. 52 307) | 49 907 (méd. 49 917) | 44 862 (méd. 44 861) | 35 830 (méd. 35 822) |
| vLLM FLASHINFER | 53 518 (méd. 52 886) | 52 270 (méd. 52 110) | 50 866 (méd. 50 862) | 45 542 (méd. 45 511) | 36 893 (méd. 36 897) |
| TensorRT-LLM un bloc, mnt 40960 | 61 793 (méd. 62 246) | 55 936 (méd. 55 937) | 51 660 (méd. 51 654) | 44 018 (méd. 44 045) | 35 101 (méd. 35 096) |
| TensorRT-LLM un bloc, mnt N | 23 315 (méd. 60 757) | 55 375 (méd. 55 889) | 51 333 (méd. 51 608) | 44 016 (méd. 44 012) | 35 082 (méd. 35 080) |
| TensorRT-LLM blocs de 8192 | 21 869 (méd. 40 206) | 55 140 (méd. 55 260) | 50 612 (méd. 50 613) | 44 184 (méd. 44 188) | 34 157 (méd. 34 155) |
| SGLang | 62 650 [62090..62717] | 57 897 [57714..58011] | 54 592 [54557..54596] | 47 143 [47111..47217] | 38 090 [38086..38094] |

### Qwen3.5-2B

| Moteur | pp2000 | pp5000 | pp10000 | pp20000 | pp40000 |
|---|---|---|---|---|---|
| herbert-cuda | 25 572 [25480..25721] | 25 336 [25279..25426] | 24 416 [24399..24537] | 22 746 [22677..22836] | 19 703 [19696..19769] |
| llama.cpp défaut | 20 742 [20725..20767] | 21 035 [21007..21040] | 20 812 [20799..20834] | 19 774 [19752..19805] | 18 027 [18022..18074] |
| llama.cpp ub1024-b4096-faon | 20 075 [20037..20145] | 21 110 [21082..21122] | 20 972 [20935..20983] | 20 098 [20071..20106] | 18 307 [18280..18310] |
| vLLM défaut | 25 604 (méd. 25 628) | 25 911 (méd. 25 897) | 25 885 (méd. 25 882) | 24 607 (méd. 24 599) | 21 740 (méd. 21 735) |
| vLLM FLASHINFER | 25 568 (méd. 25 532) | 25 995 (méd. 25 975) | 26 070 (méd. 26 066) | 24 816 (méd. 24 789) | 22 037 (méd. 22 034) |
| TensorRT-LLM un bloc, mnt 40960 | 27 515 (méd. 27 519) | 26 748 (méd. 26 754) | 26 174 (méd. 26 172) | 24 444 (méd. 24 444) | 21 509 (méd. 21 511) |
| TensorRT-LLM un bloc, mnt N | 27 692 (méd. 27 725) | 26 692 (méd. 26 686) | 26 183 (méd. 26 187) | 24 450 (méd. 24 449) | 21 511 (méd. 21 510) |
| TensorRT-LLM blocs de 8192 | 27 362 (méd. 27 380) | 26 369 (méd. 26 376) | 25 831 (méd. 25 832) | 24 278 (méd. 24 280) | 21 026 (méd. 21 025) |
| SGLang | 27 621 [27461..27631] | 27 216 [27157..27228] | 26 906 [26871..26909] | 25 399 [25393..25402] | 22 609 [22585..22618] |

### Qwen3.5-4B

| Moteur | pp2000 | pp5000 | pp10000 | pp20000 | pp40000 |
|---|---|---|---|---|---|
| herbert-cuda | 11 413 [11388..11421] | 11 267 [11264..11276] | 10 809 [10769..10841] | 9 978 [9966..10013] | 8 679 [8670..8695] |
| llama.cpp défaut | 9 175 [9171..9179] | 9 172 [9162..9175] | 8 996 [8981..9003] | 8 511 [8501..8521] | 7 414 [7398..7415] |
| llama.cpp ub1024-b4096-faon | 8 666 [8646..8669] | 8 839 [8834..8849] | 8 723 [8721..8726] | 8 282 [8278..8296] | 7 402 [7392..7410] |
| vLLM défaut | 10 806 (méd. 10 813) | 10 782 (méd. 10 784) | 10 510 (méd. 10 510) | 9 796 (méd. 9 797) | 8 572 (méd. 8 567) |
| vLLM FLASHINFER | 10 805 (méd. 10 808) | 10 796 (méd. 10 797) | 10 626 (méd. 10 619) | 9 903 (méd. 9 899) | 8 713 (méd. 8 712) |
| TensorRT-LLM un bloc, mnt 40960 | — | — | — | — | — |
| TensorRT-LLM un bloc, mnt N | 11 207 (méd. 11 203) | 10 986 (méd. 10 987) | 10 626 (méd. 10 626) | 9 866 (méd. 9 866) | — |
| TensorRT-LLM blocs de 8192 | 11 178 (méd. 11 179) | 10 937 (méd. 10 943) | 10 526 (méd. 10 527) | 9 671 (méd. 9 680) | 8 298 (méd. 8 298) |
| SGLang | 11 136 [11131..11150] | 11 177 [11175..11179] | 10 862 [10857..10865] | 10 158 [10156..10159] | — |

« — » : ne passe pas sur les 16 Go (voir Méthode).

## Faisabilité (`runs/feas-long/`, `runs/feas-long2/`, `runs/feas-long3/`)

À 40 000 tokens, une itération :
- herbert-cuda 4B : 8 574 tok/s ;
- llama.cpp 4B : 7 344 en réglage par défaut, 7 384 avec `ub1024-b4096-fa` ;
- vLLM 4B : 8 587 avec l'attention par défaut ;
- TensorRT-LLM : un bloc 40960 manque de mémoire sur le 4B ; avec une fraction KV de 0.2, il démarre mais refuse la requête (« User-provided base page indices is too short », pool KV trop court pour 40 000) ; blocs de 8 192 sur le 4B : 8 318 ; 0.8B : 35 108 en un bloc, 34 199 en blocs ; 2B : 21 564 en un bloc ;
- SGLang : 0.8B 38 121 (mf 0.7, mrr 1) et 38 123 (mf 0.6) ; 2B 22 616 et 22 636 ; 4B rien (voir Méthode). Avec mf 0.8 et `--max-mamba-cache-size 2`, aucun modèle ne démarre (« state cache is too small »).

## Fichiers

- `runs/campaign-long/` : toutes les sorties, `run.out` et `run-trt-mntN.out`, `env.txt`, `clocks*.csv`, `summary.md` et `summary.json` ;
- `runs/feas-long*/` : la faisabilité ;
- `runs/prompts-long/` : les cinq prompts et le journal de construction ;
- `_scripts/` : identiques à ceux lancés.

Les valeurs seulement.
