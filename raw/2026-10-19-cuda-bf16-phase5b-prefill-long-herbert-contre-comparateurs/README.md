# Phase 5 de #003, étape 5b : le prefill long (10 000 à 60 000 tokens) sur la 4080, herbert contre les comparateurs, banc de chaque moteur et client HTTP commun (2026-10-19)

Étape 5b du protocole de la phase 5 (`protocol.md` de `herbert-cuda-bench` et son amendement de 16 h 10, écrits avant toute mesure), après les prédictions : P14 « herbert se fait déboîter au prefill long », P15 « le décode aussi ». Les deux instruments de la phase 2, sans changement de méthode, aux longueurs 10K, 20K, 40K et 60K, avec les deux réglages par comparateur retenus en 5a.

**Machine** : RTX 4080 (Ryzen 9 9900X, Linux), pilote 595.99.02, SM 2 805 et mémoire 11 201 MHz verrouillés (vérifiés par le script au départ), profil `CudaNoStablePerfLimit` (P0 pour tous les moteurs CUDA). De 20 h 24 à 23 h 53.

**Horloges** (`phase5b/clocks.csv`, à la seconde) : mémoire à 11 201 MHz et P0 sur tous les échantillons ; les relevés `http/pstate-*.txt` pris au début et à la fin de chaque serveur sont tous en P0. Sur les 9 616 échantillons au-dessus de 100 W, **le SM est sous 2 805 MHz sur 5 620 (58 %)**, jusqu'à 2 205 MHz. La raison active est la limite logicielle de puissance (`0x4`) pour 5 274 d'entre eux et aucune pour 344. Comme en 5a, le prefill long atteint la limite de puissance de la carte à horloges verrouillées, pour tous les moteurs.

**Versions** (`phase5b/env-start.txt`, `env-mid-a.txt`, `env-end.txt`) :
- herbert-cuda-bf16 `011bfdf` (le candidat par défaut depuis `638364b`) ;
- herbert-cuda-f16 `0c7d88d` au défaut (hc-f16) et avec `--tune gemm.f16=1` (hc-f16-acc16) ;
- herbert-vk-f16 `cc51363` au défaut (hvf) et avec `--tune host.poll=1` (hvf-poll), à 10K et 20K au banc (`--ask`, sans `--ask-file` à ce commit ; le prompt tient dans un argument de 128 Kio sous Linux jusqu'à 20K), aux quatre longueurs au client HTTP ;
- llama.cpp `b11718` CUDA et Vulkan, GGUF F16 ;
- vLLM 0.31.0, SGLang 0.5.22 (`SGLANG_MAMBA_CONV_DTYPE=float16`), transformers 5.20.1, en float16.

Contexte de 61 440 positions partout.

## Déroulé (`phase5b.out`, `_scripts/_p5b-003.sh`, `herbert-cuda-bench` `5db1f2b`)

**A. Banc de chaque moteur** :
- herbert (les cinq unités), llama.cpp CUDA et llama.cpp Vulkan, alternés tour par tour : N = 5, ordre inversé un tour sur deux.
  - herbert : `--ask-file ppN.txt --no-system --no-think --max-tokens 128 --warmup 1`. Prefill et decode (les 127 tokens après le premier) sur la ligne `[turn 1]`.
  - llama.cpp : prefill aux deux réglages de 5a. CUDA `-ub 1024 -b 4096 -fa on` et `-ub 1024 -b 2048 -fa on` ; Vulkan `-ub 1024 -b 4096 -fa on` et `-ub 512 -b 4096 -fa on`. Decode `-p 0 -n 128 -d N`, au défaut et avec `-fa on`.
- vLLM : `bench latency` avec `flashinfer` et `--max-num-batched-tokens 2048`, sorties 1 et 128, 5 mesures. Prefill N / L(1), decode 127 / (L(128) − médiane L(1)).
- SGLang : `one_batch`, `mf06` (fraction 0,6) et `mf07-mrr1` (0,7, `--max-running-requests 1`), un processus par longueur. Prefill sur les 3 passes à sortie 1. Decode par la médiane du processus à sortie 128, la première passe écartée.
- transformers : `sdpa-dynamic` et `sdpa-static-compile`, 1 et 128 tokens, même définition que vLLM.

**B. Client HTTP commun** (`_scripts/_http_client.py --skip-blank`, `prompts-long.json` de 5a : cinq passages par longueur, trois tours, 128 tokens) :
- un serveur après l'autre, sans cache de préfixe, contexte 61 440 ;
- llama-server `-fa on -ub 1024 -b 4096`, et `-ub 512` pour Vulkan 4B ;
- vLLM et SGLang à chacun de leurs deux réglages, transformers serve avec et sans `--compile`.

`summary.txt` est la sortie de `_p5b_summary.py phase5b`. Il donne toutes les cellules (réglage, n, écart), tous les réglages des comparateurs et tous les rapports, pour chacune des trois formes de herbert-cuda.

## Ce qui n'a pas tourné (`phase5b.out`, journaux complets dans `phase5b/`)

- **SGLang 4B `mf06`** : échec à toutes les longueurs, comme en 5a. Au banc, `RuntimeError: Hybrid (mamba/linear-attention) state cache is too small to serve any requests` (`max_mamba_cache_size=3`). Au client HTTP, le serveur répond à la sonde de santé, puis aucune requête n'aboutit : le client s'arrête sans valeur (`TypeError` sur une médiane vide).
- **SGLang 4B `mf07-mrr1`** au banc : 40K et 60K sont sautés (« skipping … max batch size limit »), comme en 5a. Au client HTTP, le même réglage sert les quatre longueurs.
- **SGLang 0.8B `mf07-mrr1`** : manque de mémoire à 60K au banc (`torch.OutOfMemoryError`).
- **transformers `sdpa-static-compile`** : `Segmentation fault (core dumped)` au premier cas de decode (10K, 128 tokens), en 0.8B comme en 4B, après le cas de prefill à 10K. Ce réglage n'a donc de valeur qu'au prefill à 10K. Ses écarts y sont de 52 % (0.8B) et 26 % (4B), contre 0,0 à 0,1 % pour `sdpa-dynamic`.
- **transformers 4B `sdpa-dynamic`** : manque de mémoire à 60K (`OutOfMemoryError`), comme en 5a.
- **transformers serve 4B** (avec ou sans `--compile`) : 10K et 20K servis, puis manque de mémoire à 40K. Le client s'arrête avant d'écrire son JSON. Les valeurs de 10K et 20K ne sont que dans `http/client-4B-transformers-*.log` (sdpa : 7 830 et 7 347 tok/s de prefill, 55,8 et 50,7 de decode), hors des tableaux.

## A. Banc de chaque moteur, médianes, tok/s

Le meilleur des réglages de chaque comparateur, cellule par cellule (le réglage est dans `summary.txt`). Les écarts sont au plus de 1,7 % pour les cinq unités de herbert, et au plus de 1 % pour les comparateurs, hors transformers `sdpa-static-compile`.

Prefill :

| | 10K | 20K | 40K | 60K |
|---|---|---|---|---|
| 0.8B hc-bf16 | 53 141 | 44 759 | 33 248 | 26 646 |
| 0.8B hc-f16 | 52 903 | 44 587 | 33 131 | 26 578 |
| 0.8B hc-f16-acc16 | 59 635 | 48 783 | 35 469 | 28 059 |
| 0.8B hvf (poll) | 49 156 (49 252) | 41 665 (41 523) | | |
| 0.8B llama.cpp CUDA | 47 752 | 42 898 | 35 232 | 29 757 |
| 0.8B llama.cpp Vulkan | 27 841 | 25 413 | 21 410 | 18 413 |
| 0.8B vLLM | 51 952 | 45 827 | 37 025 | 30 738 |
| 0.8B SGLang | 54 333 | 47 043 | 37 875 | 31 767 |
| 0.8B transformers | 41 659 | 29 493 | 26 012 | 22 820 |
| 4B hc-bf16 | 10 599 | 9 551 | 8 025 | 6 904 |
| 4B hc-f16 | 10 547 | 9 501 | 7 986 | 6 874 |
| 4B hc-f16-acc16 | 12 635 | 11 163 | 9 097 | 7 643 |
| 4B hvf (poll) | 9 698 (9 702) | 8 891 (8 891) | | |
| 4B llama.cpp CUDA | 11 402 | 10 630 | 9 174 | 8 041 |
| 4B llama.cpp Vulkan | 8 634 | 8 061 | 7 127 | 6 344 |
| 4B vLLM | 10 673 | 9 967 | 8 743 | 7 787 |
| 4B SGLang | 10 895 | 10 159 | | |
| 4B transformers | 9 226 | 7 454 | 6 825 | |

Decode de 128 tokens à la profondeur N :

| | 10K | 20K | 40K | 60K |
|---|---|---|---|---|
| 0.8B hc-bf16 | 298,6 | 240,2 | 171,7 | 133,4 |
| 0.8B hc-f16 | 299,1 | 240,6 | 171,9 | 133,5 |
| 0.8B hc-f16-acc16 | 298,9 | 240,5 | 172,0 | 133,5 |
| 0.8B hvf (poll) | 259,2 (283,6) | 209,1 (226,0) | | |
| 0.8B llama.cpp CUDA | 307,4 | 291,1 | 262,6 | 239,4 |
| 0.8B llama.cpp Vulkan | 288,2 | 270,9 | 243,8 | 220,5 |
| 0.8B vLLM | 343,7 | 328,9 | 296,4 | 270,4 |
| 0.8B SGLang | 303,9 | 289,9 | 263,1 | 240,3 |
| 0.8B transformers | 101,3 | 100,4 | 98,4 | 96,6 |
| 4B hc-bf16 | 72,6 | 67,1 | 57,8 | 50,9 |
| 4B hc-f16 | 72,6 | 67,2 | 57,8 | 50,9 |
| 4B hc-f16-acc16 | 72,6 | 67,2 | 57,8 | 50,9 |
| 4B hvf (poll) | 68,1 (69,8) | 62,4 (64,1) | | |
| 4B llama.cpp CUDA | 72,0 | 69,6 | 65,2 | 61,4 |
| 4B llama.cpp Vulkan | 68,0 | 65,5 | 61,3 | 58,0 |
| 4B vLLM | 73,4 | 71,7 | 67,4 | 63,4 |
| 4B SGLang | 71,7 | 70,2 | | |
| 4B transformers | 56,7 | 51,4 | 44,4 | |

### Rapports hc-bf16 / moteur (banc de chaque moteur)

| | 10K | 20K | 40K | 60K |
|---|---|---|---|---|
| 0.8B prefill / SGLang | ×0,978 | ×0,951 | ×0,878 | ×0,839 |
| 0.8B prefill / vLLM | ×1,023 | ×0,977 | ×0,898 | ×0,867 |
| 0.8B prefill / llama.cpp CUDA | ×1,113 | ×1,043 | ×0,944 | ×0,895 |
| 0.8B decode / vLLM | ×0,869 | ×0,731 | ×0,579 | ×0,494 |
| 0.8B decode / llama.cpp CUDA | ×0,971 | ×0,825 | ×0,654 | ×0,557 |
| 0.8B decode / SGLang | ×0,982 | ×0,829 | ×0,653 | ×0,555 |
| 4B prefill / llama.cpp CUDA | ×0,930 | ×0,898 | ×0,875 | ×0,859 |
| 4B prefill / vLLM | ×0,993 | ×0,958 | ×0,918 | ×0,887 |
| 4B prefill / SGLang | ×0,973 | ×0,940 | | |
| 4B decode / vLLM | ×0,989 | ×0,937 | ×0,858 | ×0,803 |
| 4B decode / llama.cpp CUDA | ×1,008 | ×0,965 | ×0,887 | ×0,830 |

Le même tableau pour hc-f16-acc16 :

| | 10K | 20K | 40K | 60K |
|---|---|---|---|---|
| 0.8B prefill / SGLang | ×1,098 | ×1,037 | ×0,936 | ×0,883 |
| 0.8B prefill / vLLM | ×1,148 | ×1,064 | ×0,958 | ×0,913 |
| 0.8B prefill / llama.cpp CUDA | ×1,249 | ×1,137 | ×1,007 | ×0,943 |
| 4B prefill / llama.cpp CUDA | ×1,108 | ×1,050 | ×0,992 | ×0,950 |
| 4B prefill / vLLM | ×1,184 | ×1,120 | ×1,041 | ×0,981 |

Au decode, hc-f16-acc16 rend les mêmes valeurs que hc-bf16 (×0,999 à ×1,001).

**Bf16 contre f16** :
- hc-f16 / hc-bf16 : ×0,995 à ×0,997 au prefill, ×1,000 à ×1,002 au decode.
- hc-f16-acc16 / hc-bf16 au prefill, de 10K à 60K : 0.8B ×1,122 → ×1,053 ; 4B ×1,192 → ×1,107.

**Perte de 10K à 60K** :
- prefill 0.8B : hc-bf16 −50 %, SGLang −42 %, vLLM −41 %, llama.cpp CUDA −38 % ;
- prefill 4B : hc-bf16 −35 %, llama.cpp CUDA −29 %, vLLM −27 % ;
- decode 0.8B : hc-bf16 −55 %, vLLM −21 %, SGLang −21 %, llama.cpp CUDA −22 % ;
- decode 4B : hc-bf16 −30 %, vLLM −14 %, llama.cpp CUDA −15 %.

**herbert-vk-f16** (10K et 20K) :
- prefill : hc-bf16 / hvf ×1,081 et ×1,074 (0.8B), ×1,093 et ×1,074 (4B) ;
- decode : hc-bf16 / hvf-poll ×1,053 et ×1,063 (0.8B), ×1,040 et ×1,048 (4B).

**Tokens** : chaque unité de herbert rend la même suite de 127 ids à ses 5 tours, à chaque longueur. Les tokens entre moteurs sont dans le dossier de l'étape 5a.

## B. Client HTTP commun, médianes (15 requêtes par cellule)

Prefill = tokens d'entrée / TTFT ; decode en tok/s. Longueurs d'entrée exactes dans `summary.txt` (10 009 à 60 013 selon le gabarit de chaque serveur).

| | 10K | 20K | 40K | 60K |
|---|---|---|---|---|
| 0.8B hc-bf16 | 50 599 · 298,5 | 42 827 · 240,2 | 32 072 · 171,8 | 25 873 · 133,4 |
| 0.8B hc-f16-acc16 | 56 095 · 298,8 | 46 464 · 240,4 | 34 060 · 171,9 | 27 148 · 133,5 |
| 0.8B hvf | 45 213 · 262,9 | 39 069 · 210,4 | 30 516 · 151,2 | 24 706 · 119,0 |
| 0.8B llama.cpp CUDA | 34 733 · 294,0 | 33 568 · 278,5 | 29 498 · 251,9 | 25 698 · 230,4 |
| 0.8B vLLM flashinfer | 48 890 · 343,1 | 43 773 · 325,8 | 34 982 · 295,3 | 28 614 · 269,7 |
| 0.8B SGLang mf06 | 55 182 · 347,9 | 47 700 · 329,3 | 37 147 · 293,4 | 30 076 · 265,0 |
| 0.8B transformers sdpa | 32 649 · 96,3 | 28 097 · 95,5 | 24 897 · 94,2 | 21 975 · 91,8 |
| 4B hc-bf16 | 10 464 · 72,6 | 9 426 · 67,1 | 7 936 · 57,8 | 6 812 · 50,9 |
| 4B hc-f16-acc16 | 12 429 · 72,6 | 10 937 · 67,1 | 8 958 · 57,8 | 7 519 · 50,9 |
| 4B hvf | 9 486 · 68,2 | 8 702 · 62,3 | 7 382 · 53,7 | 6 458 · 47,6 |
| 4B llama.cpp CUDA | 9 409 · 71,2 | 9 207 · 68,8 | 8 188 · 64,5 | 7 421 · 60,7 |
| 4B vLLM flashinfer | 10 403 · 73,5 | 9 686 · 71,8 | 8 529 · 67,4 | 7 569 · 63,5 |
| 4B SGLang mf07-mrr1 | 10 515 · 73,9 | 9 864 · 72,3 | 8 665 · 67,6 | 7 685 · 63,5 |

Rapports hc-bf16 / serveur (prefill · decode) :

| | 10K | 20K | 40K | 60K |
|---|---|---|---|---|
| 0.8B / SGLang mf06 | ×0,917 · ×0,858 | ×0,898 · ×0,730 | ×0,863 · ×0,585 | ×0,860 · ×0,504 |
| 0.8B / vLLM flashinfer | ×1,035 · ×0,870 | ×0,978 · ×0,737 | ×0,917 · ×0,582 | ×0,904 · ×0,495 |
| 0.8B / llama.cpp CUDA | ×1,457 · ×1,015 | ×1,276 · ×0,863 | ×1,087 · ×0,682 | ×1,007 · ×0,579 |
| 4B / SGLang mf07-mrr1 | ×0,995 · ×0,982 | ×0,956 · ×0,928 | ×0,916 · ×0,856 | ×0,886 · ×0,802 |
| 4B / vLLM flashinfer | ×1,006 · ×0,988 | ×0,973 · ×0,935 | ×0,930 · ×0,858 | ×0,900 · ×0,802 |
| 4B / llama.cpp CUDA | ×1,112 · ×1,020 | ×1,024 · ×0,975 | ×0,969 · ×0,896 | ×0,918 · ×0,838 |

Tous les serveurs, tous les rapports (hc-f16, llama.cpp Vulkan, le second réglage de vLLM et SGLang, transformers avec `--compile`, et ceux de hc-f16-acc16) sont dans `summary.txt`.

## Fichiers

- `phase5b.out` : la sortie complète du script.
- `phase5b/chat/` : journaux de herbert ; `phase5b/llama/` : JSON et stderr de chaque `llama-bench` ; `phase5b/vllm/`, `phase5b/sglang/`, `phase5b/transformers/` : sorties complètes des bancs ; `phase5b/http/` : sortie du client, JSON par serveur, journal de chaque serveur, P-state de début et de fin.
- `phase5b/clocks.csv`, `phase5b/env-*.txt`, `phase5b/end.txt`.
- `_scripts/` : `_p5b-003.sh`, `_p5_transformers.py`, `_http_client.py`, `_p5b_summary.py`.
