# Phase 1 de la campagne de #003 : le meilleur réglage de chaque comparateur sur la 4080 (2026-10-18)

But (`protocol.md`, écrit avant toute mesure, après les prédictions, `predictions.md` ; ces deux fichiers sont à la racine du dépôt) : trouver pour chaque comparateur le réglage qui lui donne son meilleur prefill et son meilleur décodage, une requête à la fois, le défaut mesuré à côté. Aucune mesure de herbert-cuda dans cette phase.

**Machine** : RTX 4080 (Ryzen 9 9900X, Linux), pilote 595.99.02, SM verrouillé à 2 805 MHz et mémoire à 11 201 MHz. Tous les moteurs CUDA tournent en P0 grâce au profil `CudaNoStablePerfLimit`. D'après les relevés à la seconde (`*/clocks.csv`), la mémoire est restée à 11 201 MHz et la carte en P0 sur tous les échantillons. Le SM est à 2 805 MHz sur l'immense majorité ; quelques dizaines d'échantillons de la phase 1 descendent jusqu'à 2 730. La charge CPU est relevée à chaque étape (`*.out`), car d'autres travaux tournent sur la machine.

**Versions** (`phase1/env.txt`) :
- llama.cpp release `b11718` (`547208dca`), CUDA et Vulkan ; référence de #002 `08ffba21e` (CUDA) ;
- vLLM 0.31.0 (torch 2.13.0, flashinfer 0.7.0.post1) ; référence de #002 0.28.0 ;
- SGLang 0.5.22 (torch 2.14.1, flashinfer 0.7.1) ;
- transformers 5.20.1 (torch 2.14.1, flash-linear-attention 0.5.2, sans `causal-conv1d`).

**Définitions** :
- prefill = longueur du prompt / temps jusqu'au premier token ;
- décodage = 255 tokens à la profondeur 759 ;
- llama.cpp : `llama-bench` (`-p … -n 0`, `-p 0 -n 255 -d 759`, `-r 1`), un processus par réglage, 3 tours alternés ;
- vLLM : `vllm bench latency` (2 chauffes, 5 mesures), prefill = len / L1 et décodage = 254 / (L255 − L1) ;
- SGLang : `sglang.benchmark.one_batch`, chaque forme trois fois dans le processus ;
- transformers : `generate` glouton chronométré, 1 chauffe et 5 mesures, mêmes définitions que vLLM.

## Les passages, dans l'ordre (tous gardés)

- `phase1/` (`phase1.out`, `_scripts/_p1*.sh`) : le balayage complet. Trois défauts des scripts, pas des moteurs. vLLM tournait sans `--max-num-seqs`, et le 4B est refusé dès qu'il capture des graphes (« max_num_seqs (256) exceeds available Mamba cache blocks (127) ») ; FLASHINFER compile avec `/usr/bin/nvcc` 12.0 et échoue. SGLang n'a qu'une mesure par forme. Pour transformers 4B, le modèle d'un réglage n'est pas libéré avant le suivant, d'où le manque de mémoire. Les résultats vLLM et SGLang de ce passage sont remplacés par les suivants (`vllm-phase1` et `sglang-phase1` dans `summary.txt`).
- `after-p1/vllm/` : tout vLLM refait, avec `--max-num-seqs 8` comme #002 et `CUDA_HOME` sur le toolkit 12.8. 96 mesures, aucun échec.
- `after-p1/sglang-rep/` : SGLang refait, trois mesures par forme. `after-p1/transformers-4B.*` : les quatre réglages 4B perdus, un processus chacun.
- `after-p1c/` : vLLM avec la taille de capture des graphes CUDA donnée explicitement (`-cc.max_cudagraph_capture_size=512` et `4096`), plus le 0.8B sans `--max-num-seqs` dans le même passage (`mns256`).
- `after-p1d/` : tokens contre l'oracle. transformers et llama.cpp passent ; vLLM et SGLang échouent, parce que `_p1_tokens.py` n'avait pas de garde `__main__` (leurs moteurs lancent des sous-processus).
- `after-p1e/` : `--enable-torch-compile` de SGLang une dernière fois, graphes bornés au lot 8, fraction statique à 0,6 et 0,5. Échec.
- `after-p1f/` : tokens de vLLM et SGLang, script corrigé.

`summary.txt` est la sortie de `_p1_summary.py phase1 after-p1 after-p1c after-p1e`. Elle donne, par moteur, modèle et forme, la médiane de chaque réglage, son écart et le meilleur réglage face au défaut, puis le tableau ci-dessous.

## Meilleur réglage de chaque moteur (médianes, tok/s ; défaut en dessous)

| moteur | modèle | pp128 | pp512 | pp2048 | pp4000 | décodage |
|---|---|---|---|---|---|---|
| llama.cpp CUDA b11718 | 0.8B | 15 841 (ub256 fa on) | 34 532 (ub512 fa on) | 38 987 (ub1024 b4096 fa on) | 41 185 (ub1024 fa on) | 325,4 (défaut) |
| | défaut | 15 619 | 34 368 | 38 257 | 38 510 | 325,4 |
| | 4B | 5 246 (ub1024 b4096 fa on) | 8 915 (ub2048 fa on) | 9 387 (ub512 fa on) | 9 244 (ub512 fa on) | 74,4 (fa on) |
| | défaut | 5 202 | 8 914 | 9 384 | 9 240 | 74,4 |
| llama.cpp Vulkan b11718 | 0.8B | 7 172 (ub512 b4096 fa off) | 11 903 (ub2048 b4096 fa on) | 14 186 (ub2048 fa on) | 13 938 (ub2048 b4096 fa on) | 306,2 (fa on) |
| | défaut | 7 127 | 11 859 | 12 021 | 11 678 | 305,9 |
| | 4B | 1 931 (ub512 b4096 fa off) | 2 626 (ub256 b4096 fa on) | 2 819 (ub2048 fa on) | 2 762 (ub2048 b4096 fa on) | 70,3 (défaut) |
| | défaut | 1 927 | 2 492 | 2 490 | 2 430 | 70,3 |
| llama.cpp CUDA 08ffba21e | 0.8B défaut | 14 701 | 29 747 | 32 602 | 32 851 | 322,4 |
| | 4B défaut | 5 019 | 8 319 | 8 715 | 8 597 | 74,1 |
| vLLM 0.31.0 | 0.8B | 10 938 (mns256*) | 42 236 (mns256*) | 57 043 (mns8 cg512*) | 54 523 (mns8 mnbt2048) | 376,1 (mns8 triton) |
| | défaut mns8 | 8 934 | 35 748 | 54 359 | 52 988 | 366,2 |
| | 4B | 5 979 (mns8 cg4096*) | 9 888 (mns8 flashinfer cg512*) | 10 987 (mns8 flashinfer) | 10 897 (mns8 flashinfer cg512*) | 76,6 (mns8 triton cg512*) |
| | défaut mns8 | 5 881 | 9 771 | 10 943 | 10 845 | 76,1 |
| vLLM 0.28.0, mns8 | 0.8B défaut | 9 526 | 35 665 | 54 102 | 52 926 | 356,7 |
| | 4B défaut | 5 815 | 9 795 | 10 939 | 10 789 | 75,8 |
| SGLang 0.5.22 | 0.8B | 14 455 (triton) | 47 249 (triton) | 64 251 (nochunk) | 58 418 (nochunk) | 337,0 (triton) |
| | défaut | 14 269 | 47 004 | 63 796 | 58 401 | 330,7 |
| | 4B | 6 153 (chunk2048) | 10 678 (défaut) | 11 416 (nochunk) | 11 343 (flashinfer) | 75,1 (triton) |
| | défaut | 6 081 | 10 678 | 11 368 | 11 327 | 74,9 |
| transformers 5.20.1 | 0.8B | 8 495 (sdpa static compile) | 30 822 (sdpa dynamic) | 44 416 (sdpa static compile) | 40 943 (sdpa static compile) | 270,3 (eager static) |
| | défaut sdpa dynamic | 7 636 | 30 822 | 43 291 | 39 653 | 102,4 |
| | 4B | 5 683 (sdpa static compile) | 9 002 (sdpa static compile) | 9 395 (sdpa dynamic) | 8 816 (sdpa static compile) | 68,0 (eager static compile) |
| | défaut sdpa dynamic | 5 064 | 8 878 | 9 395 | 8 448 | 60,5 |

\* passage `after-p1c`. « mns8 » = `--max-num-seqs 8` ; mns256 = sans `--max-num-seqs` (le vrai défaut de vLLM, refusé par le 4B). Les écarts (max − min) / médiane de chaque cellule sont dans `summary.txt`.

## Ce qui ressort des mesures (faits, sans verdict)

- **vLLM et la capture des graphes.** `--max-num-seqs 8` borne la capture des graphes CUDA à 16 tokens (`cudagraph_capture_sizes [1, 2, 4, 8, 16]` dans les journaux). Sans cette option, le 0.8B capture jusqu'à 512. Prefill 0.8B avec mns8 : 8 974 tok/s à 128 tokens et 35 182 à 512. Avec `-cc.max_cudagraph_capture_size=512` : 10 811 et 42 228. Sans `--max-num-seqs` : 10 938 et 42 236. Pour le 4B, la taille de capture change le prefill de 1 % au plus.
- **Versions de llama.cpp** (défauts, CUDA) : b11718 contre `08ffba21e`, en 0.8B, ×1,06 à 128 tokens, ×1,16 à 512, ×1,17 à 2 048 et 4 000. En 4B : ×1,04 à ×1,08. Décodage : 325,4 contre 322,4 (0.8B), 74,4 contre 74,1 (4B).
- **Versions de vLLM** (mns8, défauts) : 0.31.0 contre 0.28.0, prefill à ±7 % (0.8B à 128 tokens : 8 934 contre 9 526), décodage 366,2 contre 356,7 (0.8B) et 76,1 contre 75,8 (4B).
- **Réglages qui bougent le plus** :
  - llama.cpp Vulkan : `-ub 2048`, ×1,18 à ×1,19 au prefill long (0.8B), ×1,13 à ×1,14 (4B) ;
  - llama.cpp CUDA 0.8B à 4 000 tokens : `-ub 1024`, ×1,07 ;
  - transformers, au décodage : le cache statique, ×2,64 (0.8B, eager static) ; static compile, ×1,12 (4B).
  - Ailleurs, le meilleur réglage est à quelques pour cent du défaut.
- **llama.cpp Vulkan sur la 4080** (`Vulkan1`, NV_coopmat2) : prefill 2 à 3,5 fois sous son backend CUDA. Décodage à 94 % (0.8B) et 95 % (4B) du CUDA.
- **Dispersions à connaître** :
  - transformers compilé, 45 à 100 % : la première mesure de chaque forme recompile (par ex. 0,516 s puis 0,015 s à 128 tokens), la médiane ne la voit pas ;
  - SGLang 0.8B à 4 000 tokens, 17 % : 52 771 puis 58 454 et 58 400 ;
  - vLLM 0.8B cg512 à 2 048 tokens, 10,5 %.
- **Refus des moteurs** : `--enable-torch-compile` de SGLang échoue sur les deux modèles, en quatre tentatives (phase 1, reprise, `after-p1e` à 0,6 et 0,5). Les noyaux Triton d'inductor ne compilent pas (« 'NoneType' object has no attribute '_unflatten_ir' », Triton 3.8.0), puis la capture des graphes manque de mémoire (2,6 à 3,8 Gio demandés).

## Justesse : premier token différent de l'oracle CPU (`tokens-diff.txt`)

Mêmes ids de prompt que l'oracle (`herbert-cuda-bench --prepare`, longueurs 24, 39, 759 et 2 882 tokens, celles des journaux de l'oracle), génération gloutonne, comparaison coupée au premier token de fin de tour (transformers, sans `generation_config.json`, continue après `<|im_end|>`). Oracle : journaux CPU de herbert-jit-bf16 v0.1.0 (`after-p1d/oracle-*`). « = » : identique sur toute la longueur de l'oracle. Pour p3, l'oracle compte 255 tokens et les moteurs 256 : l'écart de longueur est ignoré. Pour situer, herbert-cuda (le candidat, mesuré sur une seconde RTX 4080) : 0.8B = 8 = 8, 4B = 32 = 105.

| moteur, réglage | 0.8B p1 | p2 | p3 | p4 | 4B p1 | p2 | p3 | p4 |
|---|---|---|---|---|---|---|---|---|
| transformers sdpa dynamic | = | 8 | 20 | 8 | = | 43 | 151 | 131 |
| transformers eager dynamic | = | 8 | 172 | 61 | = | 32 | = | 105 |
| transformers sdpa static | = | 15 | 111 | 8 | = | 32 | 151 | 106 |
| llama.cpp CUDA défaut | = | 8 | 172 | 8 | = | = | 162 | 155 |
| llama.cpp CUDA fa on ub1024 | = | 8 | 172 | 8 | = | = | 162 | 105 |
| llama.cpp CUDA fa off | = | 8 | 172 | 8 | = | = | 162 | 155 |
| llama.cpp Vulkan défaut | = | 8 | 172 | 8 | = | = | 151 | 105 |
| llama.cpp Vulkan fa on ub2048 | = | 8 | 172 | 8 | = | = | 162 | 105 |
| llama.cpp Vulkan fa off | = | 8 | 172 | 8 | = | = | 162 | 105 |
| vLLM défaut (mns8) | = | 8 | 111 | 8 | = | 23 | 94 | 82 |
| vLLM triton | = | 15 | 111 | 8 | = | 32 | 85 | 45 |
| vLLM flashinfer | = | 8 | 233 | 8 | = | 32 | = | 2 |
| vLLM eager | = | 15 | 233 | 8 | = | 32 | 85 | 155 |
| SGLang défaut | = | 8 | 111 | 8 | = | 118 | 99 | 105 |
| SGLang triton | = | 8 | 233 | 8 | = | 32 | 99 | = |
| SGLang flashinfer | = | 8 | 111 | 8 | = | 118 | 99 | 105 |
| SGLang chunk2048 | = | 8 | 111 | 8 | = | 118 | 99 | 105 |

Les logits ne sont pas relevés pour les comparateurs : on ne sait pas si ces divergences tombent sur des égalités, comme celles de herbert-cuda. vLLM flashinfer s'écarte dès le token 2 sur le 4B p4 ; c'est noté sans analyse.

## Fichiers

- `protocol.md`, `predictions.md` : à la racine du dépôt (les copies de ce dossier n'y sont pas).
- `_scripts/` : tous les scripts lancés sur la RTX 4080 (sans `_f16-rtx4080-b-003.sh`, qui a tourné sur une autre machine).
- `_p1_summary.py`, `_p1_tokens_diff.py` : les résumés.
- `*.out` : la sortie de chaque file d'attente, avec ses marqueurs `ALLDONE-*` et ses lignes `FAIL`.
