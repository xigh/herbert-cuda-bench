# 2026-10-21 · herbert-cuda-bf16 face aux comparateurs de la phase 5b, dans le même passage et en alterné

Objet, après les interrupteurs d'attention : une mesure alternée face aux comparateurs. RTX 4080 verrouillée à 2 805 / 11 201 MHz, P0 (profil `CudaNoStablePerfLimit`), le 2026-10-21 de 01 h 19 à 03 h 24. Le service d'inférence résident (`resident-gpu`) a été arrêté pendant toute la mesure, puis relancé par un piège à la fin ; il était actif à 03 h 24. Le GPU était vide à chaque lancement : `waits.txt` n'existe pas, aucune attente n'a eu lieu.

## Ce qui a tourné

`_scripts/_av1-003.sh` : 5 rounds ; dans chaque round, le 0.8B puis le 4B, et pour chaque modèle les cinq unités, l'ordre étant inversé un round sur deux. Longueurs : 10K, 20K, 40K et 60K tokens. Les réglages des comparateurs sont, pour chacun, le meilleur de la phase 5b, sans changement de méthode.

- **hc-def** : herbert-cuda-bf16 `73b5a70`, défaut (sha256 du binaire dans `av1/env-start.txt`). Chat, `--ask-file`, `--ctx 61440`, sans réflexion, 128 tokens générés, `--warmup 1`.
- **hc-sw** : le même binaire avec `--tune sdpa.gqa=1 --tune sdpa.split=64 --tune sdpa.pipe=1`.
- **llama-cuda** : llama.cpp b11718, `Qwen3.5-*-F16.gguf`. Prefill `-ub 1024 -b 4096 -fa on` ; decode `-p 0 -n 128 -d N`, par défaut ; `-r 1` par round.
- **vllm** : vLLM 0.31.0, `--dtype float16 --attention-backend FLASHINFER`, `bench latency` avec 1 et 128 tokens de sortie, 2 tours de chauffe et 5 itérations par processus, un processus par longueur et par sortie.
- **sglang** : SGLang 0.5.22, `--dtype float16`, `one_batch`. Le 0.8B en `mf06` à toutes les longueurs ; le 4B en `mf07-mrr1` à 10K et 20K seulement : à 40K et 60K il est sauté, et `mf06` échoue à toutes les longueurs.

## Définitions

Celles de `_p5b_summary.py`, reprises dans `_scripts/_av1_summary.py` :

- herbert : la ligne `[turn]` du chat, prefill, puis decode des tokens générés après le premier ;
- llama.cpp : `avg_ts` de `llama-bench` ;
- vLLM :
  - prefill = N / L(sortie 1) ;
  - decode = 127 / (L(sortie 128) − médiane de L(sortie 1)) ;
  - les deux calculés ici round par round ;
- SGLang :
  - prefill : `prefill_throughput` des passes à 1 token de sortie ;
  - decode : débit médian des passes à 128 tokens, sans la première de chaque processus.

Pour chaque unité et chaque forme : la médiane sur les rounds de la médiane de chaque round (n = 5, écart max − min en % de la médiane). Chaque rapport est donné de deux façons : le rapport des médianes, puis la médiane des rapports round par round, avec le minimum et le maximum de ces derniers.

## Valeurs (tok/s, `summary.txt`)

| 0.8B | pp10000 | pp20000 | pp40000 | pp60000 | tg128@10000 | tg128@20000 | tg128@40000 | tg128@60000 |
|---|---|---|---|---|---|---|---|---|
| hc-def | 52 863 | 44 577 | 33 079 | 26 531 | 298,6 | 240,2 | 171,8 | 133,4 |
| hc-sw | 55 609 | 48 465 | 37 620 | 31 041 | 369,5 | 347,8 | 308,9 | 278,2 |
| llama-cuda | 47 478 | 42 630 | 35 019 | 29 582 | 307,4 | 291,1 | 262,7 | 239,4 |
| vllm | 50 660 | 45 635 | 36 792 | 30 539 | 344,3 | 328,5 | 297,2 | 271,1 |
| sglang | 54 049 | 46 876 | 37 710 | 31 535 | 303,2 | 289,6 | 262,4 | 239,6 |

| 4B | pp10000 | pp20000 | pp40000 | pp60000 | tg128@10000 | tg128@20000 | tg128@40000 | tg128@60000 |
|---|---|---|---|---|---|---|---|---|
| hc-def | 10 541 | 9 499 | 7 989 | 6 866 | 72,5 | 67,1 | 57,8 | 50,9 |
| hc-sw | 10 795 | 9 917 | 8 619 | 7 617 | 76,2 | 73,9 | 69,0 | 64,7 |
| llama-cuda | 11 257 | 10 512 | 9 064 | 7 955 | 72,0 | 69,6 | 65,2 | 61,3 |
| vllm | 10 620 | 9 913 | 8 696 | 7 740 | 73,4 | 71,7 | 67,4 | 63,4 |
| sglang | 10 818 | 10 109 | — | — | 71,7 | 70,2 | — | — |

Écarts entre les rounds : au plus 1,2 % (llama-cuda pp20000 en 0.8B). L'exception est hc-sw tg128@10000 en 0.8B, à 2,1 %.

## Rapports hc-sw / comparateur (médiane des rapports par round ; `summary.txt` donne aussi le min..max)

| 0.8B | pp10000 | pp20000 | pp40000 | pp60000 | tg128@10000 | tg128@20000 | tg128@40000 | tg128@60000 |
|---|---|---|---|---|---|---|---|---|
| llama-cuda | ×1,171 | ×1,137 | ×1,075 | ×1,050 | ×1,202 | ×1,195 | ×1,176 | ×1,162 |
| vllm | ×1,098 | ×1,062 | ×1,023 | ×1,017 | ×1,072 | ×1,059 | ×1,040 | ×1,026 |
| sglang | ×1,029 | ×1,035 | ×0,997 | ×0,985 | ×1,217 | ×1,201 | ×1,177 | ×1,161 |

| 4B | pp10000 | pp20000 | pp40000 | pp60000 | tg128@10000 | tg128@20000 | tg128@40000 | tg128@60000 |
|---|---|---|---|---|---|---|---|---|
| llama-cuda | ×0,958 | ×0,944 | ×0,950 | ×0,957 | ×1,059 | ×1,061 | ×1,058 | ×1,055 |
| vllm | ×1,017 | ×1,000 | ×0,991 | ×0,984 | ×1,039 | ×1,030 | ×1,023 | ×1,021 |
| sglang | ×0,998 | ×0,981 | — | — | ×1,064 | ×1,053 | — | — |

hc-def / comparateur, même passage, à 60K :
- 0.8B : ×0,898 contre llama-cuda, ×0,869 contre vLLM et ×0,842 contre SGLang au prefill ; ×0,557, ×0,492 et ×0,557 au decode ;
- 4B : ×0,862 contre llama-cuda et ×0,887 contre vLLM au prefill ; ×0,830 et ×0,801 au decode.

Ce sont les valeurs de la phase 5b à ±0,003 près.

hc-sw / hc-def, de 10K à 60K :
- 0.8B, prefill : ×1,051, ×1,087, ×1,137 et ×1,169 ;
- 0.8B, decode : ×1,237, ×1,448, ×1,799 et ×2,085 ;
- 4B, prefill : ×1,025, ×1,045, ×1,080 et ×1,110 ;
- 4B, decode : ×1,052, ×1,100, ×1,193 et ×1,272.

## Tokens

- **Stabilité** : chaque unité de herbert rend la même suite de `[ids]` à ses 5 rounds, à chaque longueur.
- **hc-sw contre hc-def**, sur la ligne `[ids]` du round 1 (127 éléments) :
  - identiques en 0.8B à 20K, et en 4B à 10K, 20K et 40K ;
  - différents dès l'élément 10 en 0.8B à 10K, 22 à 40K, 17 à 60K, et 23 en 4B à 60K.
  - `sdpa.pipe` rendant les mêmes suites que le défaut, l'écart vient du decode de `sdpa.gqa`, qui somme dans un autre ordre.
  - Aucun oracle CPU à ces longueurs.

## Horloges

- **Ensemble de la mesure** : 4 926 échantillons sur 2 805 / 11 201 MHz en P0, les autres entre 2 685 et 2 790 MHz, toujours en P0.
- **Au-dessus de 100 W** : le SM est sous 2 805 MHz sur 2 444 des 3 953 échantillons, jusqu'à 2 430 MHz.
- **Raison** : `0x4` (limite logicielle de puissance) sur 2 408 d'entre eux, comme en 5a et 5b. Toutes les unités sont concernées, alternées dans les mêmes conditions.

## Fichiers

- `av1.out` : la sortie du script (heures de chaque unité, arrêt et relance du service résident).
- `av1/` :
  - `chat/` : les journaux de herbert ;
  - `llama/` : json et stderr de `llama-bench` ;
  - `vllm/` : json et journaux ;
  - `sglang/` : `results.jsonl` et les journaux ;
  - `clocks.csv` : un échantillon par seconde ;
  - `launches.txt`, `env-start.txt`, `env-end.txt`, `end.txt`.
- `summary.txt` : la sortie de `_scripts/_av1_summary.py av1`.
- `_scripts/` : le script de la mesure et le résumé (copies de `herbert-cuda-bench/alt-vs/`).
