# 2026-10-25 · Qwen3.5 en bf16 sur la 4080 : decode de 128 tokens après 10 000, 20 000 et 40 000 tokens de prompt (cinq moteurs, trois modèles), et tuiles du GEMM du 2B fixées à la main (pour l'article #003)

Deux tests ajoutés le 2026-10-25 pour l'article #003, par ordre d'importance :
1. Le décodage après un prompt long, face à TensorRT-LLM … mesurer le décodage de 128 tokens après 10 000, 20 000 et 40 000 tokens de prompt, sur les trois modèles et les cinq moteurs, avec la méthode des deux mesures de cette nuit. Il le rattache à une prédiction P17 (« moins bien avec plus de pp ») ; `herbert-cuda-bench/predictions.md` s'arrête à P15 (« herbert se fait battre aussi au décodage », le décodage qui suit un prompt long).
2. Le 2B, le seul modèle où Herbert est derrière au prefill … le choix automatique des tuiles (gemm.auto=3) n'a été réglé que pour les dimensions du 0.8B et du 4B (1 024 et 2 560) … quelques tuiles fixées à la main sur le 2B, à 2 000 et 10 000 tokens, contre le réglage par défaut.

Machine, versions, prompts et réglages des comparateurs : ceux de la mesure du prefill long (`../2026-10-25-cuda-bf16-art-long-prefill`) (`../2026-10-25-cuda-bf16-art-long-prefill/`) ; `runs/dec-long/env.txt`, `runs/tile2b/env.txt`. Horloges verrouillées à 2 805 / 11 201 MHz, vérifiées avant chaque unité, aucune alerte `!!`. Comme dans la mesure du prefill long (`../2026-10-25-cuda-bf16-art-long-prefill`), l'horloge SM descend sous la limite de 320 W pendant les prefills longs : 558 relevés sur 4 110 sous 2 805 MHz pendant le test 1, 30 sur 266 pendant le test 2 ; mémoire à 11 201 MHz partout. `resident-gpu.service` arrêté puis relancé, actif à la fin. Lancés à la suite par `_scripts/_chain-dec-tile.sh`, 06:15–07:29.

## Test 1 : decode après un prompt long (`_scripts/_dec-long-003.sh`)

**Méthode** :
- **herbert-cuda** `31b86e1`, réglages par défaut : `--ctx 40960 --no-system --no-think --max-tokens 128 --warmup 1 --ask-file ppN.txt`. Le débit de decode est celui de la ligne `[turn 1]` : les 127 tokens après le premier, 127 dans tous les runs. Prompts pp10000, pp20000 et pp40000 de la mesure du prefill long (`../2026-10-25-cuda-bf16-art-long-prefill`).
- **llama.cpp** : `llama-bench -p 0 -n 128 -d 10000,20000,40000 -r 1`, réglage par défaut et `-fa on`.
- herbert-cuda et llama.cpp **alternés tour par tour, N = 5** : médiane [min..max].
- **vLLM** et **TensorRT-LLM** : les deux mesures de la mesure de 128 à 4 000 tokens (`../2026-10-25-cuda-bf16-art-qwen35-bf16`) et la mesure du prefill long (`../2026-10-25-cuda-bf16-art-long-prefill`), la latence de bout en bout avec 1 et avec 128 tokens générés (2 chauffes, 5 itérations), puis decode = 127 / (L128 − L1) sur les moyennes, et entre parenthèses sur les médianes. vLLM : attention par défaut et FLASHINFER. TensorRT-LLM : blocs de 8 192 et un seul bloc avec `max_num_tokens` = N.
- **SGLang** : `one_batch`, entrée N ×3, 128 tokens, `median_decode_throughput`, médiane [min..max] des 3. Réglages : mf 0.7, `--max-running-requests 1`.
- Trous attendus d'après la mesure du prefill long (`../2026-10-25-cuda-bf16-art-long-prefill`) : SGLang 4B à 40 000 (« skipping », pool KV trop petit) et TensorRT-LLM 4B en un seul bloc à 40 000 (manque de mémoire).

**Decode en tok/s** (`runs/dec-long/summary.md`, `_scripts/_dec-long-summary.py`) :

### Qwen3.5-0.8B

| Moteur | après 10000 | après 20000 | après 40000 |
|---|---|---|---|
| herbert-cuda | 369,2 [369,0..369,4] | 347,1 [347,1..347,1] | 308,9 [303,8..308,9] |
| llama.cpp défaut | 307,1 [307,0..307,2] | 290,9 [290,8..290,9] | 262,2 [262,0..262,3] |
| llama.cpp -fa on | 307,1 [307,1..307,1] | 290,8 [290,6..291,0] | 262,1 [262,0..262,2] |
| vLLM défaut | 341,3 (méd. 341,4) | 324,8 (méd. 324,9) | 294,7 (méd. 294,6) |
| vLLM FLASHINFER | 344,6 (méd. 344,5) | 327,5 (méd. 327,9) | 298,4 (méd. 298,5) |
| TensorRT-LLM blocs de 8192 | 344,8 (méd. 344,8) | 327,6 (méd. 327,7) | 289,7 (méd. 289,6) |
| TensorRT-LLM un bloc, mnt N | 344,5 (méd. 344,5) | 328,5 (méd. 328,8) | 291,0 (méd. 291,5) |
| SGLang | 304,1 [304,0..304,3] | 290,1 [289,9..290,2] | 262,7 [262,7..262,9] |

### Qwen3.5-2B

| Moteur | après 10000 | après 20000 | après 40000 |
|---|---|---|---|
| herbert-cuda | 167,9 [166,7..167,9] | 163,0 [162,2..163,0] | 154,2 [154,2..154,2] |
| llama.cpp défaut | 153,8 [153,8..153,8] | 149,6 [149,5..149,6] | 141,7 [141,7..141,7] |
| llama.cpp -fa on | 153,8 [153,8..153,8] | 149,6 [149,5..149,6] | 141,7 [141,7..141,7] |
| vLLM défaut | 159,8 (méd. 159,7) | 156,6 (méd. 156,6) | 149,7 (méd. 149,8) |
| vLLM FLASHINFER | 160,8 (méd. 160,8) | 157,8 (méd. 157,8) | 150,5 (méd. 150,6) |
| TensorRT-LLM blocs de 8192 | 161,4 (méd. 161,5) | 157,8 (méd. 158,0) | 149,0 (méd. 148,9) |
| TensorRT-LLM un bloc, mnt N | 161,7 (méd. 161,8) | 158,0 (méd. 158,0) | 149,2 (méd. 149,3) |
| SGLang | 151,5 [151,5..151,5] | 148,3 [148,2..148,3] | 141,2 [141,1..141,2] |

### Qwen3.5-4B

| Moteur | après 10000 | après 20000 | après 40000 |
|---|---|---|---|
| herbert-cuda | 76,1 [76,1..76,1] | 73,9 [73,9..73,9] | 69,0 [69,0..69,0] |
| llama.cpp défaut | 71,9 [71,9..71,9] | 69,5 [69,5..69,5] | 65,1 [65,1..65,1] |
| llama.cpp -fa on | 71,9 [71,9..71,9] | 69,5 [69,5..69,5] | 65,1 [65,1..65,1] |
| vLLM défaut | 73,4 (méd. 73,4) | 71,8 (méd. 71,8) | 67,3 (méd. 67,2) |
| vLLM FLASHINFER | 73,6 (méd. 73,6) | 72,0 (méd. 72,0) | 67,5 (méd. 67,5) |
| TensorRT-LLM blocs de 8192 | 74,4 (méd. 74,3) | 72,1 (méd. 72,1) | 67,4 (méd. 67,5) |
| TensorRT-LLM un bloc, mnt N | 74,3 (méd. 74,3) | 72,0 (méd. 72,0) | — |
| SGLang | 71,8 [71,8..71,8] | 70,1 [70,1..70,2] | — |

Pour mémoire, le decode p3 (255 tokens après 759) de la mesure de 128 à 4 000 tokens (`../2026-10-25-cuda-bf16-art-qwen35-bf16`) : 0.8B 404,6 (herbert-cuda), 378,4 (TensorRT-LLM), 365,8 (vLLM), 332,8 (SGLang), 325,3 (llama.cpp) ; 2B 174,7, 168,9, 166,3, 159,2, 158,3 ; 4B 78,7, 77,0, 76,2, 74,6, 74,4.

## Test 2 : tuiles du GEMM du 2B (`_scripts/_tile2b-003.sh`)

**Ce que fait le code** (`31b86e1`) :
- `src/cuda/engine.rs:249` : `gemm.auto=3` n'a de choix mesurés que pour hidden 2560 (4B) et 1024 (0.8B), sur cette carte. Le 2B (hidden 2048) retombe sur la politique 1. Aux hauteurs de 512 et plus, celle-ci prend la tuile 64×128, K 32, trois étages (`Tile::T64X128`). Le 0.8B (à partir de 1 024 lignes) et le 4B (à partir de 512) reçoivent la 64×128 warps 2×2, K 32, deux étages (`T64X128_W2X2_S2`).
- `engine.rs:318` : `prefill.block=auto` donne 2 048 lignes au 4B, 4 096 au 0.8B et 1 024 à tout autre modèle, dont le 2B.

Les deux facteurs sont donc séparés dans les variantes.

**Variantes** (herbert-cuda seul, 2B, pp2000 et pp10000 de la mesure du prefill long (`../2026-10-25-cuda-bf16-art-long-prefill`), `--ctx 40960`, 1 token, `--warmup 1`, alternées sur N = 5 tours, ordre inversé un tour sur deux) :
- `def` : le défaut, soit `gemm.auto=3`, `gemm.layout=xor` et des blocs de 1 024 ;
- `def-b2048`, `def-b4096` : le défaut avec la taille de bloc du 4B ou du 0.8B ;
- `t64x128` : `gemm.auto=0`, 64×128, K 32, trois étages à toutes les hauteurs (la tuile large de la politique 1), blocs de 1 024 ;
- `w2x2s2` : `gemm.auto=0`, 64×128 warps 2×2, K 32, deux étages, blocs de 1 024 ;
- `w2x2s2-b2048`, `w2x2s2-b4096` : la même tuile avec des blocs de 2 048 ou de 4 096 ;
- `t128-k64s2` : `gemm.auto=0`, 128×128 warps 4×4, K 64, deux étages, blocs de 1 024.

Les lignes `[tuning]` de chaque variante sont dans `runs/tile2b/summary.md`.

**Prefill en tok/s** (`runs/tile2b/summary.md`, `_scripts/_tile2b-summary.py`), médiane [min..max] et rapport des médianes à `def` :

| Variante | pp2000 | pp10000 | × `def` pp2000 | × `def` pp10000 |
|---|---|---|---|---|
| def | 25 684 [25635..25754] | 24 524 [24510..24530] | 1,000 | 1,000 |
| def-b2048 | 26 796 [26690..26917] | 25 570 [25554..25584] | 1,043 | 1,043 |
| def-b4096 | 26 755 [26724..26908] | 25 749 [25703..25770] | 1,042 | 1,050 |
| t64x128 | 25 619 [25580..25640] | 24 527 [24502..24568] | 0,997 | 1,000 |
| w2x2s2 | 27 722 [27650..27879] | 26 419 [26353..26429] | 1,079 | 1,077 |
| w2x2s2-b2048 | 29 056 [29017..29291] | 27 706 [27655..27724] | 1,131 | 1,130 |
| w2x2s2-b4096 | 29 064 [28996..29263] | 27 805 [27759..27860] | 1,132 | 1,134 |
| t128-k64s2 | 24 459 [24435..24465] | 23 456 [23437..23470] | 0,952 | 0,956 |

Le token généré est le même pour toutes les variantes : 27775 à pp2000, 1919 à pp10000. Un seul token est généré ; la suite n'est pas comparée.

Pour situer ces valeurs, voici le 2B de la mesure du prefill long (`../2026-10-25-cuda-bf16-art-long-prefill`) dans un autre travail (même méthode, mêmes prompts) :
- herbert-cuda par défaut : 25 572 à pp2000, 24 416 à pp10000 ;
- TensorRT-LLM : 27 692 et 26 183 (meilleur réglage) ;
- SGLang : 27 621 et 26 906 ;
- vLLM : 25 604 et 26 070.

`runs/tile2b-check/check.log` n'est qu'un contrôle de syntaxe des variantes, lancé sans chauffe et avec `resident-gpu` actif : ce n'est pas une mesure.

## Fichiers

- `runs/dec-long/` : toutes les sorties du test 1, `env.txt`, `clocks.csv`, `summary.md`, `summary.json` ;
- `runs/tile2b/` : toutes les sorties du test 2 ;
- `runs/chain-dec-tile.out` : la sortie de la chaîne ;
- `_scripts/` : identiques à ceux lancés.

Les valeurs seulement. Le moteur n'a pas été modifié.
