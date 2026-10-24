# Phase 2 de la campagne de #003 : herbert-cuda contre les comparateurs à leur meilleur réglage (2026-10-18)

Protocole : `protocol.md`, section « Phase 2 », commité avant toute mesure (`herbert-cuda-bench` `726a484`). Les prédictions P1 à P9 sont dans `predictions.md`, écrites le matin avant tout balayage. herbert-cuda est mesuré sous deux formes côte à côte, le binaire **nu** (aucune option) et le **candidat** (`--tune gemm.auto=3 --tune gemm.layout=xor --tune prefill.block=auto`). Ce que désigne « herbert-cuda » dans #003 reste à décider.

**Machine** : RTX 4080 (Ryzen 9 9900X, Linux), pilote 595.99.02, SM 2 805 et mémoire 11 201 MHz verrouillés. Tous les moteurs CUDA tournent en P0 grâce au profil `CudaNoStablePerfLimit`, complété avant la phase pour `herbert-cuda-serve`, `vllm` et `transformers` ; l'ancien fichier est dans `preparation/`. Relevés à la seconde (`phase2/clocks.csv`) : 4 878 échantillons en P0 à 2 805 / 11 201. Les 613 échantillons en P2 à 10 801 MHz tombent tous dans le contrôle sans profil (18:17–18:31). Une centaine d'échantillons ont le SM entre 2 715 et 2 790 MHz.

**Versions** (`phase2/env-start.txt`) :
- herbert-cuda : arbre `af9a631`, dont le code est celui de `71920b9`, reconstruit avec la feature `serve` (`preparation/build-phase2.log`, sha256 dans `env-*.txt`) ;
- herbert-vk-bf16 : `2f3ae48` (la build de #002) et son serveur `9738be7` (celui de V2) ;
- comparateurs : llama.cpp `b11718` (CUDA et Vulkan), vLLM 0.31.0, SGLang 0.5.22, transformers 5.20.1. Dans le venv de transformers, ont été ajoutés les extras `serving`, `requests`, Pillow et torchvision 0.29.1, sans toucher à torch 2.14.1+cu130 (`preparation/install-transformers-serving.log`) : `transformers serve` charge le processeur d'images de Qwen3.5.

## Instrument A : le banc de chaque moteur (`phase2/`, `_scripts/_p2-003.sh`)

- herbert-cuda (nu, candidat) et herbert-vk : leur binaire de chat, comme V1 de #002, avec `--warmup 1` et la ligne `[turn]`. Prefill sur les fixtures `pp128`, `pp512` (513 tokens), `pp2048` et `pp4000` (4 001 tokens), sans réflexion. Décodage sur p3 (759 + 255, réflexion). herbert-vk passe aussi avec `--tune host.poll=1` au décodage.
- llama.cpp : `llama-bench`, chaque fois avec les deux meilleurs réglages de la phase 1.
- Alternance : herbert-cuda, herbert-vk et llama.cpp tour par tour, N = 5, ordre inversé un tour sur deux.
- vLLM (`-cc.max_cudagraph_capture_size=512`, backend par défaut et `TRITON_ATTN`), SGLang (défaut et triton) et transformers (`sdpa` et `eager`, en cache statique compilé) : un passage par réglage et par modèle, avec leurs répétitions internes (5, 3 et 5).
- Pour chaque comparateur, on retient le meilleur de ses deux réglages, forme par forme.

Médianes en tok/s (`summary.txt` donne le réglage retenu, n et l'écart de chaque cellule) :

| 0.8B | pp128 | pp512 | pp2048 | pp4000 | décodage p3 |
|---|---|---|---|---|---|
| herbert-cuda nu | 25 970 | 46 516 | 49 285 | 47 273 | 401,6 |
| herbert-cuda candidat | 27 906 | 49 747 | 62 735 | 60 147 | 401,0 |
| herbert-vk `2f3ae48` | 27 218 | 49 121 | 58 729 | 56 830 | 341,8 (380,4 avec `host.poll=1`) |
| llama.cpp CUDA b11718 | 15 675 | 34 440 | 39 014 | 40 579 | 325,3 |
| llama.cpp Vulkan b11718 | 7 131 | 11 896 | 14 186 | 13 934 | 306,0 |
| vLLM 0.31.0 | 10 866 | 42 277 | 54 079 | 53 235 | 375,9 |
| SGLang 0.5.22 | 14 439 | 47 366 | 64 225 | 58 455 | 337,5 |
| transformers 5.20.1 | 8 712 | 26 333 | 44 218 | 40 880 | 263,9 |

| 4B | pp128 | pp512 | pp2048 | pp4000 | décodage p3 |
|---|---|---|---|---|---|
| herbert-cuda nu | 4 765 | 7 702 | 8 439 | 8 225 | 78,6 |
| herbert-cuda candidat | 6 518 | 9 594 | 11 575 | 11 222 | 78,5 |
| herbert-vk `2f3ae48` | 6 193 | 9 439 | 10 791 | 10 523 | 75,5 (77,6 avec `host.poll=1`) |
| llama.cpp CUDA b11718 | 5 247 | 8 921 | 9 390 | 9 249 | 74,4 |
| llama.cpp Vulkan b11718 | 1 928 | 2 631 | 2 829 | 2 771 | 70,3 |
| vLLM 0.31.0 | 5 627 | 9 866 | 10 964 | 10 873 | 76,6 |
| SGLang 0.5.22 | 6 098 | 10 675 | 11 362 | 11 327 | 75,1 |
| transformers 5.20.1 | 5 670 | 8 995 | 9 337 | 8 801 | 68,0 |

Écarts à connaître :
- transformers compilé : 45 à 92 % au prefill, parce que la première mesure de chaque forme recompile ; la médiane ne la voit pas.
- vLLM 4B à 128 tokens : 8,4 %.
- SGLang 0.8B à 4 000 tokens : 9,8 %.
- Ailleurs : moins de 5 %.

## Face aux prédictions (ratio herbert-cuda / moteur, forme par forme ; valeurs, sans verdict)

| prédiction | candidat 0.8B | candidat 4B | nu 0.8B | nu 4B |
|---|---|---|---|---|
| P1 prefill / llama.cpp CUDA | ×1,44 à ×1,78 | ×1,08 à ×1,24 | ×1,17 à ×1,66 | ×0,86 à ×0,91 |
| P2 prefill / vLLM | ×1,13 à ×2,57 | ×0,97 à ×1,16 | ×0,89 à ×2,39 | ×0,76 à ×0,85 |
| P3 prefill / SGLang | ×0,98 à ×1,93 | ×0,90 à ×1,07 | ×0,77 à ×1,80 | ×0,72 à ×0,78 |
| P4 décodage / llama.cpp CUDA | ×1,233 | ×1,055 | ×1,234 | ×1,056 |
| P5 décodage / vLLM | ×1,067 | ×1,025 | ×1,068 | ×1,026 |
| P6 décodage / SGLang | ×1,188 | ×1,046 | ×1,190 | ×1,046 |
| P7 prefill / transformers | ×1,42 à ×3,20 | ×1,07 à ×1,28 | ×1,12 à ×2,98 | ×0,84 à ×0,94 |
| P8 décodage / transformers | ×1,519 | ×1,154 | ×1,521 | ×1,155 |
| P9 prefill / herbert-vk | ×1,01 à ×1,07 | ×1,02 à ×1,07 | ×0,83 à ×0,95 | ×0,77 à ×0,82 |
| P9 décodage / herbert-vk (avec `host.poll=1`) | ×1,173 (×1,054) | ×1,040 (×1,012) | ×1,175 (×1,056) | ×1,041 (×1,013) |

Le détail par longueur est dans `summary.txt`, section « Ratios ».

## Instrument B : le client HTTP commun (`phase2/http/`, `phase2c/`)

C'est le client de V2, en `_scripts/_http_client.py` :
- Messages API en flux (SGLang et transformers par l'API OpenAI) ;
- glouton, une requête à la fois, sans cache de préfixe ;
- mêmes prompts que V2, 128 tokens générés, 3 tours sur 5 passages par longueur ;
- prefill = tokens d'entrée comptés par le serveur / TTFT ; décodage mesuré entre le premier et le dernier delta.

Chaque serveur tourne seul, carte en P0 (`pstate-*.txt`).

**Les chiffres de transformers dans `phase2/http/` sont invalides.** `transformers serve` envoie un delta `reasoning_content` « \n » avant que le modèle ait calculé quoi que ce soit. Vérifié (`phase2/http-check/`) : sur le 4B et 4 096 tokens, le « \n » arrive à 9,9 ms et le premier token du modèle à 503 ms. Le client prenait ce « \n » pour le premier token. J'ai refait transformers avec l'option `--skip-blank` du client, qui ignore les deltas faits seulement d'espaces (`phase2c/`). Deux serveurs ont été repassés avec la même option en contrôle : herbert-cuda candidat bouge de 0 à 1,8 %, vLLM de 0 à 2,9 %, sur le prefill comme sur le décodage. Les lignes transformers ci-dessous viennent de `phase2c/`, les autres de `phase2/http/`.

| 0.8B, serveur | prefill p500 (~510 tok) | p2048 (~2 058) | p4096 (~4 106) | décodage p500 |
|---|---|---|---|---|
| herbert-cuda nu | 44 109 | 43 071 | 42 902 | 404,3 |
| herbert-cuda candidat | 47 494 | 58 055 | 53 543 | 403,9 |
| herbert-vk `9738be7` | 38 955 | 46 366 | 47 652 | 360,7 |
| llama-server CUDA (`-ub 1024 -b 4096 -fa on`) | 16 779 | 24 770 | 28 303 | 308,5 |
| llama-server Vulkan (`-ub 2048 -b 4096 -fa on`) | 7 565 | 10 694 | 11 437 | 292,4 |
| vLLM cg512 / triton cg512 | 28 307 / 27 286 | 44 144 / 44 942 | 47 392 / 46 272 | 367,6 / 377,4 |
| SGLang défaut / triton | 38 984 / 39 663 | 49 189 / 48 818 | 54 254 / 51 501 | 384,4 / 386,6 |
| transformers sdpa / sdpa compile | 24 981 / 22 604 | 37 688 / 36 684 | 36 595 / 35 601 | 100,0 / 256,6 |

| 4B, serveur | prefill p500 | p2048 | p4096 | décodage p500 |
|---|---|---|---|---|
| herbert-cuda nu | 8 093 | 7 613 | 7 790 | 79,4 |
| herbert-cuda candidat | 9 910 | 10 490 | 10 661 | 79,3 |
| herbert-vk `9738be7` | 9 513 | 9 399 | 9 722 | 76,1 |
| llama-server CUDA (défaut) | 4 310 | 6 255 | 7 164 | 73,6 |
| llama-server Vulkan (`-ub 2048 -b 4096 -fa on`) | 1 794 | 2 380 | 2 507 | 69,0 |
| vLLM cg512 / triton cg512 | 8 920 / 8 912 | 9 823 / 9 718 | 10 229 / 10 011 | 76,5 / 76,9 |
| SGLang défaut / triton | 10 172 / 10 196 | 10 353 / 10 193 | 10 670 / 10 363 | 77,6 / 77,6 |
| transformers sdpa / sdpa compile | 8 360 / 8 078 | 8 575 / 8 403 | 8 174 / 8 136 | 59,8 / 68,6 |

Le TTFT de chaque longueur, les prompts courts (p10, p100) et les écarts sont dans `summary.txt` et `http-*.json`. vLLM compte 2 tokens d'entrée de plus que les autres serveurs sur les prompts courts, à cause de son gabarit. Sur le 0.8B, vLLM s'arrête parfois avant 128 tokens (fin de tour) sur p10 et p100.

## Contrôle sans le profil P0 (`phase2/control/`)

Le profil a été retiré de 18:17 à 18:31, puis remis par un `trap` (vérifié dans `phase2.out`). La carte passe en P2, mémoire à 10 801 MHz. Trois tours de herbert-cuda (nu et candidat) et de llama.cpp CUDA par défaut, plus un passage vLLM cg512 et un passage SGLang par défaut. Ratio sans profil / avec profil :
- décodage : ×0,963 à ×0,973 pour tous les moteurs CUDA ;
- prefill : ×0,973 à ×1,002, sauf vLLM 0.8B : ×1,014 à 128 tokens et ×1,036 à 2 048 ; les écarts de ces deux cellules dans le passage avec profil sont de 1,0 et 2,0 %.

## llama.cpp b11719 contre b11718 (`phase2b/`, `_scripts/_p2b-llama-b11719-003.sh`)

b11719 est sortie à 10:59, après le début de la phase 1 sur b11718 (10:38). Les prédictions demandent la dernière version stable au moment de la campagne. Elle change une ligne du noyau d'attention CUDA en tuiles (`ggml-cuda/fattn-tile.cuh`, `5aeac37a3`) ; Vulkan n'est pas touché. Construite après la phase 2 avec la même configuration, puis alternée avec b11718 (N = 5, deux réglages de prefill et deux de décodage) : ×0,996 à ×1,002 sur toutes les formes. Ses tokens contre l'oracle sont ceux de b11718 (`phase2b/tokens-diff.txt`, à comparer à la phase 1).

## Fichiers

- `summary.txt` : sortie de `_p2_summary.py phase2 phase2b phase2c`.
- `phase2.out`, `phase2b.out`, `phase2c.out` : sorties des files. Les lignes « Killed » de `phase2.out` sont l'arrêt forcé des serveurs SGLang après 60 s de SIGTERM, et non des échecs de mesure.
- `preparation/` : build, installation, ancien profil, smoke tests.
- `_scripts/` : les scripts lancés. `_http_client.py` est la version avec `--skip-blank`. La phase 2 a utilisé celle de `herbert-cuda-bf16/bench/`, identique sans l'option.
