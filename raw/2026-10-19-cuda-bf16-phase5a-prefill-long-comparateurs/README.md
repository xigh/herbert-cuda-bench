# Phase 5 de #003, étape 5a : le prefill long (10 000 à 60 000 tokens), tokens entre moteurs et meilleur réglage des comparateurs (2026-10-19)

Étape 5a du protocole de la phase 5 (`protocol.md`, section « Phase 5 », écrit vers 16 h avant toute mesure), après les prédictions P14–P15 (`predictions.md`, dictées à 16 h 06, `herbert-cuda-bench` `acdf81c`, complétées en `d400593`). Objet : regarder le prefill avec beaucoup plus de tokens, de 10K jusqu'à 60K. Cette étape fait deux choses. D'abord, chaque moteur génère 64 tokens gloutons après chaque prompt long, et les suites sont comparées moteur contre moteur : l'oracle CPU réserve 8 192 positions, il ne sert pas ici. Ensuite, le balayage de réglages des comparateurs est refait aux longueurs longues, prefill seulement, pour retenir deux réglages par moteur pour l'étape 5b. Aucune vitesse de herbert dans cette étape, hors la vérification de démarrage (plus bas).

**Machine** : RTX 4080 (Ryzen 9 9900X, Linux), pilote 595.99.02, SM 2 805 et mémoire 11 201 MHz verrouillés, profil `CudaNoStablePerfLimit` (P0 pour tous les moteurs CUDA). Relevés à la seconde (`phase5a/clocks.csv`, `clocks-a2.csv`) : mémoire à 11 201 MHz et P0 sur tous les échantillons (14 335). **Le SM est sous 2 805 MHz sur 5 042 des 13 505 échantillons de 5a**, jusqu'à 2 250 MHz. Sur ces 5 042, la raison active est la limite logicielle de puissance (`0x4`) pour 4 264, aucune raison pour 772. Sous 2 700 MHz, la puissance médiane est de 314 W (au plus 320,5) : aux longueurs longues, le prefill atteint la limite de puissance de la carte, à horloges verrouillées. En 5a2, le SM est sous 2 805 MHz sur 68 des 830 échantillons, à 2 670 au plus bas.

**Versions** (`phase5a/env.txt`) : herbert-cuda-bf16 `011bfdf` (code de `638364b` plus `--ask-file`), herbert-cuda-f16 `0c7d88d` (défaut et `--tune gemm.f16=1`), herbert-vk-f16 `cc51363`. llama.cpp `b11718` (CUDA et Vulkan) sur les GGUF F16. vLLM 0.31.0, SGLang 0.5.22 (`SGLANG_MAMBA_CONV_DTYPE=float16`) et transformers 5.20.1, tous les trois en float16. Contexte de 61 440 positions partout.

## Prompts

`long-prompts/pp{10000,20000,40000,60000}.txt` (`_scripts/_build_long_prompts.py`) : prose de Moby-Dick, chacun pris à un endroit différent du livre. Longueur exacte gabarit compris (12 tokens avec `--no-system --no-think`). Pour les tokens, les ids sont ceux de `herbert-cuda-bench --prepare` (`phase5a/tokens/cases-{0.8B,4B}.json`), les mêmes pour tous les moteurs. Les bancs de vLLM et SGLang tirent des ids aléatoires de longueur N, comme aux phases 1 à 4. transformers prend les ids du texte de `pp60000.txt`, coupés à N ; à 60 000, ce sont ceux du cas `pp60000`, gabarit compris (plus bas). `prompts-long.json` (`_build_long_http_prompts.py`) contient cinq passages par longueur pour le client HTTP de 5b.

## Les passages, dans l'ordre (tous gardés)

- `phase5-smoke/`, `phase5-smoke.out`, `smoke-p2/` (`_scripts/_p5-smoke-003.sh`) : vérification de démarrage à 60 000 tokens, faite avant les prédictions. Une passe par moteur, sans comparaison. Ses temps n'ont pas été communiqués avant les prédictions ; ils sont déposés ici, bruts. Elle a montré deux choses. Un argument de ligne de commande est limité à 128 Kio sous Linux, d'où `--ask-file` dans herbert-cuda. SGLang à `--mem-fraction-static 0.8` manque de mémoire à 60K.
- `phase5a/` (`phase5a.out`, `_scripts/_p5a-003.sh`, `herbert-cuda-bench` `9161e84`), de 16 h 08 à 19 h 50 :
  - les tokens ;
  - le balayage de llama.cpp : 17 réglages, défaut et `-ub` × `-b` × `-fa`, un processus `llama-bench` par réglage, 3 tours alternés ;
  - vLLM : 7 réglages, 5 mesures chacun ;
  - SGLang : 8 réglages, dont `--mem-fraction-static` 0,7 et 0,6, un processus par longueur, 3 mesures ;
  - transformers : 6 réglages, 5 mesures.
- `phase5a2.out` (`_scripts/_p5a2-003.sh`, `herbert-cuda-bench` `5db1f2b`), de 19 h 56 à 20 h 10, même arborescence `phase5a/`. Deux reprises :
  - **transformers à 60 000 tokens, refait.** En 5a, les 12 cas à 60 000 avaient échoué sur la vérification de longueur du script, pas sur la mémoire : le texte brut de `pp60000.txt` fait 59 988 tokens, puisque les 12 tokens du gabarit n'y sont pas (`AssertionError(torch.Size([1, 59988]))`). Refait avec les ids du cas `pp60000` (`CASES_JSON`, 60 000 tout juste).
  - **SGLang, là où aucun réglage n'avait tourné** (4B à 40K et 60K, 0.8B à 60K hors 0,6) : fractions 0,72, 0,75 et 0,78, et `--max-running-requests 1` à 0,8 et à 0,7.

`summary.txt` est la sortie de `_p5a_summary.py phase5a` ; `tokens-diff.txt` celle de `_p5_tokens_diff.py phase5a/tokens`.

## Tokens : 64 tokens gloutons après chaque prompt long, moteur contre moteur

Classes de suites identiques sur leur longueur commune, et premier token qui diffère (indice à partir de 0) contre la classe de herbert-cuda-bf16. herbert-vk-f16 (chat, 10K et 20K seulement) imprime 63 ids générés au lieu de 64. Réglages des comparateurs : ceux des tokens de 4a, à `max_model_len` / `context_length` 61 440 ; SGLang à 0,6.

| modèle | prompt | classes |
|---|---|---|
| 0.8B | 10K | hc-bf16 seul. Les 20 autres suites sont identiques entre elles et s'écartent de hc-bf16 au token 17 : hc-f16, hc-f16 `gemm.f16=1`, herbert-vk-f16, les 6 réglages de llama.cpp, les 4 de SGLang, les 3 de transformers et les 4 de vLLM. |
| 0.8B | 20K | Même partage, écart au token 32. transformers eager-dynamic manque de mémoire. |
| 0.8B | 40K | hc-bf16 seul. Classe 1 (écart 49) : hc-f16, SGLang ×4, transformers sdpa ×2, vLLM défaut, flashinfer et triton. Classe 2 (écart 20) : hc-f16 `gemm.f16=1`, llama.cpp Vulkan `-fa off`, vLLM eager. Classe 3 (écart 22) : llama.cpp CUDA défaut. Classe 4 (écart 22) : les 4 autres réglages de llama.cpp. Entre classes 1 et 2 : 20. |
| 0.8B | 60K | hc-bf16 seul. Classe 1 (écart 17) : hc-f16, vLLM eager et flashinfer. Classe 2 (écart 17) : hc-f16 `gemm.f16=1`, SGLang ×4, transformers sdpa ×2, vLLM défaut et triton. Classes 3 à 5 (écart 13) : llama.cpp, en trois groupes. Entre classes 1 et 2 : 32. |
| 4B | 10K, 20K | Les 16 suites sont identiques : les trois herbert-cuda, herbert-vk-f16, llama.cpp ×6, transformers sdpa ×2, vLLM ×4. |
| 4B | 40K | Classe 0 : les trois herbert-cuda, llama.cpp CUDA défaut, llama.cpp Vulkan défaut, transformers sdpa-dynamic, vLLM ×4. Écarts : llama.cpp CUDA `-fa off` et Vulkan `-fa on -ub 2048` au token 6 ; CUDA `-fa on -ub 1024` au 42 ; Vulkan `-fa off` au 16. |
| 4B | 60K | Classe 0 : les trois herbert-cuda, vLLM ×4. llama.cpp : 4 réglages au token 28, CUDA `-fa off` au 14, Vulkan `-fa off` au 39. |

SGLang 4B n'a aucun token : à 0,6, il refuse de servir un contexte de 61 440 (« Hybrid (mamba/linear-attention) state cache is too small to serve any requests … max_num_reqs=0 »). transformers 4B : sdpa-dynamic passe de 10K à 40K, sdpa-static de 10K à 20K, eager-dynamic nulle part. Le reste manque de mémoire (`phase5a/tokens/tokens.jsonl`, lignes `error`).

## Le balayage : meilleur réglage de chaque comparateur (médianes, tok/s)

| moteur | modèle | 10 000 | 20 000 | 40 000 | 60 000 |
|---|---|---|---|---|---|
| llama.cpp CUDA F16 | 0.8B | 47 420 | 42 762 | 35 029 | 29 638 |
| | 4B | 11 310 | 10 602 | 9 120 | 7 993 |
| llama.cpp Vulkan F16 | 0.8B | 27 706 | 25 336 | 21 376 | 18 387 |
| | 4B | 8 613 | 8 050 | 7 102 | 6 327 |
| vLLM | 0.8B | 51 739 | 45 939 | 37 006 | 30 729 |
| | 4B | 10 673 | 9 966 | 8 749 | 7 794 |
| SGLang | 0.8B | 54 456 | 47 167 | 38 014 | 31 770 |
| | 4B | 10 908 | 10 171 | — | — |
| transformers | 0.8B | 41 995 | 32 539 | 28 339 | 26 809 |
| | 4B | 9 257 | 7 967 | 6 824 | — |

Réglage de chaque cellule et tous les autres dans `summary.txt`. Pour llama.cpp CUDA, `-ub 1024 -b 4096 -fa on` est en tête dans les 8 cellules ; pour Vulkan, il l'est en 0.8B, et `-ub 512 -b 4096 -fa on` en 4B. Pour vLLM, `mnbt2048` est en tête au 0.8B à 10K, flashinfer ailleurs. SGLang : `mf06` est le seul réglage à passer le 0.8B à 60K ; à 10K et 20K, les quatre premiers sont à moins de 0,2 % les uns des autres, et la tête change d'une cellule à l'autre (mf07, chunk2048, mf075, `mf07-mrr1`). transformers : sdpa static compile partout où il passe, sdpa-dynamic au 4B à 40K.

**Manques de mémoire** (FAIL dans `summary.txt`, détail dans les journaux) :
- llama.cpp 4B : les 4 réglages `-fa off` à `-ub` 1024 ou 2048 échouent à toutes les longueurs, en CUDA (« CUDA error: out of memory ») comme en Vulkan (« ErrorOutOfDeviceMemory »). Le JSON garde les longueurs passées avant l'échec : la CUDA `-ub 1024 -b 2048 -fa off` donne 10K à 40K au tour 1. Le résumé ne les compte pas. En 0.8B, tout passe.
- SGLang :
  - 0.8B : à 0,8, tous les réglages échouent à 40K et 60K ; à 0,7, à 60K ;
  - 4B : à 0,8, à partir de 20K ; à 0,6, à toutes les longueurs ;
  - **4B à 40K et 60K, aucun des 13 réglages ne passe.** À 0,75 et au-dessus, manque de mémoire ; à 0,72 et en dessous, le pool KV fait moins de 40 000 tokens, et one_batch saute le cas (« skipping … due to max batch size limit ») ;
  - `--max-running-requests 1` ne change rien à ces limites.
- transformers :
  - eager : rien au-delà de 10K en 0.8B, rien en 4B (« Tried to allocate 5.96 GiB » à 20K, 53,64 Gio à 60K) ;
  - 4B à 60K : aucun réglage ne passe ;
  - sdpa static compile 4B : échoue à partir de 40K.

## Deux réglages retenus par moteur pour 5b

Le critère est celui de 4a (`_p4a_pairs.py`) : la paire dont le meilleur des deux s'approche le plus du meilleur de tous les réglages, en moyenne géométrique sur les deux modèles et les quatre longueurs. Les cellules où aucun réglage n'a tourné sont exclues. Ajout de 5a : quand un réglage est en tête partout, toutes les paires qui le contiennent valent 1,0000. On départage alors par la même moyenne prise sur le moins bon des deux, c'est-à-dire le partenaire le plus proche du meilleur lui aussi.

- llama.cpp CUDA : `-ub 1024 -b 4096 -fa on` et `-ub 1024 -b 2048 -fa on` (1,0000 ; second 0,9949) ;
- llama.cpp Vulkan : `-ub 1024 -b 4096 -fa on` et `-ub 512 -b 4096 -fa on` (1,0000) ;
- décodage des deux llama.cpp : défaut et `-fa on`, comme en 4a (le balayage ne mesure pas le décodage) ;
- vLLM : flashinfer et `mnbt2048` (1,0000) ;
- SGLang : mf06 et `mf07-mrr1` (0,7, `--max-running-requests 1`) (0,9998). mf06 ne passe aucune longueur en 4B, et le 4B ne passe nulle part au-delà de 20K ;
- transformers : sdpa-dynamic et sdpa static compile (1,0000).

Les valeurs seulement.
