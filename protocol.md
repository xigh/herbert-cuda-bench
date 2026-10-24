# Campagne de #003 (CUDA) — phase 1 : le meilleur réglage de chaque comparateur

> Copie publique. Ce fichier est le protocole de la campagne, écrit avant les mesures de chaque phase. Cette version publie les phases 1, 2 et 5 (étapes 5a et 5b, sur la RTX 4080) ; les phases 3, 4 et 6 n'y sont pas, ni l'étape 5c. Les renvois à la phase 4 qui restent dans le texte de la phase 5 sont gardés tels quels. La machine est une RTX 4080 dans un Ryzen 9 9900X sous Linux ; les chemins locaux sont remplacés par des chemins neutres (`$MODELS_DIR`, `$BENCH_DIR`, `~/`). Les commits nommés (`71920b9`, `011bfdf`…) sont ceux de l'historique de développement des moteurs, publiés sans historique : voir le `README.md`.

Écrit le 18 octobre 2026, après les prédictions (`predictions.md`, 10 h 30 et complément) et avant toute mesure de cette phase. But : trouver, pour chaque moteur de comparaison, le réglage qui lui donne le meilleur prefill et le meilleur décodage sur la 4080, une requête à la fois, pour que la campagne compare herbert-cuda aux comparateurs à leur meilleur, et pas seulement à leur défaut. Le défaut est toujours mesuré à côté, dans le même balayage.

## Machine et conditions

- La machine RTX 4080 (Ryzen 9 9900X, Linux), pilote 595.99.02, SM verrouillé à 2 805 MHz et mémoire à 11 201 MHz.
- Tous les moteurs CUDA en P0, par le profil d'application `CudaNoStablePerfLimit`.
- Horloges relevées chaque seconde pendant toute la phase ; un contrôle `nvidia-smi` au début de chaque moteur.
- Pas d'autre charge GPU. La charge CPU de la machine est relevée (`uptime`) à chaque tour : d'autres travaux s'y exécutent.

## Formes

- Prefill : 128, 512, 2 048 et 4 000 tokens, un token généré.
- Décodage : 255 tokens à la profondeur 759 (forme p3).
- Deux modèles, Qwen3.5-0.8B et Qwen3.5-4B, en bf16 (safetensors, GGUF BF16 pour llama.cpp).

## Moteurs et réglages balayés

**llama.cpp, dernière release (`b11718`), CUDA et Vulkan.** C'est `llama-bench` :
- micro-lot `-ub` 256, 512, 1 024 et 2 048, lot `-b` 2 048 et 4 096 (`-b` ≥ `-ub`), flash attention `-fa` 0 et 1 ;
- prefill `-p` aux quatre longueurs avec `-n 0`, décodage `-p 0 -n 255 -d 759` ;
- `-r 3`, un processus par réglage, réglages alternés.

**vLLM 0.31.0.** C'est `vllm bench latency` (`--batch-size 1`, 2 tours de chauffe, 5 mesures ; prefill = longueur / latence à 1 token, décodage = 254 / (latence à 255 − latence à 1)), comme #002. Réglages :
- défaut (graphes CUDA, compilation par défaut) ;
- `--enforce-eager` ;
- `--max-num-batched-tokens` à 2 048 et 16 384 (taille des morceaux du prefill) ;
- le backend d'attention par `VLLM_ATTENTION_BACKEND` (FLASH_ATTN, FLASHINFER, TRITON_ATTN) quand le modèle l'accepte.

Un réglage refusé par le moteur est noté comme tel.

**SGLang 0.5.22.** C'est `python -m sglang.bench_one_batch` (`--batch-size 1`, mêmes définitions du prefill et du décodage), avec `CUDA_HOME=/usr/local/cuda-12.8`. Réglages :
- défaut ;
- `--disable-cuda-graph` ;
- `--attention-backend` flashinfer et triton ;
- `--chunked-prefill-size` 2 048 et désactivé (−1) ;
- `--enable-torch-compile`.

**transformers 5.20.1.** C'est `generate` greedy dans un script chronométré (premier token et décodage séparés par un `TextStreamer` horodaté), flash-linear-attention installé, `causal-conv1d` absent (noté). Réglages :
- `attn_implementation` sdpa et eager ;
- cache dynamique contre cache statique ;
- cache statique avec `torch.compile`.

**Références de #002 remesurées dans les mêmes conditions** : llama.cpp `08ffba21e` et vLLM 0.28.0 à leur défaut. Elles servent à séparer ce qui vient des versions de ce qui vient de la P0.

## Ce qui sort de cette phase

- Pour chaque moteur, modèle et forme : la médiane de chaque réglage, son écart, et le meilleur réglage, avec le défaut à côté.
- Les tokens de sortie de chaque moteur, à son meilleur réglage, contre l'oracle CPU sur p1–p4 (la justesse avant la vitesse).
- Les logs complets.

Aucune comparaison avec herbert-cuda dans cette phase : elle vient en phase 2, aux réglages retenus ici, avec les deux instruments (bancs des moteurs et client commun).

# Phase 2 : herbert-cuda contre les comparateurs à leur meilleur réglage

Écrit le 18 octobre 2026 vers 16 h, après la phase 1 et avant toute mesure de cette phase. herbert-cuda est mesuré sous deux formes côte à côte, le binaire nu et le candidat ; ce que désigne « herbert-cuda » dans #003 reste à décider.

## Machine et conditions

Les mêmes qu'en phase 1 : la RTX 4080, pilote 595.99.02, SM 2 805 et mémoire 11 201 MHz verrouillés, P0 pour tous les moteurs CUDA par le profil `CudaNoStablePerfLimit`. Horloges relevées à la seconde ; charge CPU relevée à chaque tour.

## Moteurs

- **herbert-cuda** `71920b9` (`~/herbert-cuda-mma`) : `nu` (aucune option) et `candidat` (`--tune gemm.auto=3 --tune gemm.layout=xor --tune prefill.block=auto`).
- **herbert-vk-bf16** `2f3ae48` (`~/herbert-vk-bf16-v1`, la build de #002) : défaut, plus `--tune host.poll=1` au décodage (#002 rapporte les deux). Pour la prédiction P9.
- **llama.cpp** `b11718`, CUDA et Vulkan.
- **vLLM** 0.31.0, toujours `--max-num-seqs 8` (le 4B l'exige).
- **SGLang** 0.5.22.
- **transformers** 5.20.1.

Chaque comparateur passe avec ses **deux meilleurs réglages** de la phase 1 ; on retient le meilleur forme par forme, comme en phase 1 :
- llama.cpp CUDA, prefill : défaut et `-ub 1024 -b 4096 -fa on` ; décodage : défaut et `-fa on` ;
- llama.cpp Vulkan, prefill : `-ub 2048 -b 4096 -fa on` et `-ub 256 -b 4096 -fa on` ; décodage : défaut et `-fa on` ;
- vLLM : `-cc.max_cudagraph_capture_size=512`, avec le backend d'attention par défaut et avec `TRITON_ATTN` ;
- SGLang : défaut et `--attention-backend triton` ;
- transformers : `sdpa` et `eager`, tous deux en cache statique avec `torch.compile`.

## Instrument 1 : le banc de chaque moteur

- herbert-cuda et herbert-vk : leur binaire de chat, comme #002 (V1). Prefill par les fixtures `pp128`, `pp512`, `pp2048`, `pp4000` (sans réflexion, `--max-tokens 1`, `--warmup 1`) ; décodage par p3 (759 tokens, réflexion, `--max-tokens 256`, `--warmup 1`). On lit la ligne `[turn]`. Les fixtures font 128, 513, 2 048 et 4 001 tokens une fois le gabarit appliqué ; le tok/s se calcule sur la longueur réelle.
- llama.cpp : `llama-bench`, comme en phase 1.
- vLLM : `vllm bench latency` ; SGLang : `sglang.benchmark.one_batch` ; transformers : `_p1_transformers.py`. Définitions de la phase 1.
- herbert-cuda (deux variantes), herbert-vk et llama.cpp alternés tour par tour, N = 5, ordre inversé un tour sur deux. vLLM, SGLang et transformers ont un démarrage long : un passage par réglage et par modèle, avec leurs répétitions internes (5, 3 et 5), placés dans le même passage.

## Instrument 2 : le client HTTP commun

`_http_client.py` (le client de V2, #002) : Messages API en flux, glouton, une requête à la fois, sans cache de préfixe, mêmes prompts (`$BENCH_DIR/prompts.json`), 128 tokens générés, 3 tours. SGLang passe par `--api openai`. Serveurs :
- `herbert-cuda-serve` nu et candidat ;
- `herbert-vk-serve` `9738be7` (celui de V2) ;
- `llama-server` CUDA et Vulkan, à leur meilleur réglage de prefill long ;
- `vllm serve`, avec les deux réglages ;
- SGLang, avec les deux réglages ;
- `transformers serve`, s'il démarre ; un échec est noté comme tel.

## Contrôle sans le profil P0

À la fin, le profil est retiré : le fichier est renommé, puis remis en place par un `trap` même en cas d'échec. Trois tours de herbert-cuda (nu et candidat) et de llama.cpp CUDA, plus un passage vLLM et un passage SGLang, au premier réglage de chacun. L'état P2 est vérifié dans les relevés d'horloge. Ce contrôle sépare ce qui vient de la P0 de ce qui vient des moteurs.

## Ce qui sort de cette phase

Pour chaque moteur, modèle, forme et instrument : la médiane, son écart et le réglage retenu. Les valeurs face aux prédictions P1–P9, sans verdict. Les logs complets sont conservés.

# Phase 5 : le prefill long, de 10 000 à 60 000 tokens (la RTX 4080)

Écrit le 19 octobre 2026 vers 16 h, après la phase 4 et avant toute mesure de cette phase et avant les prédictions qui la concernent. Objet : regarder le prefill avec beaucoup plus de tokens, 10K, 20K, 40K, « jusqu'à 60K » (le `config.json` de Qwen3.5 déclare 262 144 positions). Décisions du 19 : herbert-vk-f16 mesuré par son chat là où le prompt passe en argument, et par son serveur au client HTTP à toutes les longueurs ; les tokens au-delà de l'oracle se comparent entre moteurs. **Aucune mesure de cette phase avant que les prédictions sur le prefill long soient écrites, datées et commitées dans `predictions.md`.**

## Ce qui a été fait avant ce protocole (vérifications, pas des mesures)

- Prompts `pp10000`, `pp20000`, `pp40000`, `pp60000` (`phase5/_scripts/_build_long_prompts.py`, `~/c003/long-prompts`) : prose de Moby-Dick, longueur exacte gabarit compris (12 tokens avec `--no-system --no-think`), chacun à un endroit différent du livre.
- Un argument de ligne de commande est limité à 128 Kio sous Linux : `pp40000` (156 Kio) et `pp60000` (240 Kio) ne passent pas par `--ask`. herbert-cuda a reçu `--ask-file` (herbert-cuda-bf16 `011bfdf`, herbert-cuda-f16 `0c7d88d`, rien d'autre ne change). herbert-vk-f16 n'a pas d'équivalent et son dépôt n'est pas touché.
- Démarrage à 60 000 tokens sur la 4080, une passe par moteur, sans comparaison (`phase5/_scripts/_p5-smoke-003.sh`, `~/c003/phase5-smoke`) : herbert-cuda (bf16, f16), herbert-vk-f16 (chat à 10K et 20K, serveur à 60K), llama.cpp CUDA et Vulkan, vLLM passent en 0.8B et 4B ; SGLang manque de mémoire à 60K (`--mem-fraction-static 0.8`, 0.8B et 4B) ; transformers manque de mémoire en 4B, passe en 0.8B (5,95 Gio au plus). Les temps imprimés par ces passes ne sont ni comparés ni rapportés avant les prédictions.
- L'oracle CPU (herbert-jit-bf16 v0.1.0) réserve 8 192 positions (constante de `src/main.rs`) : il ne sert pas ici.

## Formes et moteurs

Longueurs : 10 000, 20 000, 40 000 et 60 000 tokens de prompt (herbert : gabarit compris ; comparateurs : `--input-len N` de leur banc, ids aléatoires comme aux phases 1 à 4). Un token généré pour le banc de chaque moteur ; 128 au client HTTP (le décodage à cette profondeur en sort aussi).

- **herbert-cuda-bf16** `011bfdf` (code de `638364b` plus `--ask-file`), **herbert-cuda-f16** `0c7d88d` sous ses deux formes (défaut et `--tune gemm.f16=1`), `--ctx 61440`.
- **herbert-vk-f16** `cc51363` : chat à 10K et 20K (le prompt passe en argument), serveur à toutes les longueurs au client HTTP, `--ctx 61440`.
- **llama.cpp** `b11718`, GGUF F16, CUDA et Vulkan.
- **vLLM 0.31.0**, **SGLang 0.5.22** (`SGLANG_MAMBA_CONV_DTYPE=float16`), **transformers 5.20.1**, en float16 ; `--max-model-len` / `--context-length` 61 440.

Comme en phase 4, les comparateurs tournent en f16 ; ils ne sont pas remesurés en bf16.

## Étape 5a : le meilleur réglage des comparateurs aux longueurs longues

Les réglages retenus en 4a l'ont été jusqu'à 4 000 tokens ; aux longueurs longues, le balayage est refait sur la grille de la phase 1, prefill seulement : llama.cpp (`-ub` × `-b` × `-fa`, `-r 3`), vLLM (défaut, `--enforce-eager`, `--max-num-batched-tokens` 2 048 et 16 384, backends d'attention ; `--max-num-seqs 8 -cc.max_cudagraph_capture_size=512`), SGLang (défaut, `--disable-cuda-graph`, backends, `--chunked-prefill-size` 2 048 et −1, trois mesures par forme ; plus `--mem-fraction-static` 0,7 et 0,6, ajoutés parce que 0,8 manque de mémoire à 60K), transformers (sdpa/eager, cache dynamique/statique, `torch.compile`). Un réglage qui manque de mémoire à une longueur est noté comme tel pour cette longueur. Deux réglages retenus par moteur, par le critère de 4a (`_p4a_pairs.py` : la paire la plus proche du meilleur de tous les réglages, moyenne géométrique sur les deux modèles et les quatre longueurs, cellules manquantes exclues).

**Tokens** : chaque moteur, à ses réglages de 4a, génère 64 tokens gloutons après chaque prompt long, mêmes ids de prompt pour tous (`herbert-cuda-bench --prepare`, sans réflexion) ; herbert-cuda (trois formes) par `herbert-cuda-bench`. On rapporte la première divergence de chaque paire de moteurs, sans référence CPU.

## Étape 5b : herbert contre les comparateurs aux longueurs longues

Les deux instruments de la phase 2, sans changement de méthode :
- **Banc de chaque moteur** : herbert-cuda-bf16, herbert-cuda-f16 (deux formes), herbert-vk-f16 (10K et 20K) et llama.cpp F16 CUDA et Vulkan alternés tour par tour, N = 5, ordre inversé un tour sur deux, `--warmup 1` ; vLLM, SGLang et transformers un passage par réglage et par modèle, à leurs deux réglages de 5a.
- **Client HTTP commun** (`_http_client.py --skip-blank --prompts prompts-long.json`) : cinq passages différents par longueur (comme `prompts.json`), trois tours, 128 tokens générés ; `herbert-cuda-serve` (trois formes), `herbert-vk-serve` de `cc51363`, `llama-server` F16 CUDA et Vulkan, `vllm serve`, SGLang, `transformers serve`, chacun sans cache de préfixe.

Mêmes conditions que les phases précédentes : SM 2 805 / mémoire 11 201 MHz verrouillés, P0 par le profil, horloges relevées à la seconde, charge CPU à chaque tour.

## Ce qui sort de cette phase

Pour chaque moteur, modèle, longueur et instrument : la médiane, son écart et le réglage retenu ; les ratios herbert-cuda (trois formes) / chaque moteur ; les premières divergences entre moteurs ; les manques de mémoire. Valeurs face aux prédictions sur le prefill long. Logs complets, par étape.

## Ajout du 19 octobre à 16 h 10, après P14–P15 et avant toute mesure

P15 porte sur le décodage ; ce protocole ne le mesurait qu'au client HTTP (128 tokens après le prompt). Ajouté au banc de chaque moteur de l'étape 5b, avec la même méthode qu'en phase 2 (décodage p3 à la profondeur 759) mais à la profondeur du prompt long : 128 tokens générés après chaque prompt de 10K à 60K. herbert : `--max-tokens 128` dans le même passage que le prefill, lu sur la ligne `[turn]` ; llama.cpp : `llama-bench -p 0 -n 128 -d N` ; vLLM : `--output-len 1` et `128`, décodage = 127 / (L128 − L1) ; SGLang : `one_batch --output-len 128` ; transformers : 1 et 128 tokens, même définition que vLLM.
