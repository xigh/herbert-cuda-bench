# herbert-cuda-bf16 : pourquoi le decode s'effondre en profondeur (0.8B, RTX 4080, 2026-10-20)

Après la phase 5 de #003 : chercher la cause de la chute du decode avec la profondeur. En phase 5, le decode de 128 tokens de hc-bf16 en 0.8B passait de 298,6 tok/s à 10K à 133,4 à 60K (−55 %), contre 343,7 → 270,4 pour vLLM (−21 %) et 307,4 → 239,4 pour llama.cpp CUDA (−22 %).

**Machine et conditions** : RTX 4080 (Ryzen 9 9900X, Linux), pilote 595.99.02, SM 2 805 et mémoire 11 201 MHz verrouillés, vérifiés au départ de chaque script. Profil P0. herbert-cuda-bf16 `011bfdf`, le binaire de chat de la phase 5 (sha256 `6710c86a…`, `dd1/env.txt`). Modèle Qwen3.5-0.8B, prompts longs de la phase 5 (`~/c003/long-prompts`, Moby-Dick), `--ctx 61440 --no-system --no-think --warmup 1`. Les deux services d'inférence résidents de la machine sont restés en marche pendant toutes les mesures : `resident-gpu` (6,8 Go sur la 4080) et `resident-cpu` (sur le CPU). Le 4B n'est pas mesuré : d'après une estimation (9,3 Go de poids, plus le KV de 61 440 positions, pour 9,2 Go libres), il ne tient pas à côté du service résident. Il n'a pas été essayé.

## Les passages (tous gardés)

- `dd0-try-profile-10k.log` : premier essai de `--profile` à 10K, pour lire le format de sortie.
- `dd1/` (`_scripts/_dd1-003.sh`, `dd1.out`), de 21 h 38 à 21 h 41. Rien d'autre que le service résident sur la carte : la mémoire GPU n'a pas dépassé 9,7 Go (`dd1/clocks.csv`).
  - A : `--profile` (lancements un par un avec events, pas de graphe) à 10K, 20K, 40K et 60K, 64 tokens.
  - B : `--tune sdpa.split` à 8 (le défaut), 16, 32, 64, 128 et 256, à 10K et 60K, 128 tokens, graphe actif, alterné, N = 3.
  - C : `--profile` à 60K pour chaque valeur de `sdpa.split`.
- `dd2/` (`_dd2-003.sh`), de 21 h 43 à 21 h 46 : le même balayage sur p3 (759 tokens, réflexion, 256 générés, la forme de decode de la phase 2 et de #002), à 20K et à 40K. **Faussé, laissé hors du résumé.** À 21 h 45, un autre travail a lancé sur la 4080 les tests de `other-tests` (`--test-threads=1`, jusqu'à 4,4 Go, 14 cœurs). La mémoire GPU est montée à 15,9 Go, six lancements sont morts après le chargement, et un decode p3 à 8 `splits` est tombé à 165,9 tok/s.
- `dd2b/` (`_dd2b-003.sh`), de 21 h 46 à 21 h 50 : dd2 refait. Avant chaque lancement, le script attend qu'aucun autre processus que le service ne tienne la carte (`waits.txt`) et note les processus présents (`apps.txt`). Le filtre reconnaissait le service par son nom de binaire : il a laissé passer un second `resident-gpu-serve` lancé par l'autre travail (6,7 Go). Ces lancements ont échoué à 20K et 40K, ou ont donné un decode p3 de 200 à 224 tok/s. Le résumé écarte les 14 lancements où `apps.txt` montre un autre processus que le service (PID 1657107). Il en reste au moins un par cellule ; n est donné partout.
  - Un lancement retenu est probablement faussé lui aussi : `r1-s16-p3`, 317,0 tok/s contre 403,7 et 403,8 aux deux autres tours, pendant que les tests de l'autre travail tournaient. Il est dans l'écart de la cellule (21,5 %), pas dans la médiane.

`summary.txt` est la sortie de `_scripts/_dd_summary.py` sur ce dossier.

## A. Le pas de decode par profondeur (`sdpa.split=8`, `--profile`)

| Profondeur | Pas (GPU, médiane) | `sdpa_split` | `sdpa_combine` | Reste du pas | K et V lus une fois par pas | Débit qu'implique `sdpa_split` |
|---|---|---|---|---|---|---|
| 10K | 3 948,5 µs | 873,5 µs (22,1 %) | 18,4 µs | 3 056,6 µs | 123 Mo | 141 Go/s |
| 20K | 4 773,9 µs | 1 704,7 µs (35,7 %) | 18,8 µs | 3 050,4 µs | 246 Mo | 144 Go/s |
| 40K | 6 439,8 µs | 3 378,4 µs (52,5 %) | 19,7 µs | 3 041,7 µs | 492 Mo | 146 Go/s |
| 60K | 8 115,2 µs | 5 042,7 µs (62,1 %) | 19,9 µs | 3 052,6 µs | 738 Mo | 146 Go/s |

- De 10K à 60K, le pas prend +4 166,7 µs, dont +4 169,2 µs pour `sdpa_split`. Le reste du pas ne bouge pas (3 042 à 3 057 µs).
- Le volume compte 2 têtes KV × 256 × (K et V) × 2 octets × 6 couches d'attention × la position médiane du decode (prompt + 32). Le noyau lit ce volume à un débit constant, autour de 145 Go/s, soit 20 % des 716,8 Go/s de la carte si K et V n'étaient lus qu'une fois.

**Ce que fait le noyau, d'après le code** (`src/cuda/decode.rs:264`, `sdpa_split`) :
- La grille fait têtes × `splits` = 8 × 8 = 64 blocs de 256 threads, pour 76 SM. Chaque bloc prend un huitième des positions, soit 7 500 à 60K.
- Il les parcourt en trois boucles : les scores, un warp par position ; les exponentielles ; puis la somme sur V, où chaque thread porte une dimension et fait, à chaque position, un chargement de 2 octets et un `ffma`.
- Chaque bloc lit les K et V de sa tête KV : les 4 têtes de requête d'un même groupe GQA les lisent chacune.
- Le nombre de blocs ne dépend pas de la profondeur, seule la longueur des boucles en dépend. C'est cohérent avec un temps linéaire en profondeur à débit fixe. C'est une lecture du code, pas une mesure par compteurs : Nsight Compute n'a pas tourné.

## B. Le nombre de `splits` (graphe actif, decode en tok/s, médiane, n)

| Forme | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| p3 (759 + 255) | 401,0 (3) | 403,7 (3) | 403,8 (2) | 402,5 (2) | 400,0 (1) | |
| 10K | 298,7 (3) | 338,7 (3) | 361,3 (3) | 359,7 (3) | 362,2 (3) | 352,6 (3) |
| 20K | 240,1 (3) | 295,7 (2) | 329,1 (2) | 330,0 (2) | 335,8 (1) | |
| 40K | 171,8 (3) | 235,1 (2) | 284,6 (2) | 275,7 (2) | 290,0 (1) | |
| 60K | 133,5 (3) | 194,5 (3) | 246,8 (3) | 230,6 (3) | 251,8 (3) | 238,7 (3) |

- Rapport au défaut (8 `splits`), avec 128 : p3 ×0,998, 10K ×1,213, 20K ×1,399, 40K ×1,688, 60K ×1,886.
- De 10K à 60K, le decode perd 55 % avec 8 `splits` et 30 % avec 128.

**60K, `--profile` par valeur de `splits`** (C) :

| `splits` | Pas | `sdpa_split` | Débit si K et V lus une fois | `sdpa_combine` |
|---|---|---|---|---|
| 8 | 8 113,2 µs | 5 045,7 µs | 146 Go/s | 19,5 µs |
| 16 | 5 761,0 µs | 2 684,2 µs | 275 Go/s | 20,5 µs |
| 32 | 4 659,2 µs | 1 572,9 µs | 469 Go/s | 22,9 µs |
| 64 | 4 939,8 µs | 1 857,5 µs | 397 Go/s | 27,0 µs |
| 128 | 4 579,4 µs | 1 480,7 µs | 498 Go/s | 48,5 µs |
| 256 | 4 803,6 µs | 1 648,6 µs | 447 Go/s | 95,9 µs |

Le combinateur grandit avec le nombre de `splits` (19,5 → 95,9 µs). 64 est plus lent que 32 et 128 au profil à 60K et au banc à 40K et 60K ; à 20K, il est à égalité avec 32.

**Face aux comparateurs de la phase 5.** Ce sont deux passages différents, pas une mesure alternée : ces rapports ne la remplacent pas.
- vLLM : 343,7 tok/s à 10K et 270,4 à 60K.
- hc-bf16 à 128 `splits` : 362,2 et 251,8.
- Le rapport hc-bf16 / vLLM passerait donc de ×0,869 à ×1,054 à 10K, et de ×0,494 à ×0,931 à 60K.

**Tokens.** Le nombre de `splits` change l'ordre des sommes, donc la suite générée :
- à 60K, 16, 32, 64 et 128 donnent la même suite, qui s'écarte de celle à 8 au token 17 ; 256 s'en écarte au token 31 ;
- à 10K, les écarts tombent aux tokens 10, 12, 28 et 98 selon la valeur ;
- sur p3, 128 donne la même suite que 8 ; 16 et 32 s'en écartent au token 144, 64 au token 172.

Chaque valeur rend la même suite à chacun de ses tours. Aucune référence CPU n'existe au-delà de 8 192 positions. Sur p3, l'oracle reste à rejouer contre chaque valeur.

## Ce qui n'est pas fait

- Le 4B : il faut libérer la 4080 du service résident.
- Les compteurs matériels (Nsight Compute) sur `sdpa_split`, pour confirmer que le noyau est limité par la latence et non par la bande passante.
- Une version du noyau qui partage K et V entre les 4 têtes d'un groupe GQA, et qui choisit le nombre de `splits` selon la profondeur. Rien n'est écrit, et aucune valeur n'a changé dans le moteur.
- La même question pour l'attention du prefill : à 60K, `sdpa_flash` prend 61,7 % du temps du prefill (`dd1/a/profile-pp60000.log`). C'est la piste du prefill au-delà de 20K.
