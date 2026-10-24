# 2026-10-21 · herbert-cuda-bf16, l'occupation de l'attention du prefill : 32 lignes de requête par bloc (`sdpa.rows=32`)

Objet : travailler sur l'occupation du prefill, après la mesure alternée. RTX 4080 verrouillée à 2 805 / 11 201 MHz, P0, le 2026-10-21 de 03 h 26 à 03 h 48. Le service d'inférence résident a été arrêté pendant le banc, puis relancé par un piège ; il était actif à 03 h 48. Le GPU était vide à chaque lancement : aucune attente n'a eu lieu.

## Le point de départ

`sdpa_flash` traite 16 lignes de requête par bloc, avec 4 warps, une par tête du groupe GQA. Le bloc prend 67,6 Ko de mémoire partagée (Q des 4 warps, une tuile K et une tuile V de 32 clés, lignes bourrées de 16 octets) et 207 registres par thread. Il ne tient qu'un bloc par SM, soit 4 warps : l'occupation théorique est de 8,33 %. Nsight Compute estimait un gain possible de 71,5 %.

## Ce qui a été écrit (herbert-cuda-bf16 `8db5b92`, derrière `--tune sdpa.rows=32`, défaut 16)

- **Le bloc** couvre deux groupes de 16 lignes, avec 8 warps qui partagent chaque tuile K et V : la warp w prend la tête w mod g et le groupe de lignes w / g.
- **La mémoire partagée** : 8 × 16 × 528 + 2 × 32 × 528 = 101 376 octets, le maximum d'un bloc sur sm_89. La configuration est de 102,40 Ko, dont 1,02 Ko pour le pilote.
- **Les clés parcourues** : chaque warp va jusqu'à la dernière ligne du bloc. Au-delà de ses propres lignes, une tuile est entièrement masquée : ses exponentielles valent zéro et sa correction 1. C'est déjà ce qui arrive au défaut quand les lignes d'un bloc s'arrêtent avant la fin d'une tuile. Les calculs de chaque ligne sont donc les mêmes.
- **Le PTX par défaut** est inchangé : les 436 modules de `herbert-cuda-probe --emit` sont identiques à ceux de `696b75d`, et 4 nouveaux s'y ajoutent.
- **`ptxas -v` (sm_89)**, dans `ptx/ptxas-q32.txt`, sans aucun débordement :
  - `q32` : 207 registres ;
  - `pipe_q32` : 222 ;
  - `pipe` : 255.
- Le diff construit et mesuré (`_scripts/_occ.diff`) est exactement celui du commit.

## Ce qui a tourné (`_scripts/_occ1-003.sh`, binaire `~/herbert-cuda-bf16-occ`, sha256 dans `occ1/env.txt`)

Quatre variantes : default, `rows32`, `pipe` (`sdpa.pipe=1`) et `pipe-rows32`.

- **A. Tokens** : `compare.sh` sur p1–p4 et pp128–pp4000, contre les journaux de l'oracle de `runs/plan-final-oracle/{0.8B,4B}` (copiés, pas relancés).
- **B. Vitesse** : prefill alterné, N = 3, ordre inversé un round sur deux. Longueurs : pp4000 (`--ask`), puis 10K, 20K, 40K et 60K (`--ask-file`, `--ctx 61440`). Sans réflexion, 16 tokens générés, `--warmup 1`.
- **C. `--profile` à 60K.**
- **D. Nsight Compute** (`--set full`, `--clock-control none`) : 2 lancements de `sdpa_flash` sur le dernier bloc de prefill à 60K, pour `rows32` et `pipe-rows32`, en 0.8B et en 4B. Le défaut et `pipe` ont été mesurés dans dd4 et dd7 (dossier `2026-10-20-cuda-bf16-attention-gqa-pipe`).

La première chaîne a échoué au build : son shell détaché n'avait pas `~/.cargo/env` (`chain-occ1-failed-build.out`, `build-occ-failed.log`). Elle a été relancée avec, et rien d'autre n'a changé.

## Tokens

- **Contre l'oracle**, les quatre variantes ont les mêmes écarts que le défaut :
  - 0.8B : p2 et p4 s'écartent au token 8, les autres sont identiques ;
  - 4B : p2 s'écarte au token 32, p4 au token 105, pp512 au token 0, les autres sont identiques.
- **Entre variantes**, les lignes `[ids]` sont identiques, pour les 4 variantes :
  - les 20 prompts de compare ;
  - les 10 formes du banc B (5 longueurs × 2 modèles), à chacun de leurs 3 rounds.

## Vitesse du prefill (tok/s, médiane de 3 rounds, `summary.txt`)

| 0.8B | pp4000 | pp10000 | pp20000 | pp40000 | pp60000 |
|---|---|---|---|---|---|
| default | 60 149 | 53 190 | 44 576 | 33 123 | 26 556 |
| rows32 | 60 697 | 53 602 | 46 494 | 34 930 | 27 817 |
| pipe | 61 422 | 55 758 | 48 378 | 37 648 | 31 036 |
| pipe-rows32 | 61 465 | 54 982 | 48 550 | 37 483 | 30 407 |

| 4B | pp4000 | pp10000 | pp20000 | pp40000 | pp60000 |
|---|---|---|---|---|---|
| default | 11 136 | 10 509 | 9 474 | 7 972 | 6 856 |
| rows32 | 11 213 | 10 673 | 9 691 | 8 219 | 7 109 |
| pipe | 11 312 | 10 782 | 9 911 | 8 618 | 7 618 |
| pipe-rows32 | 11 282 | 10 791 | 9 926 | 8 582 | 7 548 |

Rapports au défaut (médiane des rapports par round), de pp4000 à pp60000 :
- 0.8B :
  - `rows32` : ×1,011, ×1,008, ×1,042, ×1,054, ×1,047 ;
  - `pipe` : ×1,023, ×1,051, ×1,085, ×1,136, ×1,169 ;
  - `pipe-rows32` : ×1,022, ×1,034, ×1,088, ×1,130, ×1,145.
- 4B :
  - `rows32` : ×1,007, ×1,016, ×1,023, ×1,031, ×1,037 ;
  - `pipe` : ×1,013, ×1,027, ×1,046, ×1,081, ×1,111 ;
  - `pipe-rows32` : ×1,012, ×1,027, ×1,048, ×1,076, ×1,101.

Écarts entre les rounds : au plus 0,9 %.

## `--profile` à 60K (temps GPU par bloc de prefill, médiane ; `sdpa_flash` par bloc et sa part)

- 0.8B (15 blocs de 4 096) :
  - default : 155,96 ms, dont 96,08 ms (61,6 %) ;
  - rows32 : 148,09 ms, dont 88,57 ms (59,8 %) ;
  - pipe : 133,27 ms, dont 72,34 ms (54,3 %) ;
  - pipe-rows32 : 136,02 ms, dont 76,08 ms (55,9 %).
- 4B (30 blocs de 2 048) :
  - default : 296,98 ms, dont 126,96 ms (42,8 %) ;
  - rows32 : 286,93 ms, dont 117,33 ms (40,9 %) ;
  - pipe : 269,07 ms, dont 94,90 ms (35,3 %) ;
  - pipe-rows32 : 270,80 ms, dont 99,88 ms (36,9 %).

## Nsight Compute, dernier bloc à 60K

- **Occupation** :
  - `rows32` et `pipe-rows32` : théorique 16,67 %, réalisée 16,66 à 16,68 %, soit 8 warps par SM, contre 4 au défaut.
  - Limite d'un bloc par SM, à la fois par la mémoire partagée et par les registres.
- **Durée**, 0.8B :

  | Variante | Durée | Sans warp prêt |
  |---|---|---|
  | default (dd4) | 21,55 ms | 89,7 % |
  | pipe (dd7) | 15,63 ms | 87,1 % |
  | rows32 | 21,15 ms | 88,8 % |
  | pipe-rows32 | 17,70 ms | 87,6 % |

- **Durée**, 4B : `rows32` 7,26 ms et `pipe-rows32` 6,08 ms, en une seule vague (76 blocs).
- **Attentes** : 0.8B `rows32`, 17,9 cycles par instruction émise, dont 10,9 à attendre que le pipeline d'exécution se libère (« execution pipe ») ; `pipe-rows32`, 16,1 cycles, dont 11,9.
- **Tensor cores** : le pipeline le plus chargé, à 38,0 % (rows32) et 45,4 % (pipe-rows32), ncu rapportant ces taux à ses propres pics.
- **Bancs de la mémoire partagée** : 13 % de « wavefronts » en trop dans les deux cas.

## Part du plafond des tensor cores (`_scripts/_tensor_share.py`, `tensor-share.txt`)

Le travail mma exécuté par `sdpa_flash` sur le dernier bloc à 60K, compté depuis le noyau, tuiles masquées comprises : 1,277 TFLOP en 0.8B, 0,595 en 4B. Divisé par la durée mesurée par ncu, puis rapporté au plafond mesuré de `mma.sync` bf16 → f32 sur la 4080, 108,5 TFLOPS (mesuré par la sonde, horloges verrouillées ; ncu tourne ici sans contrôle des horloges) :

| Variante | 0.8B | 4B |
|---|---|---|
| default | 59,3 TFLOPS (55 %) | — |
| pipe | 81,7 (75 %) | — |
| rows32 | 60,4 (56 %) | 81,9 (76 %) |
| pipe-rows32 | 72,1 (66 %) | 97,8 (90 %) |

## Fichiers

- `occ1/` :
  - `compare-{0.8B,4B}/` et `compare-*.out` : tokens et journaux de compare ;
  - `b/` : le banc alterné ;
  - `c/` : les profils ;
  - `ncu/` : rapports `.ncu-rep`, page de détails, journaux ;
  - `clocks.csv`, `launches.txt`, `env.txt`, `end.txt`.
- `ptx/` : les quatre modules `sdpa_flash` comparés et la sortie de `ptxas -v`.
- `summary.txt` : la sortie de `_scripts/_occ1_summary.py occ1`.
- `_scripts/` : le banc, la chaîne, les résumés et le diff.
- `chain-occ1*.out`, `build-occ*.log` : la chaîne et les builds, dont celui qui a échoué.
