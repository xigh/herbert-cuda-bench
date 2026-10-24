# herbert-cuda-bf16 : les deux noyaux d'attention corrigés, decode en une passe par groupe GQA (`sdpa.gqa=1`) et prefill à copies recouvertes (`sdpa.pipe=1`) (2026-10-20)

Suite du diagnostic du decode en profondeur, la même nuit. Trois tâches : écrire le noyau du decode, mesurer le 4B, et regarder le prefill long, en s'arrêtant dès qu'une correction aide. Les deux corrections sont dans herbert-cuda-bf16, derrière deux interrupteurs, désactivés par défaut. Le PTX par défaut est identique, octet pour octet, à celui de `011bfdf`. Le choix du défaut est fait plus tard (dossier `2026-10-21-cuda-bf16-defaut-attention`).

**Machine et conditions** : RTX 4080 (Ryzen 9 9900X, Linux), pilote 595.99.02, SM 2 805 et mémoire 11 201 MHz verrouillés, vérifiés au départ de chaque script, P0. Qwen3.5-0.8B et 4B.

**Le service d'inférence résident de la machine** (`resident-gpu.service`) a été arrêté trois fois pour le 4B, puis relancé à chaque fois par un `trap`. Il a été vérifié actif après chaque relance :
- dd3, de 22 h 03 à 22 h 22 ;
- dd6, de 22 h 29 à 22 h 44 ;
- dd8, de 22 h 49 à 23 h 13.

**Le serveur de débit d'un autre travail**, qui tournait sur la 4080, a été arrêté à 22 h 02 ; sa ligne de commande a été relevée avant l'arrêt.

**Le contrôle de la carte.** Chaque lancement attend que la carte ne porte que le service, identifié par son PID, ou rien du tout quand le service est arrêté. Les attentes sont dans `waits.txt` et les lancements dans `launches.txt`. Aucune attente n'a eu lieu, de dd3 à dd8.

## Les commits (herbert-cuda-bf16, non poussés)

- `bc8e601` : `decode::sdpa_gqa`, derrière `--tune sdpa.gqa=1`.
  - Un bloc par tête KV et par tranche de positions, contre un bloc par tête de requête auparavant. K et V ne sont lus qu'une fois pour les 4 têtes du groupe GQA.
  - Une seule passe, avec le softmax en ligne : chaque warp tient le maximum courant, la somme et Σ e·v de ses 8 dimensions.
  - Les 8 warps du bloc se partagent les positions, avec un chargement de 16 octets de K et un de V par lane et par position.
  - Les résultats partiels gardent le format de `sdpa_split`, donc `sdpa_combine` est inchangé. `sdpa.split` donne le nombre de tranches par tête KV.
  - `ptxas` sm_89 : 104 registres, aucun débordement, 33 Kio de mémoire partagée.
- `5ec3063` : `prefill::sdpa_flash` avec `pipe`, derrière `--tune sdpa.pipe=1`.
  - La copie de la tuile K suivante part dès que S est calculé, celle de la tuile V suivante dès que P·V est fini. Chacune est attendue par `cp.async.wait_group 1` juste avant son usage.
  - Pas de mémoire partagée en plus. Mêmes calculs, dans le même ordre.
  - `ptxas` : 255 registres, aucun débordement.
- `73b5a70` : correction de `5ec3063`, dont le message affirmait à tort que le PTX par défaut était inchangé. Le calcul du début de la tuile suivante était fait même sans `pipe`, ce qui ajoutait une instruction au `sdpa_flash` par défaut. dd7 l'a vu (`chain7.out` : PTX DIFFERS) ; ses mesures du « défaut » portent donc cette instruction en trop.
  - Après correction, `chain8.out` compare les modules émis par `herbert-cuda-probe --emit` aux mêmes modules de `011bfdf` : 432 modules, 0 différence, plus les 4 nouveaux noyaux (`ptx-new/`).
  - La même comparaison, faite sur le Mac, donne le même résultat.

## Les passages (tous gardés)

- **`dd3/`, 4B, ancien noyau** (`_dd3-003.sh`) : `--profile` de 10K à 60K, puis `sdpa.split` à 8, 16, 32 et 128 sur p3, 10K, 20K, 40K et 60K (alterné, N = 3), puis un profil par valeur à 60K.
- **`dd4/`, Nsight Compute 2025.1.1** (`--set full`), 0.8B à 60K :
  - `sdpa_split` à 8 et à 128 `splits`, sur les deux premiers lancements du decode ;
  - `sdpa_flash` sur deux lancements du dernier bloc de prefill.
  - Les rapports `.ncu-rep` sont gardés, avec leur texte (`--page details`).
- **`dd5/`, 0.8B, `sdpa.gqa`** (build `bc8e601`, `chain.out`) :
  - tokens p1 à p4 contre les journaux de l'oracle CPU de `runs/plan-final-oracle/0.8B`, copiés et non relancés ;
  - banc alterné de l'ancien noyau (8 et 128 `splits`) contre gqa à 16, 32, 64 et 128, sur p3, 10K et 60K, N = 3 ;
  - profils à 60K.
- **`dd6/`, la même chose sur le 4B**, avec l'ancien noyau à 8 et 16 `splits` contre gqa à 32, 64 et 128.
- **`dd7/`, 0.8B, `sdpa.pipe`** (build `5ec3063`, `chain7.out`) :
  - tokens p1 à p4 et pp128 à pp4000 contre l'oracle ;
  - prefill à 10K, 20K, 40K et 60K, alterné, N = 3 ;
  - profils à 60K ;
  - Nsight Compute sur `sdpa_flash` avec `pipe`.
- **`dd8/`, la confirmation** (build `73b5a70`, `chain8.out`) :
  - 0.8B puis 4B, quatre variantes : défaut, gqa (`sdpa.gqa=1 sdpa.split=64`), pipe (`sdpa.pipe=1`) et les deux ;
  - tokens p1 à p4 et pp128 à pp4000 contre l'oracle (défaut et les deux) ;
  - banc alterné N = 3 sur p3, pp4000, 10K, 20K, 40K et 60K : prefill et decode lus sur la même ligne `[turn]`, 128 tokens générés aux longueurs longues ;
  - profils à 60K.

`summary.txt` est la sortie de `_scripts/_dd_summary.py`, sur l'ensemble des passages du diagnostic et de celui-ci. Les données brutes de dd1, dd2 et dd2b sont dans le dossier `2026-10-20-cuda-bf16-decode-profondeur`.

## Nsight Compute (dd4, 0.8B, 60K)

| Noyau | Grille | Durée | DRAM | Occupation théorique / réalisée | Cycles sans warp prêt |
|---|---|---|---|---|---|
| `sdpa_split`, 8 `splits` | 64 × 256 | 838 µs | 149 Go/s (20,8 %) | 100 % / 16,6 % | 89,9 % |
| `sdpa_split`, 128 `splits` | 1 024 × 256 | 243 µs | 572 Go/s (79,9 %) | 100 % / 89,1 % | 65,8 % |
| `sdpa_flash`, dernier bloc | 166 × 2 × 128 | 21,55 ms | 2,0 % (L2 28,5 %) | 8,33 % / 8,33 % | 89,7 % |
| `sdpa_flash` avec `pipe` (dd7) | 166 × 2 × 128 | 15,63 ms | 2,8 % (L2 39,3 %) | 8,33 % / 8,33 % | 87,1 % |

- **Le decode** (`sdpa_split` à 8 `splits`) est limité par la latence : 0,14 vague par SM, et 2 warps actifs par ordonnanceur. À 128 `splits`, il lit environ 137 Mo par lancement pour 123 Mo de K et V par couche : le L2 évite déjà les relectures des 4 têtes d'un groupe.
- **Le prefill** (`sdpa_flash`) tient un seul bloc de 4 warps par SM : 67,6 Kio de mémoire partagée, 207 registres. Le pipeline des tensor cores est occupé à 31 %, et 43 % avec `pipe`. ncu estime à 71,5 % le gain possible en augmentant l'occupation, ce que `pipe` ne fait pas.

## La confirmation (dd8), tok/s, médiane de N = 3, rapport au défaut

**0.8B, prefill :**

| | p3 | pp4000 | 10K | 20K | 40K | 60K |
|---|---|---|---|---|---|---|
| défaut | 52 460 | 60 057 | 52 912 | 44 562 | 33 093 | 26 535 |
| les deux | 52 808 (×1,007) | 61 567 (×1,025) | 55 725 (×1,053) | 48 505 (×1,088) | 37 643 (×1,137) | 31 025 (×1,169) |

**0.8B, decode :**

| | p3 | pp4000 | 10K | 20K | 40K | 60K |
|---|---|---|---|---|---|---|
| défaut | 401,0 | 351,4 | 298,5 | 240,2 | 171,8 | 133,4 |
| les deux | 404,4 (×1,008) | 387,3 (×1,102) | 369,5 (×1,238) | 347,8 (×1,448) | 308,8 (×1,797) | 278,1 (×2,084) |

**4B, prefill :**

| | p3 | pp4000 | 10K | 20K | 40K | 60K |
|---|---|---|---|---|---|---|
| défaut | 10 865 | 11 157 | 10 516 | 9 475 | 7 964 | 6 856 |
| les deux | 11 039 (×1,016) | 11 304 (×1,013) | 10 800 (×1,027) | 9 915 (×1,046) | 8 614 (×1,082) | 7 614 (×1,111) |

**4B, decode :**

| | p3 | pp4000 | 10K | 20K | 40K | 60K |
|---|---|---|---|---|---|---|
| défaut | 78,4 | 76,5 | 72,6 | 67,1 | 57,8 | 50,9 |
| les deux | 78,7 (×1,003) | 78,0 (×1,020) | 76,2 (×1,051) | 73,9 (×1,100) | 69,0 (×1,193) | 64,7 (×1,272) |

- **Chaque correction n'agit que sur sa phase.** gqa seul donne le même prefill que le défaut (×0,994 à ×1,006), et le même decode que « les deux ». pipe seul donne le même decode que le défaut (×0,999 à ×1,000), et le même prefill que « les deux ». Le détail par variante est dans `summary.txt`.
- **Les écarts** (max − min) / médiane vont de 0,0 à 2,3 %.
- **Au profil à 60K** :
  - 0.8B : le bloc de prefill passe de 155 893 à 133 162 µs (`sdpa_flash` de 61,6 % à 54,3 %), et le pas de decode de 8 118 à 4 202 µs (attention de 62,1 % à 26,8 %, 1 125 µs, environ 655 Go/s) ;
  - 4B : le bloc de prefill passe de 297 258 à 269 321 µs, et le pas de decode de 20 507 à 16 238 µs (attention à 2 954 µs, environ 665 Go/s).

**Face aux comparateurs de la phase 5.** Ce sont deux passages différents, pas une mesure alternée, et ces rapports ne la remplacent pas.

| Modèle | Phase | Longueur | hc-bf16 « les deux » | Comparateur | Rapport |
|---|---|---|---|---|---|
| 0.8B | prefill | 60K | 31 025 | SGLang 31 767 | ×0,977 |
| 0.8B | prefill | 60K | 31 025 | vLLM 30 738 | ×1,009 |
| 0.8B | prefill | 60K | 31 025 | llama.cpp CUDA 29 757 | ×1,043 |
| 0.8B | decode | 60K | 278,1 | vLLM 270,4 | ×1,028 |
| 0.8B | decode | 10K | 369,5 | vLLM 343,7 | ×1,075 |
| 4B | prefill | 60K | 7 614 | llama.cpp CUDA 8 041 | ×0,947 |
| 4B | prefill | 60K | 7 614 | vLLM 7 787 | ×0,978 |
| 4B | decode | 60K | 64,7 | vLLM 63,4 | ×1,020 |

En phase 5, ces rapports étaient : prefill 0.8B contre SGLang ×0,839, decode 0.8B contre vLLM ×0,494, prefill 4B contre llama.cpp ×0,859, decode 4B contre vLLM ×0,803.

## Tokens

**Contre l'oracle CPU (dd8)** :
- **0.8B, « les deux »** :
  - p1 et pp128 à pp4000 : identiques ;
  - p2 et p4 : écart au token 8, comme le défaut ;
  - p3 : écart au token 172, alors que le défaut est identique. gqa à 64 `splits` le faisait déjà en dd5 ; gqa à 16 et 32 sont identiques sur p3.
- **4B, « les deux »** :
  - p1, p3 et pp128 à pp4000 : identiques, sauf pp512, où défaut et « les deux » s'écartent au token 0 (écart déjà connu) ;
  - p2 : identique aux 177 tokens, alors que le défaut s'écarte au token 32 ;
  - p4 : écart au token 105, comme le défaut.

**Aux longueurs longues, sans référence CPU** :
- `pipe` rend exactement la même suite que le défaut, en dd7 et en dd8 : les calculs sont les mêmes.
- `gqa` change l'ordre des sommes de l'attention du decode. À 60K, il s'écarte du défaut au token 17 en 0.8B, et au token 23 en 4B (28 pour gqa32). Chaque variante rend la même suite à ses trois tours.

## Ce qui n'est pas fait

- **Le défaut n'est pas changé.** Les deux interrupteurs restent désactivés : le choix du défaut et du nombre de `splits` est fait à part.
- **Le nombre de `splits` est fixe.** Rien n'est écrit pour le choisir selon la profondeur, puisque la grille du graphe CUDA est figée à la capture. Les mesures donnent 64 ; 32 et 128 sont à moins de 2 %.
- **L'occupation du prefill.** ncu estime à 71,5 % le gain possible. Plusieurs pistes sont écrites mais non tentées : Q en registres et une tête découpée en deux moitiés de 128 dimensions, ou des tuiles K et V à disposition XOR pour tenir deux étages dans 99 Kio.
- **Rien n'est remesuré face aux comparateurs dans le même passage**, ni au client HTTP commun.
