# Ablation de herbert-cuda dans le moteur entier : un réglage CUDA retiré à la fois (2026-10-18)

Suite des vérifications du moteur : ce que chaque instruction ou réglage propre à CUDA pèse de bout en bout, prefill et décodage, dans le moteur complet. C'est herbert-cuda contre lui-même, sans autre moteur.

Code : `herbert-cuda-bf16` `f1c5a9f` (`herbert-cuda-chat`, sha256 dans `env.txt`), construit dans `~/herbert-cuda-mma`. Script `_ablation-003.sh`, résumé `_ablation_summary.py` → `summary.txt`.

Machine : RTX 4080 (Ryzen 9 9900X, Linux), pilote 595.99.02, SM 2 805 MHz. Pendant les mesures, la mémoire était à 10 801 MHz en P2 (1 698 échantillons sur 1 796, `clocks.csv`) ; CPU au repos. Du 2026-10-18 06:29 à 06:59.

Base : le candidat, `--tune gemm.auto=3 --tune gemm.layout=xor --tune prefill.block=auto`. Variantes :
- `noldsm` : + `gemm.ldmatrix=0`, fragments du GEMM par `ld.shared.b32` ;
- `sync` : + `gemm.copy=sync`, copies du GEMM par registres au lieu de `cp.async` ;
- `noldsm-sync` : les deux ;
- `pad` : sans `gemm.layout=xor` ;
- `graph0` : + `graph=0`, le pas de décodage lancé noyau par noyau au lieu du graphe CUDA ;
- `default` : le binaire sans aucune option.

Formes : prefill des fixtures `pp128` à `pp4000` (128 à 4 001 tokens, un token généré) ; p3 (759 tokens, 255 générés, réflexion ouverte). Protocole : `--warmup 1`, un processus par mesure, N = 5 tours, variantes alternées et ordre inversé un tour sur deux, 490 exécutions sans échec. Écart (max − min) / médiane : 0,33 % en médiane, 1,9 % au plus.

## Résultats (médianes, tok/s, et rapport à la base)

0.8B :

| variante | 128 | 256 | 512 | 1 024 | 2 048 | 4 000 | décodage p3 |
|---|---|---|---|---|---|---|---|
| base | 27 627 | 45 085 | 49 551 | 60 400 | 62 714 | 60 483 | 388,2 |
| sans `ldmatrix` | ×0,998 | ×0,974 | ×0,943 | ×0,957 | ×0,971 | ×0,972 | ×1,000 |
| sans `cp.async` | ×0,869 | ×0,862 | ×0,871 | ×0,954 | ×0,951 | ×0,953 | ×1,000 |
| sans les deux | ×0,836 | ×0,828 | ×0,843 | ×0,920 | ×0,930 | ×0,938 | ×1,000 |
| sans XOR | ×0,959 | ×0,872 | ×0,975 | ×0,959 | ×0,956 | ×0,941 | ×1,000 |
| sans graphe | ×0,998 | ×1,000 | ×1,000 | ×1,000 | ×0,999 | ×1,000 | **×0,953** |
| binaire nu | ×0,924 | ×0,883 | ×0,934 | ×0,830 | ×0,795 | ×0,791 | ×1,002 |

4B :

| variante | 128 | 256 | 512 | 1 024 | 2 048 | 4 000 | décodage p3 |
|---|---|---|---|---|---|---|---|
| base | 6 385 | 9 313 | 9 667 | 11 303 | 11 677 | 11 302 | 75,7 |
| sans `ldmatrix` | ×0,978 | ×0,966 | ×0,954 | ×0,961 | ×0,961 | ×0,962 | ×1,000 |
| sans `cp.async` | ×0,938 | ×0,899 | ×0,888 | ×0,909 | ×0,911 | ×0,913 | ×1,000 |
| sans les deux | ×0,882 | ×0,839 | ×0,877 | ×0,892 | ×0,894 | ×0,896 | ×1,000 |
| sans XOR | ×0,958 | ×0,883 | ×0,952 | ×0,930 | ×0,915 | ×0,922 | ×1,000 |
| sans graphe | ×1,000 | ×0,997 | ×1,000 | ×1,001 | ×0,999 | ×0,999 | **×0,989** |
| binaire nu | ×0,725 | ×0,799 | ×0,804 | ×0,759 | ×0,731 | ×0,736 | ×1,001 |

Ids de sortie identiques à la base dans chaque tour, pour toutes les variantes et toutes les formes (`summary.txt`).

Lecture :
- Au prefill, sans les deux instructions, le moteur perd 6 à 17 % ; plus en court sur le 0.8B (−16 %), régulièrement 10 à 16 % sur le 4B. `cp.async` pèse plus que `ldmatrix` (−5 à −14 % contre −2 à −6 %).
- Le décodage n'est touché par aucun des réglages du GEMM. Il ne passe que par le matvec et ses noyaux. Seul le graphe CUDA compte : −4,7 % sans lui sur le 0.8B, −1,1 % sur le 4B.
- La disposition XOR vaut 3 à 13 %.
- Le binaire sans options est 8 à 27 % derrière le candidat au prefill : c'est l'écart déjà mesuré lors des vérifications du moteur.

## Contenu

- `_ablation-003.sh`, `_ablation_summary.py`, `summary.txt` ;
- `env.txt` (binaire, variantes), `run.out`, `end.txt`, `clocks.csv` ;
- `runs/r<tour>-<modèle>-<variante>-<forme>.log` : les 490 sorties complètes (`[tuning]`, `[turn]`, `[ids]`).
