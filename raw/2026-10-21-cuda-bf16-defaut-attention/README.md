# 2026-10-21 · herbert-cuda-bf16, l'attention corrigée passe par défaut : `sdpa.gqa=1`, `sdpa.split=64`, `sdpa.pipe=1`

Décision du 2026-10-21 au matin, après la mesure alternée et la mesure de l'occupation : les trois réglages deviennent le défaut, `sdpa.rows` restant à 16. Commit herbert-cuda-bf16 `f0b646a`. Vérification sur la RTX 4080 le 2026-10-21 de 06 h 27 à 06 h 34 : tokens seulement, rien n'est chronométré. Le service d'inférence résident a été arrêté pendant la vérification, puis relancé par un piège ; il était actif à 06 h 34.

## Ce qui change (`_scripts/_def.diff`, exactement le diff construit et commité)

- `src/cuda/tuning.rs` : défauts `sdpa_split` 8 → 64, `sdpa_gqa` 0 → 1, `sdpa_pipe` 0 → 1, commentaires mis à jour.
- `src/cuda/probe.rs` : `--emit` produit aussi `sdpa_gqa` et `sdpa_combine` à 64 tranches.
- `README.md` : l'ancienne attention s'obtient par `--tune sdpa.gqa=0 --tune sdpa.split=8 --tune sdpa.pipe=0`.

## Ce qui a tourné (`_scripts/_def1-003.sh`)

Deux builds : le nouveau, `~/herbert-cuda-bf16-def` (arbre de `dac152c` plus le diff), et l'ancien, `~/herbert-cuda-bf16-occ` (arbre de `8db5b92`). Les sha256 sont dans `def1/env.txt`. Quatre variantes :

- `new` : le nouveau défaut ;
- `old-sw` : l'ancien binaire avec `--tune sdpa.gqa=1 --tune sdpa.split=64 --tune sdpa.pipe=1` ;
- `new-old` : le nouveau binaire avec `--tune sdpa.gqa=0 --tune sdpa.split=8 --tune sdpa.pipe=0` ;
- `old` : l'ancien défaut.

Trois contrôles :

- **PTX** : `herbert-cuda-probe --emit` des deux builds, sans GPU. Chaque module de l'ancien est comparé à celui du nouveau, et `ptxas -v -arch=sm_89` est passé sur les modules nouveaux.
- **Tokens courts** : `compare.sh` sur p1–p4 et pp128–pp4000, contre les journaux de l'oracle de `runs/plan-final-oracle/{0.8B,4B}` (copiés, pas relancés).
- **Prompts longs** : 10K et 60K (`--ask-file`, `--ctx 61440`), sans réflexion, 64 tokens générés.

## Résultats (`summary.txt`, `def1/ptx/compare.txt`)

- **PTX** : les 440 modules de l'ancien build sont identiques dans le nouveau. Quatre s'y ajoutent, tous sans débordement (`def1/ptx/ptxas-s64.txt`) :
  - `sdpa_gqa_h16kv4d256_s64` et `sdpa_gqa_h8kv2d256_s64` : 104 registres, 33 024 octets de mémoire partagée ;
  - `sdpa_combine_h16d256_s64` et `sdpa_combine_h8d256_s64` : 79 registres.
- **Les ids** : `new` = `old-sw` et `new-old` = `old`, pour les 20 prompts courts et les 4 formes longues, en 0.8B et en 4B. Aucune paire ne diffère.
- **Contre l'oracle**, le nouveau défaut s'écarte comme le faisaient les interrupteurs (dossier `2026-10-20-cuda-bf16-attention-gqa-pipe`) :
  - 0.8B : p3 s'écarte au token 172, alors que l'ancien défaut lui était identique ; p2 et p4 s'écartent au token 8, comme avant ;
  - 4B : p2 devient identique, alors que l'ancien défaut s'écartait au token 32 ; p4 s'écarte au token 105 et pp512 au token 0, comme avant.
- **Nouveau contre ancien défaut sur les longs prompts** :
  - 0.8B : écart à l'élément 10 à 10K, à l'élément 17 à 60K ;
  - 4B : identiques à 10K, écart à l'élément 23 à 60K.

  Ce sont les écarts relevés par la mesure alternée, dus à l'ordre des sommes du decode de `sdpa.gqa`.

## Fichiers

- `def1/` :
  - `compare-{0.8B,4B}/` et `compare-*.out` : journaux de compare (l'oracle copié et les quatre variantes) ;
  - `long/` : les prompts longs ;
  - `ptx/` : `compare.txt`, `ptxas-s64.txt`, les journaux d'émission et les quatre nouveaux modules (`new-s64/`). Les 440 modules communs ne sont pas copiés : `herbert-cuda-probe --emit` les reproduit depuis `8db5b92` ou `f0b646a`.
  - `build.log`, `env.txt`, `end.txt`.
- `def1.out` : la sortie du script.
- `summary.txt` : la sortie de `_scripts/_def1_summary.py def1`.
- `_scripts/` : le script, le résumé et le diff.
