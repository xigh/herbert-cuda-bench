# herbert-cuda : le GEMM en `wmma` et le softmax par la mémoire partagée, les deux derniers interrupteurs de l'ablation (2026-10-18)

Suite des vérifications du moteur. Ce sont les deux interrupteurs qui rapprochent herbert-cuda de ce qu'un shader Vulkan peut écrire. Code : `herbert-cuda-bf16` `d9c7ad5` (`~/herbert-cuda-mma`). Les deux sont désactivés par défaut, et le PTX par défaut est inchangé (`ptx-default-diff.txt` vide contre l'étape précédente).

- `sdpa.softmax=shared` fait le softmax de l'attention du prefill comme `vk/fatc.rs`, faute de connaître la disposition des fragments. S part en mémoire partagée (f32), deux lanes par ligne calculent max, exponentielles et somme, puis P revient en bf16. Chaque ligne relit sa correction en mémoire partagée pour l'appliquer aux accumulateurs de O, et P est rechargé en fragments A par `ldmatrix`. 213 registres contre 207, sans spill.
- `gemm.wmma=1` fait le GEMM en `wmma.m16n16k16`, fragments dans la disposition du pilote, comme les matrices coopératives de Vulkan. L'épilogue passe par la mémoire partagée, puisque `wmma.store` ne connaît pas le bf16. Il n'existe qu'en disposition rembourrée.
- **SASS** (`sass-variants.txt`, `ptxas` 12.8) : `ptxas` traduit `wmma.load` en `LDSM.16.M88.4` et `wmma.mma` en `HMMA.16816.F32.BF16`, en même nombre que le chemin `ldmatrix` + `mma` explicite (8 et 16 par tour de K en 64 × 64). Seul l'épilogue s'ajoute : `STS.64` et `LDS.64`, 16 de chaque en 64 × 64. Côté pilote CUDA, la disposition opaque ne coûte donc aucune instruction dans la boucle. Ce que fait le pilote Vulkan reste inconnu.

Machine : RTX 4080, pilote 595.99.02, SM 2 805 MHz, carte en P2 à 10 801 MHz de mémoire pendant la mesure (1 444 échantillons sur 1 543, `clocks.csv`). Du 08:03 au 08:41 (`env.txt`, `end.txt`). Script `_wmma-softmax-003.sh`.

## GEMM isolé, `mma` contre `wmma` (`gemm/`, `gemm-summary.txt`, `_wmma_gemm_summary.py`)

Les 6 formes de chaque modèle, M de 128 à 2 048, tuiles 64 × 64 s3 et 64 × 128 w2x2 s2, disposition rembourrée, copie par `cp.async` ou par registres, 3 tours. 864 contrôles : toutes les sorties `wmma` sont identiques bit pour bit à celles de `mma` / `ldmatrix` / `cp.async`.

Rapport de temps `wmma` / `mma`, moyenne géométrique sur les 6 formes :
- avec `cp.async` : ×0,95 à ×1,04 ;
- avec la copie par registres : ×1,00 à ×1,13.

L'écart vient de l'épilogue par la mémoire partagée.

## Tokens contre l'oracle CPU (`cmp-0.8B.out`, `cmp-4B.out`)

- `wmma` et `pad` : mêmes ids que la base.
- Le softmax par la mémoire partagée change l'ordre des sommes. En 0.8B p3, il diverge de l'oracle au token 144 (la base est identique jusqu'au bout) ; en 4B p3, au token 141. À l'inverse, en 4B p2 et p4 il devient identique à l'oracle, là où la base divergeait aux tokens 32 et 105.
- Les premiers écarts sont des quasi-égalités de logits bf16 (`softmax-first-try/top-*.log`) :
  - 0.8B au token 144 : 21,375 / 21,375 contre 21,5 / 21,375 ;
  - 4B au token 141 : 21,625 / 21,375 contre 21,5 / 21,5.

## Le moteur entier (`runs/`, `engine-summary.txt`, `_engine_summary.py`)

N = 5 tours alternés, 350 exécutions, écart médian entre tours 0,37 % (au plus 1,5 %). Prefill de 128 à 4 000 tokens, décodage p3. Base : le candidat. Variantes :
- `softmax-shared` : + `sdpa.softmax=shared` ;
- `pad` : sans XOR, la référence de `wmma` ;
- `wmma` : `pad` + `gemm.wmma=1` ;
- `vklike` : `wmma` + `gemm.copy=sync` + `sdpa.softmax=shared` + `graph=0`. C'est tout ce que vk ne peut pas avoir, retiré ensemble. `gemm.ldmatrix=0` n'est pas retiré : le chemin `wmma` passe déjà par les `LDSM` du pilote CUDA.

| variante / base | 0.8B 128 | 256 | 512 | 1 024 | 2 048 | 4 000 | décodage | 4B 128 | 256 | 512 | 1 024 | 2 048 | 4 000 | décodage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax en mémoire partagée | 0,997 | 0,998 | 0,998 | 0,995 | 0,993 | 0,988 | 0,998 | 0,999 | 1,000 | 1,001 | 1,001 | 0,998 | 0,995 | 1,000 |
| pad | 0,959 | 0,871 | 0,976 | 0,958 | 0,953 | 0,944 | 1,000 | 0,958 | 0,885 | 0,944 | 0,933 | 0,916 | 0,922 | 1,000 |
| wmma (sur pad) | 0,941 | 0,870 | 0,947 | 0,936 | 0,938 | 0,933 | 1,000 | 0,953 | 0,891 | 0,927 | 0,918 | 0,912 | 0,918 | 1,000 |
| vklike | 0,848 | 0,828 | 0,911 | 0,957 | 0,923 | 0,933 | 0,953 | 0,926 | 0,847 | 0,896 | 0,912 | 0,907 | 0,909 | 0,989 |

Lecture :
- Le softmax gardé en registres vaut au plus 1,2 % de prefill.
- `wmma` contre `mma` sur la même disposition (`wmma` / `pad`) : −0 à −3 %, l'épilogue.
- Tout ce que vk n'a pas, retiré ensemble : −4 à −17 % de prefill (le plus aux prompts courts du 0.8B) ; au décodage, seulement le graphe (−4,7 % / −1,1 %).

Ce n'est pas herbert-vk : le chemin `wmma` du pilote CUDA reste ici un `LDSM`, et la copie reste la nôtre.

## Contenu

- `_wmma-softmax-003.sh`, `run.out`, `env.txt`, `end.txt`, `clocks.csv`, `build.log`.
- `gemm/`, `gemm-summary.txt`, `_wmma_gemm_summary.py` : le GEMM isolé.
- `cmp-0.8B/`, `cmp-4B/` et `.out` : les tokens contre l'oracle (journaux CPU de herbert-jit-bf16 v0.1.0).
- `runs/`, `engine-summary.txt`, `_engine_summary.py` : le moteur entier.
- `ptx-after/`, `ptx-default-diff.txt`, `ptx-variants/`, `sass-variants.txt` : le PTX et le SASS.
- `softmax-first-try/` : premier essai du softmax, avec son build, le PTX, les essais p3 et les 2 meilleurs logits (`HERBERT_VK_TOP=2`).
