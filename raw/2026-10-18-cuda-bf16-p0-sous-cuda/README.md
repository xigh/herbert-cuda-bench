# Garder la 4080 en P0 sous CUDA : le profil d'application `CudaNoStablePerfLimit` (2026-10-18)

Question ouverte des vérifications du moteur : chercher d'abord à rester en P0 avant de décider de la règle des comparaisons. Sous CUDA, la 4080 passe en P2, mémoire à 10 801 MHz au lieu de 11 201, malgré `-lgc 2805` et `-lmc 11201`. Sous Vulkan, elle reste en P0. La promesse publique de #002 était : « je compte l'étudier ».

## Ce que disent les sources (pages publiques, consultées le 2026-10-18)

- C'est un comportement voulu du pilote : tout contexte CUDA maintient la carte en P2 jusqu'à sa destruction (open-gpu-kernel-modules, discussion #261, 2022 ; forum NVIDIA « Remove P2 forced state from drivers », 2023). Sous Windows, NVIDIA Profile Inspector a un réglage « CUDA – Force P2 State ». Sous Linux, il n'y a pas de paramètre `NVreg_*`.
- La branche R580 ajoute la variable d'environnement `CUDA_DISABLE_PERF_BOOST=1`. Elle fait l'inverse : elle empêche CUDA de monter la carte, pour économiser l'énergie. Un utilisateur constate en décembre 2025 qu'elle ne permet pas de dépasser la P2. Un modérateur NVIDIA le confirme le 2026-01-23 (« the opposite problem »), et renvoie au profil d'application **`CudaNoStablePerfLimit`** : clé `0x166c5e` à 0, sélectionnée par nom de processus (open-gpu-kernel-modules, issue #333, commentaire d'un collaborateur NVIDIA du 2025-12-18). Ce réglage n'existe que par profil, pas par variable d'environnement.
- Le pilote 595.99.02 livre déjà ce profil, appliqué à `obs` et `Discord` (`driver-profile-excerpt.txt`, extrait de `/usr/share/nvidia/nvidia-application-profiles-595.99.02-rc`).

Sources : [forum NVIDIA 241998](https://forums.developer.nvidia.com/t/remove-p2-forced-state-from-drivers/241998), [forum NVIDIA 269166](https://forums.developer.nvidia.com/t/performance-state-switches-from-p0-to-p2-when-starting-program/269166), [forum NVIDIA 353276](https://forums.developer.nvidia.com/t/cuda-p2-forced-state-from-drivers/353276), [forum NVIDIA 61031](https://forums.developer.nvidia.com/t/whats-the-situation-with-vdpau-vaapi-nvdec/61031/51), [open-gpu-kernel-modules, discussion 261](https://github.com/NVIDIA/open-gpu-kernel-modules/discussions/261), [open-gpu-kernel-modules, issue 333](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/333). Les copies de ces pages ne sont pas dans ce dépôt.

## Mesures sur la RTX 4080 (`clk/`, `env.txt`)

Toujours herbert-cuda (`herbert-cuda-bf16` `fa10f73`, candidat), 4B, p4 (2 882 tokens, 186 générés), mémoire échantillonnée à 100 ms (`nvidia-smi`), deux passages par variante :

| variante | échantillons mémoire / état | prefill | décodage |
|---|---|---|---|
| sans rien (`r*-default`) | 75 × 10 801 MHz P2, 70 × 11 201 P0 (repos avant et après) | 11 314 / 11 311 tok/s | 74,02 / 73,99 tok/s |
| `CUDA_DISABLE_PERF_BOOST=1` (`r*-noboost`) | 75 × 10 801 P2, 70 × 11 201 P0 | 11 308 / 11 338 | 74,02 / 74,02 |
| profil `CudaNoStablePerfLimit` pour `herbert-cuda-chat` (`r*-profile`) | **143 × 11 201 MHz P0**, aucune P2 | 11 317 / 11 320 | **76,71 / 76,72** |

Le profil (`nvidia-application-profiles-rc.test1`) a été placé dans `~/.nv/nvidia-application-profiles-rc`, qui n'existait pas. Il a été retiré après le test (`nvidia-application-profiles-rc.removed-after-test`), en attendant la décision sur la règle des comparaisons.

Lecture :
- Avec le profil, la carte reste en P0 à 11 201 MHz sous CUDA.
- Le décodage du 4B gagne 3,6 %, exactement le rapport des horloges mémoire (11 201 / 10 801 = 1,037). Le prefill, limité par le calcul, ne bouge pas.
- Ce sont des exécutions uniques par passage : un contrôle, pas une campagne.
- La règle porte sur le nom du processus : pour llama.cpp (`llama-bench`, `llama-server`) et vLLM (`python`, `vllm`), il faudrait leurs noms, et la question de l'équité des comparaisons se pose.
