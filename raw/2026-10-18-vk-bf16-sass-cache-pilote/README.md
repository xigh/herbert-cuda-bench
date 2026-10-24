# herbert-vk-bf16 : le SASS de ses pipelines, lu dans le cache disque du pilote, sans interface (2026-10-18)

Suite de l'inspection des pipelines de herbert-vk : lire le SASS de vk, par un contrôle sans interface. La voie 1 (`VK_KHR_pipeline_executable_properties`) ne rendait que des statistiques. La voie 3 (Nsight Graphics) attendait son installation. Ici, c'est la voie 2 : le cache disque du pilote. Le dépôt herbert-vk n'est pas touché : on exécute seulement son binaire `2f3ae48` (la build de #002, `~/herbert-vk-bf16-v1`), et tout le reste se passe dans `~/vk-sass-003` sur rtx4080.

## Méthode

1. Un chargement par modèle, avec un cache disque dédié (`__GL_SHADER_DISK_CACHE=1`, `__GL_SHADER_DISK_CACHE_PATH`, `__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1`) : `herbert-vk-chat $MODELS_DIR/Qwen3.5-{0.8B,4B} --device nvidia`, p3, 4 tokens (`glcache-*/run.log`). Le cache ne contient alors que les pipelines de vk (`glcache-*/cache/`, brut du pilote).
2. Les `.bin` du cache sont des trames zstd. `_glcache_frames.py` les décompresse (`glcache-*/frames/`). Les ELF trouvés dans quelques trames sont du code hôte x86-64 (`e_machine` 62, `_glcache_elf.py`), pas du SASS.
3. Chaque pipeline est un conteneur qui commence par `NVDANVVMNVuc`. L'entier 32 bits à l'octet 76 est la taille du code : c'est la « Binary Size » relevée par `VK_KHR_pipeline_executable_properties`, et les 76 pipelines concordent. L'entier à l'octet 80 vaut le décalage du code moins 8. Le code est du SASS brut sm_89 : `EXIT`, la boucle `BRA` sur elle-même, puis des `NOP`. `_glcache_sass.py` découpe le code (`.sass.bin`) et le passe à `nvdisasm -b SM89` du toolkit 12.8 (`.sass`). Les 38 pipelines de chaque modèle se désassemblent sans erreur (`sass2.log`, code retour 0 partout).
4. `_names.py` rapproche chaque pipeline de son nom relevé par `VK_KHR_pipeline_executable_properties`, par la taille du binaire (`names.txt`). Les GEMM de même tuile ont le même code ; deux pipelines de 8 832 octets du 0.8B se distinguent par la présence de `HMMA`.

Le format du conteneur est déduit de ces deux cas, pas d'une documentation. Le pilote est 595.99.02 (`env-glcache.txt`).

## Ce que dit le SASS (4B ; le 0.8B donne les mêmes motifs)

- **GEMM coopératif 64 × 64 × 64** (`gemm_coop_*_t64x64x64`, 77 registres) : 128 `HMMA.16816.F32.BF16`, 96 `LDSM.16.M88.4`, 32 `STS.128`, 32 `LDG.E.128.STRONG.SM`. **Le pilote Vulkan traduit les chargements des matrices coopératives en `ldmatrix`** (`LDSM`). Il n'y a **aucun `LDGSTS`** (la copie asynchrone, `cp.async`) : les tuiles passent de la mémoire globale aux registres, puis à la mémoire partagée.
- **Attention du prefill** (`sdpa_tc_h16kv4d256_k32_qr`, 255 registres, 188 800 octets, 11 800 instructions) :
  - 1 024 `HMMA` ;
  - 528 `LDSM`, dont 256 en `LDSM.16.MT88.4` (transposé) ;
  - **77 `STL` et 42 `LDL.LU`**, soit de vrais accès à la mémoire locale, donc des spills. L'inspection précédente n'en avait qu'une hypothèse, tirée des 160 octets de « Local Memory Size ».
- **Récurrence DeltaNet sur tensor cores** : `dn_tc_state` fait 448 `HMMA`, 192 `LDSM` et 929 `LDG` ; `dn_tc_prep` 48 `HMMA` et 726 `LDS`.
- **Chargements globaux** : dans les 76 pipelines des deux modèles, les 4 042 `LDG` portent tous un qualificatif fort, `.STRONG.SM` (et 9 `.STRONG.GPU`). Aucun n'est un `LDG` simple. Côté CUDA, `ld.global.cg` donnait `LDG.E.128.STRONG.GPU`, et herbert-cuda est passé à `ld.global.ca` (`LDG.E.128` simple) avant de mesurer. Le coût éventuel des chargements forts dans vk n'est pas mesuré.

Pour comparaison, le GEMM de herbert-cuda fait `LDGSTS` + `LDSM` + `HMMA` ; son attention, 207 registres sans spill.

## Fichiers

- `glcache-{0.8B,4B}/` : `run.log`, `cache/` (brut du pilote), `frames/` et `frames.log`, `sass2/` (`.sass.bin`, `.sass`, `.strings` par pipeline) et `sass2.log`.
- `names.txt` : nom, taille, nombre d'instructions et quelques opcodes par pipeline.
- `_glcache_frames.py`, `_glcache_elf.py`, `_glcache_sass.py`, `_names.py` : l'outillage. `_names.py` lit `../2026-10-18-vk-bf16-statistiques-pipelines/stats-*.txt`.
