# Mes prédictions pour la campagne de #003 (CUDA), le 18 octobre 2026 à 10 h 30 (08:30 UTC)

> Copie publique. Les mots dictés sont reproduits tels qu'ils ont été dictés et écrits avant les mesures ; seuls des renvois à des notes de travail privées ont été retirés. Les prédictions de la phase 4 (P10 à P13) ne sont pas dans cette version, avec la phase 4 elle-même.

Écrites avant les balayages de réglages des comparateurs et avant toute campagne comparative de #003. Elles viennent **après** ce qui est déjà mesuré et que je connais, ce que l'article devra dire :
- herbert-cuda contre vLLM 0.28.0 et herbert-vk-bf16 ;
- les vérifications et l'ablation de herbert-cuda contre lui-même ;
- la P0 sous CUDA ;
- les contrôles de fonctionnement des versions récentes.

## Ses mots, tels quels (dictés ; « je » = herbert-cuda)

« Je fait un peu moins bien ou jeu égale à llama.cpp, mais moins bien que vllm et sglang en prefill, et moins bien que llama.cpp en decode, et aussi bien, voire un peu mieux que vllm et sglang »

## Ce qu'on vérifiera, une ligne par prédiction

- **P1, prefill** : herbert-cuda un peu derrière llama.cpp, ou à égalité.
- **P2, prefill** : herbert-cuda derrière vLLM.
- **P3, prefill** : herbert-cuda derrière SGLang.
- **P4, décodage** : herbert-cuda derrière llama.cpp.
- **P5, décodage** : herbert-cuda aussi bien que vLLM, voire un peu mieux.
- **P6, décodage** : herbert-cuda aussi bien que SGLang, voire un peu mieux.

La lecture de « un peu moins bien », « à égalité », « derrière » et « aussi bien, voire un peu mieux » se fera longueur par longueur et modèle par modèle, sans retoucher ce texte.

Ce que les prédictions ne disent pas : la version de llama.cpp (CUDA, ou Vulkan aussi), transformers, et herbert-vk-bf16.

## Complément, le 18 octobre 2026 à 10 h 32 (08:32 UTC), avant les balayages

Ses mots, tels quels : « mieux que transformers en prefill et en decode, et un poil mieux que herbert-vk-bf16 ... pour llama.cpp je veux que tu tests face aux dernières version possible et stable (git pull) »

- **P7, prefill** : herbert-cuda devant transformers.
- **P8, décodage** : herbert-cuda devant transformers.
- **P9, prefill et décodage** : herbert-cuda « un poil » devant herbert-vk-bf16.
- **Version de llama.cpp** : la dernière version possible et stable, prise par `git pull` au moment de la campagne. Les prédictions P1 et P4 se lisent contre elle.

# Prédictions pour la phase 5 (le prefill long, 10 000 à 60 000 tokens), le 19 octobre 2026 à 16 h 06 (14:06 UTC)

Dictées par Philippe après le protocole de la phase 5 (`680e3c7`), avant toute mesure de cette phase. Elles viennent après le contrôle de démarrage à 60K du protocole (SGLang manque de mémoire en 0.8B et 4B, transformers en 4B), dont Philippe a eu le résultat ; les temps de ce contrôle ne lui ont pas été donnés.

## Ses mots, tels quels (dictés)

« Je crois que je vais me faire déboîter pour les préfils longs. Et le décode aussi d'ailleurs. »

## Ce qu'on vérifiera, une ligne par prédiction

- **P14, prefill long** : herbert se fait battre nettement (« déboîter ») de 10K à 60K.
- **P15, décodage** : herbert se fait battre aussi au décodage (lecture ajoutée, hors dictée : le décodage qui suit un prompt long, à cette profondeur).

Lecture (ajoutée, hors dictée) : « herbert » couvre les trois formes de herbert-cuda du protocole (bf16, f16, f16 avec `gemm.f16=1`), chacune lue séparément, contre chaque comparateur ; herbert-vk-f16 est lu à part. Le texte dicté n'est pas retouché.
