# A'Space adapter — PathFM

**Adresse :** `fiber:a0:PathFM` — D-N053 pas A (design Clara, dispatch D-N053).
**Fork souverain :** `Amdkn/PathFM` (parent `autogluon/PathFM`, branche `main`).
**Gel amont :** `8a3a42c05ce84981bc0fa292d5503b3f2ea22a8b` (« Prepare PathFM release »).
**Bind Cordis Fiber (16L) :** `d94942ecb0b12609bf1d450e50ab98240999d549` — `aspace-plugin.json`, re-applique VERBATIM sur `main` au-dessus de l'adapter (F8 D2).
**Licence :** `CC BY-NC 4.0`.
**Version adapter :** 0.1.0 — contrat seul, patron D-N048 (dify / cordis).

## Ce que fait l'adapter

PathFM : generative forecasting de trajectoires multivariees (modeles chaotiques). Code CC BY-NC 4.0, modeles non-commerciaux, confines au profil interne.

Surface : `validate()` (constate : gel, licence, importabilite amont, poids
presents dans le fork, zero appel reseau), `doctor()` (liste de controles),
`describe()` (manifeste machine). **Aucun fichier amont touche, aucun appel
reseau, aucun poids monte dans le fork.**

## Confinement

CONFINEMENT PROFIL INTERNE (design D-N053 D3) : le CODE du repo PathFM est CC BY-NC 4.0 (LICENSE lue par Clara) et les MODELES PathFM sont non-commerciaux ; jamais en B2B (verifie a la configuration du profil commercial, H-A5). Manifeste ML corrige a CC BY-NC 4.0 (acte nomme du pas A, evidence D-N053/a/PathFM/).

## Tests

```bash
python -m pytest aspace-PathFM-adapter/tests/test_smoke.py -v
```

Smoke : constantes (gel 40-hex, adresse fiber, licence), `validate()` vert,
`doctor()` tout ok, import amont, zero import reseau dans l'adapter.
La suite amont est celle du module au gel (voir la fiche d'adaptation :
`PathFM-ADAPTATION.md`, evidence D-N053/a/PathFM/).
