# Session RL — prise en deux temps sur SO101
## Léo Pouilly — Arts et Métiers

---

## Ce dont tu as besoin

- SO101 follower branché sur `/dev/ttyACM0` (calibré)
- Arduino (Uno/Nano/Mega) avec le sketch `grip_piezo_so101.ino` flashé
- Capteur piézoélectrique câblé sur A0 avec résistance 1 MΩ vers GND
- Environnement conda `lerobot` actif
- 4 objets : gobelet plastique, boîte carton, bloc bois, bloc métal

---

## Étape 1 — Flasher l'Arduino

Ouvrir `grip_piezo_so101.ino` dans l'IDE Arduino et flasher.
Ouvrir le moniteur série à 115200 bauds — tu dois voir :

```
PIEZO_READY
TARE_OK:512
```

Si tu ne vois rien : vérifier le câblage A0 + résistance 1 MΩ.

---

## Étape 2 — Installer la dépendance série

```bash
conda activate lerobot
pip install pyserial
```

---

## Étape 3 — Vérifier les ports USB

```bash
# Identifier tous les ports
lerobot-find-port

# Résultat attendu dans ton setup :
# /dev/ttyACM0  → SO101 follower
# /dev/ttyACM1  → SO101 leader
# /dev/ttyACM2  → Arduino  (ou /dev/ttyUSB0 selon le modèle)
```

Le script détecte l'Arduino automatiquement en excluant ACM0 et ACM1.

---

## Étape 4 — Tester en simulation (recommandé d'abord)

```bash
cd ~/lerobot
python grip_rl_so101.py --simulate
```

Tu verras défiler les 200 essais simulés avec les labels automatiques.
Vérifie que le dossier `experiments/session_YYYYMMDD_HHMMSS/` est créé.

---

## Étape 5 — Lancer la vraie session

```bash
conda activate lerobot
cd ~/lerobot

python grip_rl_so101.py
```

### Déroulement de chaque essai

```
═══════════════════════════════════════════════════════════
  Essai  12/200  │  gobelet_plastique  (12/50)  │  ε=0.847
═══════════════════════════════════════════════════════════
  [0] Retour HOME...
  Placer l'objet : GOBELET_PLASTIQUE
  Appuyer sur [Entrée] quand l'objet est en position... ↵

  [1] Phase 1 — fermeture exploratoire (15°, lente)...
      pic=58.3 mV  grad_norm=14.2  duree=45 ms

  [2] Décision : A1 = 35°  [exploration]
  [3] Phase 2 — serrage à 35°...
      Maintien 2s...

  ┌─────────────────────────────────────────┐
  │  Résultat ?                              │
  │  [o] réussi    [f] trop fort    [r] raté │
  │  [q] arrêter la session proprement       │
  └─────────────────────────────────────────┘
  > o
  → réussi (+1.0)
  [OK]  r=+1.0  cumul gobelet_plastique: 8/12 (67%)

  Replacer l'objet — 8s de reset... prêt.
```

### Critères de label

| Touche | Situation | Récompense |
|--------|-----------|------------|
| `o` | Objet tenu fermement, pas déformé | +1.0 |
| `f` | Objet tenu mais écrasé / déformé | −0.5 |
| `r` | Objet tombé / pas saisi | −1.0 |
| `q` | Arrêter proprement | — |

---

## Reprise après interruption

Si tu dois t'arrêter (Ctrl+C ou `q`), tout est sauvegardé.
Pour reprendre :

```bash
python grip_rl_so101.py --reprendre experiments/session_YYYYMMDD_HHMMSS
```

La Q-table et ε sont rechargés exactement là où tu en étais.

---

## Résultats

Tout est dans `experiments/session_YYYYMMDD_HHMMSS/` :

```
session_20260515_143022/
  ├── transitions.jsonl   ← une ligne JSON par essai
  ├── qtable.npy          ← Q-table numpy (shape 6×6×5)
  ├── qtable_meta.json    ← epsilon + nb mises à jour
  ├── config.json         ← paramètres de la session
  └── rapport.txt         ← résumé statistique final
```

Pour visualiser :

```bash
python grip_rl_visualize.py experiments/session_YYYYMMDD_HHMMSS
# → génère rapport_visuel.png
```

---

## Dépannage rapide

| Symptôme | Cause | Solution |
|----------|-------|----------|
| `Arduino non trouvé` | Port pas encore reconnu | Attendre 5s après branchement |
| `TARE_OK` absent | Sketch pas flashé | Reflasher `grip_piezo_so101.ino` |
| Pas de contact détecté | Résistance 1 MΩ manquante | Vérifier câblage A0 |
| Robot ne bouge pas | `lerobot` pas installé | `pip install -e .` dans `~/lerobot` |
| `ttyACM0 busy` | Port déjà utilisé | Fermer tout autre programme série |

---

## Après la session

La Q-table apprise encode la politique optimale :
- Bin pic faible + gradient faible → gobelet → A0 (20°)
- Bin pic fort  + gradient fort   → métal   → A4 (95°)

Elle peut être branchée directement dans le `GripGuard`
pour le chemin B (inférence ACT + serrage adaptatif).
