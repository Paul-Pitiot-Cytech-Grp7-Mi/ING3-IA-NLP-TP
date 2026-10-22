# NLP et Transformers : TP

Travaux pratiques du cours « NLP et Transformers » (CY Tech, ING 3). Paul Pitiot, Option IA, Groupe B.

## Les 6 TP

Chaque notebook est livré avec ses sorties. Le sujet, les étapes traitées et les consignes sont rappelés en tête de chaque notebook, et un tableau « Consignes du sujet » en fin de notebook indique où chaque consigne est traitée.

| Série | TP | Notebook | Sujet | Commande | Durée |
|---|---|---|---|---|---|
| NLP | TP2 | [`nlp/TP2_NLP_deep_learning.ipynb`](nlp/TP2_NLP_deep_learning.ipynb) | ML classique contre deep learning (MLP, RNN, LSTM) | `make nlp-tp2` | ≈ 45 s |
| NLP | TP3 | [`nlp/TP3_NLP_prompting.ipynb`](nlp/TP3_NLP_prompting.ipynb) | Prompting d'un LLM local (`qwen2.5:1.5b`) | `make nlp-tp3` | ≈ 5 s |
| Transformer | TP1 | [`transformer/TP1_Transformer_bigramme.ipynb`](transformer/TP1_Transformer_bigramme.ipynb) | Modèle bigramme au caractère | `make transformer-tp1` | < 10 s |
| Transformer | TP2 | [`transformer/TP2_Transformer_auto_attention.ipynb`](transformer/TP2_Transformer_auto_attention.ipynb) | Auto-attention pas à pas | `make transformer-tp2` | ≈ 3 s |
| Transformer | TP3 | [`transformer/TP3_Transformer_layernorm_mlp.ipynb`](transformer/TP3_Transformer_layernorm_mlp.ipynb) | LayerNorm, MLP et résiduel | `make transformer-tp3` | ≈ 3 s |
| Transformer | TP4 | [`transformer/TP4_Transformer_modele_complet.ipynb`](transformer/TP4_Transformer_modele_complet.ipynb) | Modèle complet assemblé, entraîné, génération | `make transformer-tp4` | ≈ 12 min |

Le TP1 NLP (ML classique) n'est pas dans ce dépôt : le TP2 NLP en reprend le parsing et le découpage, et se lit seul.

## Résultat de chaque TP

| TP | Résultat principal |
|---|---|
| NLP TP2 | LogisticRegression sur TF-IDF fait 0,616 de F1 macro, devant le MLP (0,584), le LSTM (0,392) et le RNN (0,273) : le deep learning ne fait pas mieux sur 486 répliques. |
| NLP TP3 | Le LLM local fait 0,145 à 0,164 de F1 macro selon le prompt, contre 0,670 pour TF-IDF + LogisticRegression sur les mêmes 100 répliques. |
| Transformer TP1 | Le bigramme atteint 2,51 de loss de validation à `lr = 1e-2`, près du meilleur bigramme possible (2,48). |
| Transformer TP2 | La tête d'auto-attention est vérifiée : lignes de poids qui somment à 1, masque causal, échelle `1/√dh`. |
| Transformer TP3 | La `LayerNorm` écrite à la main égale `nn.LayerNorm`, le MLP `C → 4C → C` et le résiduel sont testés. |
| Transformer TP4 | Le modèle complet (0,21 M paramètres) atteint 1,82 de loss de validation, sous le bigramme du TP1, et génère 2 000 caractères en forme de pièce. |

Les TP NLP portent sur la même tâche : retrouver quel personnage de Tiny Shakespeare parle (5 locuteurs, 761 répliques, 608 en train et 153 en test, mêmes pour tous les TP NLP).

## Lancer

Testé sur un MacBook M1 (macOS), sur CPU, avec Python 3.11.

```bash
brew install uv           # gestionnaire d'environnement (installe aussi Python 3.11)
brew install ollama       # seulement pour le TP3 NLP
make setup                # environnement .venv et dépendances (≈ 1 min)
make transformer-tp4      # un TP à la fois, ou : make all (≈ 14 min)
```

Le notebook exécuté est écrit dans `out/` : les notebooks versionnés ne sont jamais réécrits. Pour travailler dans un notebook : `make lab`.

**TP3 NLP.** Il faut qu'Ollama tourne (`ollama serve`, ou l'application ouverte) ; `make nlp-tp3` télécharge `qwen2.5:1.5b` (≈ 1 Go) s'il manque. Les 300 réponses du run de référence sont dans `nlp/cache/` : le notebook les relit sans rappeler le modèle, ce qui prend 5 s au lieu de ≈ 80 s.

**Réseau.** Les TP2 et TP3 NLP et les TP1 et TP4 Transformer téléchargent Tiny Shakespeare s'il manque (`input.txt` pour les Transformer). Les TP2 et TP3 Transformer n'ont besoin de rien.

## Reproductibilité

- Les graines sont fixées et les versions des dépendances sont figées dans `requirements.txt`.
- Vérifié sur une copie propre du dépôt : `make setup` puis `make all` exécutent les 6 notebooks sans erreur, et redonnent les mêmes scores, les mêmes losses et les mêmes textes que les sorties versionnées.
- Seuls ces éléments changent d'une exécution à l'autre : les durées, l'affichage `np.float32(...)` dans le TP2 NLP, la version de `torch` affichée au TP2 NLP (2.1.2 dans les sorties enregistrées, 2.14.0 à la réexécution), la loss d'entraînement du LSTM à l'époque 30 (0,084 contre 0,082), et le TP3 NLP, qui relit le cache au lieu d'appeler le modèle.
- Sur une autre machine, les derniers chiffres des entraînements peuvent différer légèrement.
