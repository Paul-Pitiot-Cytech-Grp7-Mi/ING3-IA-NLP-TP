# Usage : make help
PY  := .venv/bin/python
NB  := .venv/bin/jupyter nbconvert --to notebook --execute --output-dir out --ExecutePreprocessor.timeout=1800
MODELE_OLLAMA := qwen2.5:1.5b

.DEFAULT_GOAL := help
.PHONY: help setup lab nlp-tp2 nlp-tp3 transformer-tp1 transformer-tp2 transformer-tp3 transformer-tp4 all clean

help:
	@echo "make setup           environnement Python 3.11 + dépendances (uv)"
	@echo "make nlp-tp2         NLP TP2  : deep learning (~45 s, réseau requis)"
	@echo "make nlp-tp3         NLP TP3  : prompting     (~5 s avec le cache, Ollama et réseau requis)"
	@echo "make transformer-tp1 Transformer TP1 : bigramme (<10 s, réseau requis)"
	@echo "make transformer-tp2 Transformer TP2 : auto-attention (~3 s)"
	@echo "make transformer-tp3 Transformer TP3 : LayerNorm et MLP (~3 s)"
	@echo "make transformer-tp4 Transformer TP4 : modèle complet (~12 min)"
	@echo "make all             les six à la suite (~14 min)"
	@echo "make lab             ouvre JupyterLab"
	@echo "Les notebooks exécutés sont écrits dans out/ ; les notebooks versionnés ne sont pas modifiés."

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt

lab:
	.venv/bin/jupyter lab

nlp-tp2:
	$(NB) nlp/TP2_NLP_deep_learning.ipynb

nlp-tp3:
	@curl -sf localhost:11434/api/tags >/dev/null || { echo "Ollama ne répond pas : ouvrir l'app Ollama ou lancer 'ollama serve' dans un autre terminal"; exit 1; }
	@ollama list | grep -q "$(MODELE_OLLAMA)" || ollama pull $(MODELE_OLLAMA)
	$(NB) nlp/TP3_NLP_prompting.ipynb

transformer-tp1:
	$(NB) transformer/TP1_Transformer_bigramme.ipynb

transformer-tp2:
	$(NB) transformer/TP2_Transformer_auto_attention.ipynb

transformer-tp3:
	$(NB) transformer/TP3_Transformer_layernorm_mlp.ipynb

transformer-tp4:
	$(NB) transformer/TP4_Transformer_modele_complet.ipynb

all: nlp-tp2 nlp-tp3 transformer-tp1 transformer-tp2 transformer-tp3 transformer-tp4

clean:
	rm -rf out transformer/input.txt
