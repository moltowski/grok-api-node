# Grok API Node — Référence Technique

> Package ComfyUI custom node intégrant l'API xAI Grok directement dans les workflows.

---

## Table des matières

1. [Structure du package](#1-structure-du-package)
2. [Entrée dans ComfyUI — `__init__.py`](#2-entrée-dans-comfyui--__init__py)
3. [Client API — `utils/grok_client.py`](#3-client-api--utilsgrok_clientpy)
4. [GrokVisionNode — `nodes/grok_vision_node.py`](#4-grokvisionnode--nodesgrok_vision_nodepy)
5. [GrokPromptBuilderNode — `nodes/grok_prompt_builder_node.py`](#5-grokpromptbuildernode--nodesgrok_prompt_builder_nodepy)
6. [Modèles disponibles](#6-modèles-disponibles)
7. [Format de l'API xAI](#7-format-de-lapi-xai)
8. [Installation sur RunPod](#8-installation-sur-runpod)
9. [Contraintes et limitations](#9-contraintes-et-limitations)

---

## 1. Structure du package

```
grok-api-node/
├── __init__.py                       ← Point d'entrée ComfyUI (NODE_CLASS_MAPPINGS)
├── nodes/
│   ├── __init__.py                   ← Module marker
│   ├── grok_vision_node.py           ← Node analyse d'image → prompt
│   └── grok_prompt_builder_node.py   ← Node génération de N prompts
├── utils/
│   ├── __init__.py                   ← Module marker
│   └── grok_client.py                ← Wrapper HTTP vers l'API xAI
└── requirements.txt                  ← requests>=2.31.0, Pillow>=9.0.0
```

Les imports sont **relatifs** (`from ..utils.grok_client import GrokClient`) ce qui est requis pour que ComfyUI charge correctement le package depuis `custom_nodes/`.

---

## 2. Entrée dans ComfyUI — `__init__.py`

ComfyUI scanne chaque sous-dossier de `custom_nodes/` et importe son `__init__.py`. Il cherche deux dictionnaires :

```python
NODE_CLASS_MAPPINGS = {
    "GrokVisionNode": GrokVisionNode,
    "GrokPromptBuilderNode": GrokPromptBuilderNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokVisionNode": "Grok Vision to Prompt",
    "GrokPromptBuilderNode": "Grok Prompt Builder",
}
```

- **`NODE_CLASS_MAPPINGS`** : clé = identifiant interne du node (utilisé dans les fichiers workflow JSON), valeur = classe Python.
- **`NODE_DISPLAY_NAME_MAPPINGS`** : clé = même identifiant interne, valeur = label affiché dans l'UI ComfyUI.

Les nodes apparaissent dans la catégorie **`Grok API`** (définie par `CATEGORY = "Grok API"` dans chaque classe).

---

## 3. Client API — `utils/grok_client.py`

### Rôle

Encapsule tous les appels HTTP vers `https://api.x.ai/v1/chat/completions`. Les nodes n'accèdent jamais directement à `requests` — tout passe par `GrokClient`.

### Initialisation

```python
client = GrokClient(api_key)
```

Stocke la clé et construit le header `Authorization: Bearer <key>` réutilisé sur chaque requête.

### `chat(model, messages, temperature, max_tokens) → str`

Appel générique chat completions. Payload envoyé :

```json
{
  "model": "grok-4-1-fast-non-reasoning",
  "messages": [...],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

**Gestion des erreurs** : ne lève pas d'exception — retourne une string d'erreur formatée `[Grok API Error ...]` en cas d'échec HTTP, timeout, ou erreur de parsing. Cela permet à ComfyUI d'afficher l'erreur comme output du node plutôt que de crasher le workflow.

Timeout : **120 secondes** (les modèles reasoning peuvent être lents).

### `vision(model, system_prompt, user_text, image_base64, image_mime, temperature, max_tokens) → str`

Construit un message multimodal et délègue à `chat()` :

```python
messages = [
    {"role": "system", "content": system_prompt},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{image_base64}"}
            }
        ]
    }
]
```

L'image est envoyée en **data URI inline** (`data:image/png;base64,...`) — aucun upload préalable requis. L'ordre text → image dans le tableau `content` correspond à la convention xAI.

---

## 4. GrokVisionNode — `nodes/grok_vision_node.py`

### Fonction

Prend une image ComfyUI + un prompt système + un message utilisateur, appelle Grok Vision, retourne un string (prompt généré).

### Pipeline d'exécution

```
IMAGE tensor (float32 [B,H,W,C])
    ↓ tensor_to_base64()
base64 PNG string
    ↓ GrokClient.vision()
    ↓ POST https://api.x.ai/v1/chat/completions
réponse texte
    ↓ return (result,)
STRING output
```

### Conversion image : `tensor_to_base64(image_tensor)`

Les tenseurs IMAGE dans ComfyUI ont la forme `[batch, height, width, channels]`, dtype `float32`, valeurs dans `[0.0, 1.0]`.

```python
img_np = (image_tensor[0].numpy() * 255).astype(np.uint8)  # 1er item du batch
pil_img = Image.fromarray(img_np, mode="RGB")
buffer = io.BytesIO()
pil_img.save(buffer, format="PNG")
return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/png"
```

- Seul le **premier item du batch** est utilisé (index 0).
- Format de sortie : **PNG** (lossless, préserve les détails fins pour l'analyse).
- L'image n'est jamais écrite sur disque — elle transite entièrement en mémoire.

### Inputs

| Nom | Type ComfyUI | Défaut | Notes |
|-----|-------------|--------|-------|
| `api_key` | STRING, password=True | `"xai-..."` | Masqué dans l'UI |
| `image` | IMAGE | — | Tenseur ComfyUI standard |
| `model` | LIST | `grok-4-1-fast-non-reasoning` | Dropdown, voir §6 |
| `system_prompt` | STRING multiline | Prompt NSFW dataset expert | Contexte du modèle |
| `user_message` | STRING multiline | `"Describe this image..."` | Instruction spécifique |
| `temperature` | FLOAT [0.0–2.0] | `0.7` | Créativité de la réponse |
| `max_tokens` | INT [64–4096] | `1024` | Longueur max du prompt généré |

### Output

| Nom | Type | Contenu |
|-----|------|---------|
| `prompt` | STRING | Prompt généré par Grok, ou message d'erreur `[GrokVisionNode Error ...]` |

---

## 5. GrokPromptBuilderNode — `nodes/grok_prompt_builder_node.py`

### Fonction

Prend jusqu'à 5 listes d'exemples + une instruction guide, génère N nouveaux prompts dans le même style.

### Pipeline d'exécution

```
examples_1..5 (STRING multilines)
    ↓ build_messages()  →  extrait les lignes non-vides, construit system + user
messages list
    ↓ GrokClient.chat()
    ↓ POST https://api.x.ai/v1/chat/completions
réponse texte brute
    ↓ parsing : split \n, strip, nettoyage des préfixes
clean_prompts list
    ↓ output_separator.join(clean_prompts)
(prompts_text STRING, prompt_count INT)
```

### Construction des messages : `build_messages(guide, examples_lists, num_prompts)`

**System prompt** (fixe) :
> "You are an expert prompt engineer for AI image generation. Your task is to generate new prompts that match the style, vocabulary, and structure of the provided examples. Output ONLY the prompts, one per line, no numbering, no explanations."

**User message** (dynamique) :
```
Here are example prompts to use as style reference:

- <exemple extrait de examples_1>
- <exemple extrait de examples_1>
- <exemple extrait de examples_2>
...

Guide/theme for the new prompts:
<guide_prompt>

Generate exactly <num_prompts> new prompts following the same style as the examples above.
Output only the prompts, one per line.
```

Chaque liste d'exemples est parsée ligne par ligne (`split("\n")`) — chaque ligne non-vide devient un exemple indépendant préfixé `- `.

### Parsing de la réponse

Le modèle retourne les prompts séparés par des sauts de ligne. Post-traitement appliqué :

1. Split sur `\n`, suppression des lignes vides.
2. Suppression des préfixes de liste courants : `1. `, `1) `, `1: `, `- `, `* `, `• `.
3. Join des prompts nettoyés avec `output_separator`.

Le `prompt_count` reflète le nombre de prompts **réellement parsés**, pas `num_prompts`.

### Inputs

| Nom | Type ComfyUI | Défaut | Notes |
|-----|-------------|--------|-------|
| `api_key` | STRING, password=True | `"xai-..."` | Masqué dans l'UI |
| `model` | LIST | `grok-4-1-fast-non-reasoning` | Dropdown, voir §6 |
| `guide_prompt` | STRING multiline | `"Generate photorealistic..."` | Thème / instruction |
| `examples_1` | STRING multiline | `""` | **Requis** |
| `num_prompts` | INT [1–50] | `10` | Nombre de prompts demandés |
| `temperature` | FLOAT [0.0–2.0] | `1.0` | Valeur haute = plus de variété |
| `max_tokens` | INT [256–8192] | `2048` | Doit être suffisant pour N prompts |
| `output_separator` | STRING | `"\n---\n"` | Séparateur entre prompts dans l'output |
| `examples_2..5` | STRING multiline (optional) | `""` | Non requis, ignorés si vides |

### Outputs

| Nom | Type | Contenu |
|-----|------|---------|
| `prompts_text` | STRING | Tous les prompts joints par `output_separator` |
| `prompt_count` | INT | Nombre de prompts parsés dans la réponse |

---

## 6. Modèles disponibles

Les deux nodes exposent le même dropdown :

| Model ID | Type | Contexte | Prix input / output (per 1M tokens) |
|----------|------|---------|--------------------------------------|
| `grok-4-1-fast-non-reasoning` | Standard | 2M tokens | $0.20 / $0.50 |
| `grok-4-1-fast-reasoning` | Reasoning (chain-of-thought) | 2M tokens | $0.20 / $0.50 |

**Différence pratique** :
- **non-reasoning** : réponse directe, plus rapide. Suffisant pour la description d'image et la génération de prompts stylistiques.
- **reasoning** : le modèle "réfléchit" avant de répondre (tokens de raisonnement internes, non visibles). Utile si la tâche nécessite de l'inférence complexe sur l'image ou une cohérence stylistique très fine sur les exemples.

Les deux modèles supportent les **entrées image** (multimodal vision).

---

## 7. Format de l'API xAI

L'API xAI est **compatible OpenAI** — même format de requête/réponse que `openai.chat.completions.create`.

**Endpoint** : `POST https://api.x.ai/v1/chat/completions`

**Headers** :
```
Authorization: Bearer xai-...
Content-Type: application/json
```

**Réponse** (format standard) :
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      }
    }
  ]
}
```

Le contenu est extrait via `response.json()["choices"][0]["message"]["content"]`.

---

## 8. Installation sur RunPod

```bash
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/moltowski/grok-api-node
pip install -r grok-api-node/requirements.txt
```

Redémarrer ComfyUI. Les nodes apparaissent dans la catégorie **Grok API** du menu Add Node.

**Mise à jour** :
```bash
cd /workspace/ComfyUI/custom_nodes/grok-api-node
git pull
```

---

## 9. Contraintes et limitations

| Contrainte | Détail |
|-----------|--------|
| **Batch** | Seul le premier item du batch IMAGE est traité dans GrokVisionNode |
| **Synchrone** | Les deux nodes bloquent l'exécution du workflow pendant l'appel API (pas de streaming) |
| **Timeout** | Fixé à 120s — les modèles reasoning peuvent prendre 10-30s sur des prompts complexes |
| **Erreurs silencieuses** | Les erreurs API sont retournées comme STRING dans l'output, pas comme exception — le workflow continue |
| **API key** | La clé est passée en clair dans le fichier workflow JSON sauvegardé par ComfyUI — ne pas versionner ces fichiers |
| **Pillow requis** | Nécessaire pour la conversion tensor → PNG dans GrokVisionNode. Déjà présent dans tout environnement ComfyUI standard |
