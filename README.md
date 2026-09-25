# Mon RAG — Érythermalgie

Cette application Streamlit permet d'importer des PDF dans une base OpenAI,
de rechercher des passages et de rédiger une réponse avec
DeepSeek V4.1 Flash via Baseten. Les passages consultés peuvent être ouverts
sous la réponse. Les questions sont traitées indépendamment.

## 1. Mettre les fichiers sur GitHub

Créer un dépôt GitHub **privé**, par exemple `rag-erythermalgie`.
Ajouter `app.py` et `requirements.txt` à la racine du dépôt.
Ce README peut également y être ajouté.

Les PDF s'ajouteront depuis l'application. Les clés API se renseignent dans
les Secrets de Streamlit : ne pas les ajouter au dépôt GitHub.

## 2. Ouvrir l'écran de déploiement Streamlit

Sur l'écran « What would you like to do? », choisir la carte GitHub,
puis « Deploy now ». Sélectionner le dépôt et sa branche (généralement `main`).
Pour « Main file path », saisir `app.py`.

Dans « Advanced settings » → « Secrets », saisir les deux clés sous cette forme :

```toml
OPENAI_API_KEY = "TA_CLE_OPENAI"
BASETEN_API_KEY = "TA_NOUVELLE_CLE_BASETEN"
```

Utiliser une nouvelle clé Baseten si la précédente a été partagée, et révoquer
l'ancienne dans Baseten. Ce projet ne contient aucune clé réelle.

Si une base existe déjà et que son identifiant est connu, ajouter aussi :

```toml
VECTOR_STORE_ID = "vs_IDENTIFIANT_DE_TA_BASE"
```

Cette troisième ligne est facultative. Sans elle, l'application retrouve les bases
nommées `rag_erythermalgie` dans le projet OpenAI associé à la clé. S'il y en a
plusieurs, elle propose de choisir. Si aucune n'existe, le bouton « Créer ma base
documentaire » permet d'en créer une.

Enregistrer les Secrets puis déployer. Un dépôt privé donne une application
privée par défaut. Vérifier dans « Settings » → « Sharing » que « Only specific
people can view this app » est sélectionné. Community Cloud autorise une
application privée par compte ; si cette place est déjà occupée, ne pas rendre
les documents publics pour contourner cette limite.

## 3. Utiliser le RAG

1. Choisir ses PDF dans « Ajouter des études ».
2. Cliquer sur « Importer les PDF ». Le statut `completed` signifie que le PDF
   est prêt pour la recherche.
3. Choisir éventuellement un ou plusieurs articles pour limiter la recherche.
4. Écrire une question puis cliquer sur « Poser la question ».
5. Ouvrir les passages sous la réponse pour vérifier les sources.

Les PDF et leur index sont conservés chez OpenAI. Redémarrer Streamlit ne recrée
pas la base. Il faut continuer à utiliser le même projet OpenAI. L'application
ne définit pas d'expiration sur les bases qu'elle crée ; une politique
d'expiration déjà présente sur une base existante reste applicable.

Les doublons sont repérés par leur nom. Pour ajouter une nouvelle version d'un
PDF déjà présent, lui donner un nom différent. L'application ne supprime ni ne
remplace automatiquement les documents existants.

Le RAG examine jusqu'à huit résultats de recherche par question : il ne garantit
pas une revue exhaustive de tous les articles. Les citations du texte sont
rédigées par le modèle ; les panneaux de sources affichent les passages réellement
renvoyés par la recherche. Le fil des réponses n'est pas archivé durablement.

## Exécution locale facultative

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

En local, renseigner les mêmes paramètres dans `.streamlit/secrets.toml`, en
gardant ce fichier hors de GitHub.

## Vérification

Vérifié localement le 25 septembre 2026 avec Streamlit AppTest et des API
simulées : création explicite de la base, reprise dans une nouvelle session,
importation sans doublon, filtre par PDF, transmission des passages à Baseten,
affichage des sources et absence de nouvelle génération au simple réaffichage.
Aucune clé réelle utilisée et aucun appel API payant effectué. Les accès réels
aux deux services seront à vérifier après configuration des Secrets.

## Documentation officielle consultée le 25 septembre 2026

- Streamlit : https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Secrets : https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- Accès privé : https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
- Recherche OpenAI : https://developers.openai.com/api/docs/guides/retrieval
- Baseten : https://docs.baseten.co/inference/model-apis/overview
- DeepSeek V4.1 Flash : https://www.baseten.co/library/deepseek-v41-flash/
