# Risques, biais et conformité RGPD — AISCA (C5.2 / C5.3)

Ce document recense les **risques** de la brique IA générative (hallucination, biais, sécurité,
données personnelles) et les **mesures de mitigation** réellement implémentées dans le projet.

> AISCA encode des compétences déclarées par un candidat, recommande des métiers et **génère** un
> plan de progression et une bio. Les enjeux portent donc sur des **données potentiellement
> personnelles** (profil candidat) et sur la **fiabilité du texte généré**.

---

## 1. Risques liés à la génération (hallucination, biais)

| Risque | Description | Mitigation implémentée |
|---|---|---|
| **Hallucination** | Le LLM invente des compétences/expériences non déclarées | System prompts **contraignants** (« sans inventer d'expériences non mentionnées », « sans exagérer le niveau réel ») dans [`genai.py`](../src/genai.py) ; évaluation `coverage` qui mesure l'ancrage du texte sur le contexte réel |
| **Sur-promesse** | Bio trop flatteuse vs niveau réel | Prompt bio explicite (« ton étudiant mais sérieux », bornes 60–90 mots) + score `length_score` |
| **Biais de recommandation** | Toujours recommander le rôle déclaré par l'utilisateur | Score **découplé du rôle déclaré** (testé par `test_ranking_does_not_depend_on_target_role`) ; le classement ne dépend que des compétences sémantiques |
| **Biais du modèle pré-entraîné** | SBERT / LLM portent des biais d'entraînement | Modèle **généraliste non fine-tuné** assumé ; sortie toujours relue par l'utilisateur (texte d'aide, pas de décision automatique) |
| **Non-déterminisme** | Sorties variables d'une exécution à l'autre | `temperature` basse par défaut (0.2) + **cache** (même profil → même sortie) + verrou **1 appel par soumission** |

### Évaluation systématique des paramètres
Le script [`scripts/param_sweep.py`](../scripts/param_sweep.py) balaie `temperature` × `top_p` et
score chaque sortie (coverage, lisibilité, diversité, score global) → `reports/param_sweep.csv`.
Cela justifie le **réglage retenu en production** plutôt qu'un choix arbitraire (C5.3).

---

## 2. Conformité RGPD (données personnelles)

Le profil candidat (nom, texte libre, compétences) peut constituer une **donnée à caractère
personnel**. Mesures prises :

| Principe RGPD | Mise en œuvre |
|---|---|
| **Minimisation** | Seuls les champs nécessaires au matching sont traités ; aucune donnée sensible (santé, origine…) demandée |
| **Limitation de finalité** | Données utilisées uniquement pour la recommandation métier + génération du plan/bio, pas de profilage tiers |
| **Souveraineté / local-first** | Stockage **local** (JSON/CSV), pas de base cloud ; l'app fonctionne **sans aucun appel externe** (repli local `flan-t5` puis template) |
| **Transparence** | L'onglet Analytics journalise chaque appel GenAI (`genai_log.csv`) : type, mode, cache_hit → traçabilité |
| **Contrôle des transferts** | Si Gemini est activé, **seul le contexte agrégé** (forces, gaps, scores) est envoyé, pas le texte brut intégral non nécessaire ; la clé API reste côté `.env` (jamais versionnée) |
| **Droit à l'effacement** | Les soumissions sont des fichiers individuels supprimables ; bouton de purge du cache GenAI dans la sidebar |
| **Pas de décision automatisée** | Les sorties sont des **aides** (plan, bio) relues par l'humain, pas une décision opposable (art. 22 RGPD) |

### Point d'attention assumé
- Si le provider **Gemini** est activé, les données transitent par un service tiers (Google) :
  à **documenter dans une politique de confidentialité** et à soumettre au **consentement** en
  contexte réel. Le mode **local/template** évite tout transfert et reste la configuration par
  défaut recommandée pour un traitement de données personnelles.

---

## 3. Sécurité

| Risque | Mitigation |
|---|---|
| **Fuite de clé API** | Clé lue depuis `.env` (hors versionnement), jamais en dur dans le code |
| **Injection de prompt** | Surface limitée (champs structurés + texte court) ; system prompt fixe ; sortie non exécutée |
| **Déni de service / coût** | Verrou **1 appel API par soumission** + cache → coût borné, respect du Free Tier |
| **Disponibilité** | Cascade de repli (Gemini → local → template) : l'app ne tombe jamais faute de LLM |

---

## 4. Synthèse pour la soutenance

AISCA traite la GenAI de façon **responsable et défendable** :
- sorties **ancrées** sur le profil réel (prompts contraints + métrique de coverage),
- **classement non biaisé** par le rôle déclaré (prouvé par un test),
- **RGPD by design** : local-first, minimisation, traçabilité, pas de décision automatisée,
- **paramètres évalués** (sweep temperature/top-p) et non choisis au hasard.

Limites restantes (perspectives) : pas de fine-tuning métier, pas de détection automatique
d'hallucination au-delà du coverage lexical, et le mode Gemini nécessiterait un cadrage RGPD
formel (politique de confidentialité + consentement) en production.
