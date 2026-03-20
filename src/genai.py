from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import (
    GENAI_CACHE_PATH,
    GENAI_LOG_PATH,
    GENAI_MODEL_NAME,
    GENAI_PROVIDER,
    GEMINI_API_KEY,
    LOCAL_GENAI_MODEL_NAME,
    ensure_directories,
)
from .data_pipeline import clean_text


@dataclass
class GenerationSettings:
    max_new_tokens: int = 140
    temperature: float = 0.2


class LocalGenAI:
    def __init__(
        self,
        model_name: str = GENAI_MODEL_NAME,
        cache_path: Path = GENAI_CACHE_PATH,
        log_path: Path = GENAI_LOG_PATH,
        allow_model_loading: bool = True,
        provider: str = GENAI_PROVIDER,
        api_key: str = GEMINI_API_KEY,
        local_model_name: str = LOCAL_GENAI_MODEL_NAME,
    ):
        ensure_directories()
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.log_path = Path(log_path)
        self.allow_model_loading = allow_model_loading
        self.provider = provider
        self.api_key = api_key
        self.local_model_name = local_model_name

        self.mode = "template_fallback"
        self._pipeline = None
        self._cache = self._load_cache()
        self._gemini_client = None
        self._gemini_types = None

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        with open(self.cache_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_cache(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as handle:
            json.dump(self._cache, handle, indent=2, ensure_ascii=False)

    def _system_instruction(self, kind: str) -> str:
        if kind == "plan":
            return (
                "Tu es un assistant pedagogique. Tu proposes uniquement un plan d'action concret, "
                "sans inventer d'experiences non mentionnees."
            )
        if kind == "bio":
            return (
                "Tu es un coach de carriere. Tu rediges une bio courte, serieuse et credible, "
                "sans exagérer le niveau reel du candidat."
            )
        if kind == "enrich_profile":
            return (
                "Tu aides un etudiant a reformuler une description tres courte en texte un peu plus explicite, "
                "sans inventer de missions majeures ni de technologies jamais citees."
            )
        return "Tu aides a produire un texte clair, court et pertinent."

    def _build_prompt(self, kind: str, context: dict[str, Any]) -> str:
        strengths = ", ".join(context.get("strengths", []))
        gaps = ", ".join(context.get("gaps", []))
        top_jobs = ", ".join(f"{item['job_title']} ({item['final_score']:.2f})" for item in context.get("top_jobs", []))
        blocks = ", ".join(f"{item['block_name']}={item['block_score']:.2f}" for item in context.get("block_scores", []))

        if kind == "plan":
            return (
                "Redige en francais un plan de progression tres concret pour un etudiant. "
                f"Candidat: {context.get('candidate_name')}. "
                f"Role vise: {context.get('target_role')}. "
                f"Score global: {context.get('overall_score')}. "
                f"Metiers recommandes: {top_jobs}. "
                f"Points forts: {strengths}. "
                f"Competences a renforcer: {gaps}. "
                f"Scores par bloc: {blocks}. "
                "Donne 3 priorites, 3 actions pratiques et un mini planning sur 30 jours."
            )

        if kind == "bio":
            return (
                "Redige une bio professionnelle courte en francais avec un ton etudiant mais serieux. "
                f"Candidat: {context.get('candidate_name')}. "
                f"Role vise: {context.get('target_role')}. "
                f"Score global: {context.get('overall_score')}. "
                f"Metiers recommandes: {top_jobs}. "
                f"Points forts: {strengths}. "
                f"Competences a renforcer: {gaps}. "
                "Longueur attendue : entre 60 et 90 mots."
            )

        if kind == "enrich_profile":
            return (
                "Reformule et enrichis legerement ce profil tres court pour aider une analyse semantique. "
                "Conserve uniquement des informations plausibles et deja suggerees. "
                "Retourne un seul paragraphe de 40 a 70 mots.\n\n"
                f"Nom: {context.get('candidate_name', 'Profil etudiant')}\n"
                f"Role vise: {context.get('target_role', 'Non precise')}\n"
                f"Outils cites: {', '.join(context.get('tools', []))}\n"
                f"Blocs cites: {', '.join(context.get('focus_blocks', []))}\n"
                f"Texte brut: {context.get('raw_text', '')}"
            )

        return clean_text(str(context))

    def _template_output(self, kind: str, context: dict[str, Any]) -> str:
        strengths = context.get("strengths", [])[:3]
        gaps = context.get("gaps", [])[:3]
        best_job = context.get("top_jobs", [{"job_title": context.get("target_role", "role cible")}])[0]["job_title"]
        name = context.get("candidate_name", "Le candidat")

        if kind == "plan":
            return (
                "Priorite 1 : consolider les competences deja visibles comme "
                + ", ".join(strengths[:2])
                + ".\n"
                + "Priorite 2 : progresser sur "
                + ", ".join(gaps[:2])
                + " pour mieux viser le poste de "
                + best_job
                + ".\n"
                + "Priorite 3 : documenter chaque etape du projet et mesurer les resultats.\n"
                + "Semaine 1 : reprendre le nettoyage et la structuration des donnees.\n"
                + "Semaine 2 : enrichir le tableau de bord avec des filtres et une lecture metier.\n"
                + "Semaine 3 : travailler le matching semantique et comparer les scores par bloc.\n"
                + "Semaine 4 : generer une synthese finale et presenter les limites de la solution."
            )

        if kind == "bio":
            return (
                f"{name} developpe un profil oriente {best_job} avec une base solide sur "
                + ", ".join(strengths[:2])
                + ". Dans ce projet, le candidat a montre une capacite a structurer des donnees, analyser des textes et produire des recommandations utiles pour l orientation metier. La suite du travail consiste surtout a renforcer "
                + ", ".join(gaps[:2] or ["la regularite des tests"])
                + " afin de gagner en precision et en autonomie sur des cas reels."
            )

        raw_text = clean_text(context.get("raw_text", ""))
        if raw_text:
            return (
                "Profil reformule : "
                + raw_text
                + " L'objectif est surtout de montrer les competences deja mobilisees dans un cadre etudiant ou de projet."
            )
        return "Profil etudiant en cours de structuration avec des experiences encore peu detaillees."

    def _load_gemini_client(self) -> Any:
        if self._gemini_client is not None:
            return self._gemini_client
        if not self.api_key:
            return None
        try:
            from google import genai
            from google.genai import types

            self._gemini_client = genai.Client(api_key=self.api_key)
            self._gemini_types = types
            return self._gemini_client
        except Exception:
            self._gemini_client = None
            self._gemini_types = None
            return None

    def _load_pipeline(self) -> Any:
        if not self.allow_model_loading:
            return None
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text2text-generation",
                model=self.local_model_name,
                tokenizer=self.local_model_name,
                device=-1,
            )
            self.mode = "local_transformers"
        except Exception:  # pragma: no cover
            self._pipeline = None
            self.mode = "template_fallback"
        return self._pipeline

    def _cache_key(
        self,
        kind: str,
        prompt: str,
        settings: GenerationSettings,
        system_instruction: str,
    ) -> str:
        payload = json.dumps(
            {
                "kind": kind,
                "prompt": prompt,
                "settings": asdict(settings),
                "provider": self.provider,
                "model_name": self.model_name,
                "local_model_name": self.local_model_name,
                "system_instruction": system_instruction,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _append_log(self, kind: str, settings: GenerationSettings, word_count: int, cache_hit: bool) -> None:
        file_exists = self.log_path.exists()
        with open(self.log_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["created_at", "kind", "mode", "temperature", "max_new_tokens", "word_count", "cache_hit"],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "kind": kind,
                    "mode": self.mode,
                    "temperature": settings.temperature,
                    "max_new_tokens": settings.max_new_tokens,
                    "word_count": word_count,
                    "cache_hit": int(cache_hit),
                }
            )

    def _generate_with_gemini(
        self,
        prompt: str,
        settings: GenerationSettings,
        system_instruction: str,
    ) -> str:
        client = self._load_gemini_client()
        if client is None or self._gemini_types is None:
            raise RuntimeError("Gemini indisponible")

        config_kwargs = {
            "system_instruction": system_instruction,
            "temperature": settings.temperature,
            "max_output_tokens": settings.max_new_tokens,
        }

        try:
            config = self._gemini_types.GenerateContentConfig(**config_kwargs)
        except TypeError:
            config_kwargs.pop("max_output_tokens", None)
            config = self._gemini_types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        text = clean_text(getattr(response, "text", "") or "")
        if not text:
            raise RuntimeError("Gemini a renvoye une reponse vide")
        self.mode = "gemini_api"
        return text

    def _generate_with_local_model(self, prompt: str, settings: GenerationSettings) -> str:
        pipeline_instance = self._load_pipeline()
        if pipeline_instance is None:
            raise RuntimeError("Modele local indisponible")
        outputs = pipeline_instance(
            prompt,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            do_sample=settings.temperature > 0.05,
        )
        text = clean_text(outputs[0]["generated_text"].strip())
        if not text:
            raise RuntimeError("Generation locale vide")
        self.mode = "local_transformers"
        return text

    def _wants_gemini(self) -> bool:
        return self.provider == "gemini" or self.model_name.startswith("gemini")

    def status(self) -> dict[str, Any]:
        gemini_ready = bool(self.api_key and self._load_gemini_client() is not None)
        if gemini_ready:
            message = f"Gemini pret avec le modele {self.model_name}."
        elif self._wants_gemini():
            message = "Gemini non disponible pour le moment, fallback local active si necessaire."
        else:
            message = f"Mode local prioritaire avec {self.local_model_name}."
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "gemini_ready": gemini_ready,
            "cache_entries": len(self._cache),
            "message": message,
        }

    def cache_stats(self) -> dict[str, int]:
        return {"entries": len(self._cache)}

    def _once_cache_key(self, kind: str, request_key: str, system_instruction: str) -> str:
        payload = json.dumps(
            {
                "mode": "single_call",
                "kind": kind,
                "request_key": request_key,
                "provider": self.provider,
                "model_name": self.model_name,
                "system_instruction": system_instruction,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def generate_from_prompt(
        self,
        kind: str,
        prompt: str,
        settings: GenerationSettings | None = None,
        system_instruction: str | None = None,
        fallback_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or GenerationSettings()
        system_instruction = system_instruction or self._system_instruction(kind)
        cache_key = self._cache_key(kind, prompt, settings, system_instruction)

        if cache_key in self._cache:
            record = self._cache[cache_key]
            self.mode = record.get("mode", self.mode)
            self._append_log(kind, settings, len(record["text"].split()), cache_hit=True)
            return {"text": record["text"], "cache_hit": True, "mode": self.mode, "prompt": prompt}

        generated_text = ""
        if self._wants_gemini():
            try:
                generated_text = self._generate_with_gemini(prompt, settings, system_instruction)
            except Exception:
                generated_text = ""

        if not generated_text:
            try:
                generated_text = self._generate_with_local_model(prompt, settings)
            except Exception:
                generated_text = ""

        if not generated_text:
            self.mode = "template_fallback"
            generated_text = self._template_output(kind, fallback_context or {"raw_text": prompt})

        self._cache[cache_key] = {
            "text": generated_text,
            "mode": self.mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "settings": asdict(settings),
        }
        self._save_cache()
        self._append_log(kind, settings, len(generated_text.split()), cache_hit=False)
        return {"text": generated_text, "cache_hit": False, "mode": self.mode, "prompt": prompt}

    def generate(self, kind: str, context: dict[str, Any], settings: GenerationSettings | None = None) -> dict[str, Any]:
        prompt = self._build_prompt(kind, context)
        return self.generate_from_prompt(
            kind,
            prompt,
            settings=settings,
            system_instruction=self._system_instruction(kind),
            fallback_context=context,
        )

    def generate_once(
        self,
        kind: str,
        context: dict[str, Any],
        request_key: str,
        settings: GenerationSettings | None = None,
    ) -> dict[str, Any]:
        settings = settings or GenerationSettings()
        system_instruction = self._system_instruction(kind)
        once_key = self._once_cache_key(kind, request_key, system_instruction)
        if once_key in self._cache:
            record = self._cache[once_key]
            self.mode = record.get("mode", self.mode)
            self._append_log(kind, settings, len(record["text"].split()), cache_hit=True)
            return {
                "text": record["text"],
                "cache_hit": True,
                "mode": self.mode,
                "prompt": record.get("prompt", ""),
                "single_call_locked": True,
            }

        result = self.generate(kind, context, settings=settings)
        self._cache[once_key] = {
            "text": result["text"],
            "mode": result["mode"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "settings": asdict(settings),
            "prompt": result["prompt"],
        }
        self._save_cache()
        result["single_call_locked"] = True
        return result

    def enrich_submission_if_needed(self, submission: dict[str, Any], min_words: int = 18) -> dict[str, Any]:
        texts = [clean_text(submission.get(field, "")) for field in ("project_text", "dashboard_text", "genai_text")]
        raw_text = " ".join(text for text in texts if text).strip()
        if len(raw_text.split()) >= min_words:
            return submission

        context = {
            "candidate_name": submission.get("candidate_name", "Profil etudiant"),
            "target_role": submission.get("target_role", "Non precise"),
            "tools": submission.get("tools", []),
            "focus_blocks": submission.get("focus_blocks", []),
            "raw_text": raw_text or "Profil encore tres court",
        }
        result = self.generate("enrich_profile", context, GenerationSettings(max_new_tokens=120, temperature=0.3))
        enriched = dict(submission)
        enriched["profile_enrichment"] = result["text"]
        enriched["profile_enrichment_meta"] = {
            "mode": result["mode"],
            "cache_hit": result["cache_hit"],
            "original_word_count": len(raw_text.split()),
        }
        return enriched
