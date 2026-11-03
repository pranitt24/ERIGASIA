import json
from typing import Dict, Any, List, Optional


GENERATION_SYSTEM_PROMPT = (
    "You are Ergasia AI, an outcome-based engineering education question generator. "
    "Given retrieved course context and a target Course Outcome (CO), Bloom's taxonomy level, "
    "and total marks, output ONLY valid JSON matching this exact schema:\n"
    "{\n"
    '  "question_text": "string",\n'
    '  "co_id": "string (e.g. CO1)",\n'
    '  "po_mapping": ["string (e.g. PO1, PO2, PO3)"],\n'
    '  "bloom_level": "string (Remember|Understand|Apply|Analyze|Evaluate|Create)",\n'
    '  "total_marks": integer,\n'
    '  "topic": "string",\n'
    '  "rubric": [\n'
    '    {"criterion": "string", "marks": number, "expected_points": "string"}\n'
    "  ]\n"
    "}"
)


class QuestionGenerator:
    """
    Stage 3: Generates CO-PO mapped, rubric-backed questions calibrated to a student's BKT mastery.
    """
    def __init__(self, llm_engine, rag_retriever: Optional[Any] = None):
        self.llm = llm_engine
        self.retriever = rag_retriever

    def retrieve_context(self, stage3_payload: Dict[str, Any], topic_hint: str = "") -> Dict[str, Any]:
        """Queries Stage 1 RAG DB if provided; otherwise uses fallback context."""
        if self.retriever is not None and hasattr(self.retriever, "retrieve_for_student"):
            return self.retriever.retrieve_for_student(stage3_payload, query=topic_hint)
        
        # Fallback if testing without an active ChromaDB instance
        return {
            "context_chunks": [
                f"Course material covering {stage3_payload['co_id']} ({topic_hint or 'Core Engineering Concepts'})."
            ],
            "example_questions": [],
            "fallbacks": ["mock_context_used"]
        }

    def generate_for_student(
        self,
        stage3_payload: Dict[str, Any],
        topic: str,
        target_marks: int = 10,
        rubric_criteria_names: Optional[List[str]] = None,
        mode: str = "generation",
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Generates a structured question for one student & CO.
        mode: 'generation' (fine-tuned LoRA) or 'base' (prompted RAG baseline for ablation).
        """
        rag_result = self.retrieve_context(stage3_payload, topic_hint=topic)
        context_str = "\n---\n".join(rag_result.get("context_chunks", []))

        bloom_range = stage3_payload.get("bloom_level_range", ["Apply", "Analyze"])
        target_bloom = bloom_range[-1]  # Aim for upper bound of assigned bucket
        verbs = ", ".join(stage3_payload.get("suggested_verbs", ["apply", "analyze"]))

        criteria_instruction = ""
        if rubric_criteria_names:
            criteria_instruction = (
                f" Ensure the rubric contains exact criteria matching: {rubric_criteria_names} "
                f"and their marks sum strictly to {target_marks}."
            )

        user_prompt = (
            f"Target Course Outcome: {stage3_payload['co_id']}\n"
            f"Topic: {topic}\n"
            f"Student BKT Mastery: {stage3_payload['mastery']:.2f}\n"
            f"Target Bloom's Taxonomy Level: {target_bloom} (Allowed range: {' - '.join(bloom_range)})\n"
            f"Recommended Action Verbs: {verbs}\n"
            f"Total Marks: {target_marks}\n"
            f"{criteria_instruction}\n\n"
            f"Retrieved Grounding Context:\n{context_str}\n\n"
            "Generate the calibrated engineering question, CO-PO mapping, and grading rubric in strict JSON."
        )

        last_response = {}
        for attempt in range(max_retries + 1):
            res = self.llm.generate_json(
                system_prompt=GENERATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                mode=mode,
            )
            last_response = res
            if res["valid_json"] and self._validate_schema(res["data"], target_marks):
                normalized = self._normalize_question_payload(
                    res["data"], stage3_payload, topic, target_marks
                )
                return {
                    "student_id": stage3_payload["student_id"],
                    "bkt_mastery": stage3_payload["mastery"],
                    "assigned_bloom_range": bloom_range,
                    "schema_valid": True,
                    "attempts_used": attempt + 1,
                    "question_data": normalized,
                    "rag_fallbacks": rag_result.get("fallbacks", []),
                }

        # Fallback repair if the model missed a minor field (useful when benchmarking 'base' mode)
        repaired = self._repair_payload(last_response.get("data"), stage3_payload, topic, target_marks)
        return {
            "student_id": stage3_payload["student_id"],
            "bkt_mastery": stage3_payload["mastery"],
            "assigned_bloom_range": bloom_range,
            "schema_valid": False,
            "attempts_used": max_retries + 1,
            "question_data": repaired,
            "raw_output": last_response.get("raw", ""),
        }

    @staticmethod
    def _validate_schema(data: Any, target_marks: int) -> bool:
        if not isinstance(data, dict):
            return False
        required = {"question_text", "co_id", "bloom_level", "total_marks", "rubric"}
        if not required.issubset(data.keys()):
            return False
        if not isinstance(data["rubric"], list) or len(data["rubric"]) == 0:
            return False
        return True

    @staticmethod
    def _normalize_question_payload(
        data: Dict[str, Any], stage3_payload: Dict[str, Any], topic: str, target_marks: int
    ) -> Dict[str, Any]:
        """Ensures total_marks and rubric marks sum cleanly for Stage 4 fairness checks."""
        data["co_id"] = stage3_payload["co_id"]
        data["topic"] = data.get("topic") or topic
        data["total_marks"] = int(target_marks)
        if "po_mapping" not in data or not isinstance(data["po_mapping"], list):
            data["po_mapping"] = ["PO1", "PO2"]

        # Rescale rubric marks proportionally if LLM sum drifted from target_marks
        rubric = data.get("rubric", [])
        current_sum = sum(float(item.get("marks", 0)) for item in rubric)
        if current_sum != target_marks and current_sum > 0:
            running = 0
            for i, item in enumerate(rubric):
                if i == len(rubric) - 1:
                    item["marks"] = round(target_marks - running, 1)
                else:
                    scaled = round((float(item.get("marks", 0)) / current_sum) * target_marks, 1)
                    item["marks"] = scaled
                    running += scaled
        return data

    @staticmethod
    def _repair_payload(
        data: Any, stage3_payload: Dict[str, Any], topic: str, target_marks: int
    ) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        return {
            "question_text": data.get("question_text", f"Explain and apply {topic} concepts."),
            "co_id": stage3_payload["co_id"],
            "po_mapping": data.get("po_mapping", ["PO1", "PO2"]),
            "bloom_level": stage3_payload["bloom_level_range"][0],
            "total_marks": target_marks,
            "topic": topic,
            "rubric": data.get("rubric") or [
                {"criterion": "Conceptual Clarity & Formulation", "marks": target_marks // 2, "expected_points": "Core principles stated clearly."},
                {"criterion": "Application & Analysis", "marks": target_marks - (target_marks // 2), "expected_points": "Complete derivation/solution."}
            ],
        }