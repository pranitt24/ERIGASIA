"""Course configuration for VIT INFT:
PEIT21T - Artificial Intelligence (Prof. Shashikant Mahajan & Prof. Suja Jaychandran).
Mapped directly to AAP Sections 1.b, 1.d (CO-PO Matrix), and 2.a (Modules 1-6)."""

COURSE = {
    "id": "PEIT21T",
    "name": "Artificial Intelligence",
    "cos": [
        {
            "co_id": "CO1",
            "title": "Explain fundamental concepts, characteristics, and intelligent agents (PEAS)",
            "po_ids": ["PO1", "PO2", "PO12"],
        },
        {
            "co_id": "CO2",
            "title": "Apply uninformed (BFS/DFS), informed (A*), adversarial (Minimax/Alpha-Beta) & local search",
            "po_ids": ["PO1", "PO2", "PO3", "PO4"],
        },
        {
            "co_id": "CO3",
            "title": "Demonstrate knowledge representation using Propositional/First-Order Logic & Belief Networks",
            "po_ids": ["PO1", "PO2", "PO3", "PO5"],
        },
        {
            "co_id": "CO4",
            "title": "Analyze planning approaches (State-Space, Hierarchical) and learning models (PAC)",
            "po_ids": ["PO1", "PO2", "PO3", "PO4", "PO12"],
        },
        {
            "co_id": "CO5",
            "title": "Evaluate AI applications in healthcare, cybersecurity & ethical/societal implications",
            "po_ids": ["PO1", "PO6", "PO7", "PO8", "PO12"],
        },
        {
            "co_id": "CO6",
            "title": "Design AI-based solutions and Agentic AI models for practical real-world problems",
            "po_ids": ["PO1", "PO2", "PO3", "PO4", "PO5", "PO9", "PO10", "PO11", "PO12"],
        },
    ],
    "modules": [
        {"name": "M1: Foundations of AI & Generative AI", "co_id": "CO1", "weight": 1.0},
        {"name": "M2: Intelligent Agents, PEAS & Multi-Agent Systems", "co_id": "CO1", "weight": 0.95},
        {"name": "M3: Uninformed, Heuristic (A*) & Adversarial Search (Minimax)", "co_id": "CO2", "weight": 1.0},
        {"name": "M4: Propositional/First-Order Logic, Resolution & Belief Networks", "co_id": "CO3", "weight": 1.0},
        {"name": "M5: State-Space Planning, Hierarchical Planning & PAC Learning", "co_id": "CO4", "weight": 1.0},
        {"name": "M6: Responsible AI, Ethics, Fairness & Real-World Applications", "co_id": "CO5", "weight": 1.0},
    ],
}
CO_TITLES = {c["co_id"]: c["title"] for c in COURSE["cos"]}
PO_OF = {c["co_id"]: c["po_ids"] for c in COURSE["cos"]}
CO_IDS = list(CO_TITLES)