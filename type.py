from typing import Dict, Any, List, TypedDict

class AgentState(TypedDict):
    original_query: str
    plan: Dict[str, Any]
    current_step: int
    intermediate_results: List[Any]
    final_results: List[Dict]
    summary: str