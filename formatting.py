import logging
from typing import List, Dict, Any
from datetime import datetime
from type import AgentState

logger = logging.getLogger(__name__)

def log_tool_execution(tool_name: str, result_count: int, step_id: int = None):
    """Log tool execution with minimal information."""
    step_info = f"Step {step_id}: " if step_id else ""
    print(f"{step_info}{tool_name} → {result_count} results")

def flatten_results(results):
    """Recursively flatten a list of results (to handle nested lists)."""
    flat = []
    for item in results:
        if isinstance(item, list):
            flat.extend(flatten_results(item))
        else:
            flat.append(item)
    return flat

def convert_datetimes(obj):
    """Recursively convert all datetime objects in a structure to ISO strings."""
    if isinstance(obj, dict):
        return {k: convert_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetimes(v) for v in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

def format_search_results(state: AgentState, results: List[Dict]) -> Dict:
    """
    Format search results in a schema-agnostic way.
    """
    try:
        # Use schema fields if available, else infer from results
        schema_fields = state.get('extracted_fields', {})
        if not schema_fields and results and isinstance(results[0], dict):
            schema_fields = {k: type(v).__name__ for k, v in results[0].items() if not k.startswith('_')}
        summary_parts = ["# Search Results\n"]
        summary_parts.append(f"Found {len(results)} matching items.\n")
        for i, result in enumerate(results[:5], 1):
            if not isinstance(result, dict):
                continue
            summary_parts.append(f"## {i}. Item")
            for field in schema_fields:
                value = result.get(field, '')
                summary_parts.append(f"- {field.title()}: {value}")
            # Add score if available
            if '_search_metadata' in result and 'score' in result['_search_metadata']:
                score = result['_search_metadata']['score']
                summary_parts.append(f"- Relevance score: {score:.3f}")
            summary_parts.append("")
        return {
            "summary": "\n".join(summary_parts),
            "status": "success",
            "results_count": len(results),
            "results": [r for r in results[:20] if isinstance(r, dict)],
            "metadata": {
                "result_type": "items",
                "total_results": len(results)
            }
        }
    except Exception as e:
        logger.error(f"Error formatting search results: {str(e)}", exc_info=True)
        return {"summary": "Error formatting results.", "status": "error", "results_count": 0, "results": []}

def format_group_results(state: AgentState, group_results: List[Dict]) -> Dict:
    """
    Format group_by results into a comprehensive, schema-agnostic summary.
    """
    try:
        if not group_results:
            return {
                "summary": "No grouping results found.",
                "status": "no_results",
                "results_count": 0,
                "results": [],
                "metadata": {"result_type": "groups"}
            }
        # Use schema fields if available, else infer from group results
        schema_fields = state.get('extracted_fields', {})
        if not schema_fields and group_results and isinstance(group_results[0], dict):
            # Try to infer from items in the first group
            items = group_results[0].get('items', [])
            if items and isinstance(items[0], dict):
                schema_fields = {k: type(v).__name__ for k, v in items[0].items() if not k.startswith('_')}
        summary_parts = [f"# Top Groups by Number of Items\n"]
        for i, group in enumerate(group_results[:10], 1):
            count = group.get('item_count', group.get('article_count', 0))
            summary_parts.append(f"## {i}. {group.get('group_name', 'N/A')} ({count} items)")
            # Show all group-level fields except 'items', 'field_values', 'latest_date', 'group_name', 'item_count'
            for k, v in group.items():
                if k not in ['items', 'field_values', 'latest_date', 'group_name', 'item_count', 'article_count']:
                    summary_parts.append(f"- {k.title()}: {v}")
            # Add field diversity if available
            if group.get('field_values'):
                summary_parts.append("   - Field diversity:")
                for field, values in group['field_values'].items():
                    if len(values) <= 5:
                        summary_parts.append(f"     • {field}: {', '.join(values)}")
                    else:
                        summary_parts.append(f"     • {field}: {', '.join(list(values)[:3])}... ({len(values)} total)")
            # Add latest date if available
            if group.get('latest_date'):
                latest_date = group['latest_date']
                if not isinstance(latest_date, str):
                    latest_date = latest_date.isoformat()
                summary_parts.append(f"   - Latest date: {latest_date.split('T')[0]}")
            # Add sample items if available
            if group.get('items'):
                summary_parts.append("   - Sample items:")
                for item in group['items'][:3]:
                    item_desc = []
                    for field in schema_fields:
                        value = item.get(field, '')
                        item_desc.append(f"{field}: {value}")
                    if item_desc:
                        summary_parts.append(f"     • {', '.join(item_desc[:3])}")
            summary_parts.append("")
        return {
            "summary": "\n".join(summary_parts),
            "status": "success",
            "results_count": len(group_results),
            "results": group_results,
            "metadata": {
                "result_type": "groups",
                "total_groups": len(group_results)
            }
        }
    except Exception as e:
        logger.error(f"Error formatting group results: {str(e)}", exc_info=True)
        return {"summary": "Error formatting group results.", "status": "error", "results_count": 0, "results": []}
