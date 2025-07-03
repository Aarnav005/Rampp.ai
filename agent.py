import google.generativeai as genai
import os
from typing import Dict, Optional, List, Any, TypedDict
import json
from dotenv import load_dotenv
from datetime import datetime
import weaviate
from weaviate.classes.query import Filter, GroupBy
import logging
from langgraph.graph import StateGraph, END
import re
from functools import partial
import urllib.parse
from search_tools import SearchTools, normalize_filter_dict, create_groupby_object
from type import AgentState
from query_decomposer import QueryDecomposer
from formatting import log_tool_execution, flatten_results, convert_datetimes, format_search_results, format_group_results
from collections import Counter
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP requests and other noise
logging.getLogger('weaviate').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)



load_dotenv()

# --- TOOL DEFINITIONS ---



# Dynamic tool registry
def get_tool_registry(search_tools):
    return {
        "filter_data": {
            "func": search_tools.filter_data,
            "description": "Unified filtering that works on any data source.",
            "parameters": ["filters"]},
        "group_data": {
            "func": search_tools.group_data,
            "description": "Unified grouping that works on any data structure.",
            "parameters": ["group_field", "limit", "include_items", "max_items_per_group"]},
        "sort_data": {
            "func": search_tools.sort_data,
            "description": "Unified sorting that works on any data structure.",
            "parameters": ["data", "sort_field", "ascending"]},
        "semantic_search": {
            "func": search_tools.semantic_search,
            "description": "Performs a semantic search using the given query or reference article.",
            "parameters": ["query", "limit", "filter_dict", "reference_id", "score_threshold", "group_field", "max_items_per_group"],
        },
    }

# --- QUERY DECOMPOSITION ---



# --- LANGGRAPH AGENT SETUP ---





# Initialize clients and tools
try:
    weaviate_client = weaviate.connect_to_local(
        host="localhost",
        port=8080,  # Matching docker-compose.yml
        grpc_port=50051
    )
    logger.info("Successfully connected to Weaviate")
except Exception as e:
    logger.error(f"Failed to connect to Weaviate: {e}")
    raise



# Node Functions

def validate_and_correct_plan(plan, schema_fields):
    """Validate and correct the execution plan based on schema fields."""
    warnings = []
    try:
        # Ensure schema_fields is a dictionary - handle all edge cases
        if not isinstance(schema_fields, dict):
            logger.warning(f"schema_fields is not a dictionary in validate_and_correct_plan: {type(schema_fields)}, using empty dict")
            schema_fields = {}
        valid_fields = set(schema_fields.keys())
        logical_operators = {'$and', '$or', '$not'}  # These are not field names
        steps = plan.get('steps', [])

        # --- PATCH: Merge semantic_search + group_data into a single semantic_search with group_by ---
        i = 0
        while i < len(steps) - 1:
            step = steps[i]
            next_step = steps[i+1]
            if step['tool'] == 'semantic_search' and next_step['tool'] == 'group_data':
                group_field = next_step['parameters'].get('group_field')
                if group_field:
                    # Merge group_field into semantic_search step
                    step['parameters']['group_field'] = group_field
                    # Optionally merge limit/max_items_per_group
                    if 'limit' in next_step['parameters']:
                        step['parameters']['limit'] = next_step['parameters']['limit']
                    if 'max_items_per_group' in next_step['parameters']:
                        step['parameters']['max_items_per_group'] = next_step['parameters']['max_items_per_group']
                    # Remove the group_data step
                    steps.pop(i+1)
                    warnings.append(f"Merged semantic_search and group_data into a single semantic_search with group_by on '{group_field}'")
                    # Do not increment i, check for further merges
                    continue
            i += 1
        
        for i, step in enumerate(steps):
            try:
                params = step.get('parameters', {})
                # Remove obsolete tool handling
                # Validate group_field
                if 'group_field' in params and params['group_field'] not in valid_fields:
                    warnings.append(f"Step {step['step_id']}: group_field '{params['group_field']}' not in schema, removing group_by.")
                    step['parameters']['group_field'] = None
                # Validate sort_field
                if 'sort_field' in params:
                    if params['sort_field'] not in valid_fields and params['sort_field'] not in ['item_count', 'article_count']:
                        warnings.append(f"Step {step['step_id']}: sort_field '{params['sort_field']}' not in schema, removing sort.")
                        step['parameters']['sort_field'] = None
                # Validate and fix filters
                if 'filters' in params:
                    if isinstance(params['filters'], str):
                        filters_str = params['filters']
                        if filters_str.strip().startswith('$PREV_STEP_RESULT'):
                            continue
                        else:
                            try:
                                import json
                                params['filters'] = json.loads(filters_str)
                            except Exception:
                                logger.warning(f"Step {step['step_id']}: filters is a string but not valid JSON: {filters_str}, skipping filter validation")
                                continue
                    if not isinstance(params['filters'], dict):
                        logger.warning(f"Step {step['step_id']}: filters is not a dictionary: {type(params['filters'])}, skipping filter validation")
                        continue
                    to_remove = []
                    for f in list(params['filters'].keys()):
                        if f not in valid_fields and f not in logical_operators:
                            warnings.append(f"Step {step['step_id']}: filter field '{f}' not in schema, removing filter.")
                            to_remove.append(f)
                    for f in to_remove:
                        del step['parameters']['filters'][f]
                # Validate and fix filter_dict (for semantic_search)
                if 'filter_dict' in params:
                    if not isinstance(params['filter_dict'], dict):
                        logger.warning(f"Step {step['step_id']}: filter_dict is not a dictionary: {type(params['filter_dict'])}, skipping filter_dict validation")
                        continue
                    to_remove = []
                    for f in list(params['filter_dict'].keys()):
                        if f not in valid_fields and f not in logical_operators:
                            warnings.append(f"Step {step['step_id']}: filter_dict field '{f}' not in schema, removing filter.")
                            to_remove.append(f)
                    for f in to_remove:
                        del step['parameters']['filter_dict'][f]
            except Exception as step_error:
                logger.error(f"Error validating step {i}: {step_error}", exc_info=True)
                warnings.append(f"Error validating step {i}: {step_error}")
                
    except Exception as e:
        logger.error(f"Error in validate_and_correct_plan: {e}", exc_info=True)
        warnings.append(f"Error in plan validation: {e}")
        
    return plan, warnings

def is_similarity_search(query: str) -> bool:
    """Detect if the query is a similarity/semantic search using robust NLP heuristics."""
    import re
    from difflib import SequenceMatcher
    
    # Lowercase and strip query
    q = query.lower().strip()
    
    # List of strong semantic intent phrases (word boundaries)
    strong_patterns = [
        r"\bsimilar( to| )?\b",
        r"\brelated( to| )?\b",
        r"\brelevant( to| )?\b",
        r"\bsemantic(ally)?\b",
        r"\bclosest( to| )?\b",
        r"\bnearest( to| )?\b",
        r"\bfind (articles|documents|items) (about|like|related to)\b",
        r"\bfind (similar|related)\b",
        r"\babout\b",
        r"\bdiscuss\b",
        r"\bdescribe\b",
        r"\btell me about\b",
        r"\bwhat is\b",
        r"\bwho is\b",
        r"\bwhat are\b",
        r"\bwho are\b",
        r"\btopic of\b",
        r"\bsubject of\b",
        r"\bmeaning of\b",
        r"\bexplain\b",
        r"\bexplore\b",
        r"\bdiscuss\b",
        r"\bdescribe\b",
        r"\bshow me (articles|missions|documents|items) about\b",
    ]
    # Check for any strong pattern
    for pat in strong_patterns:
        if re.search(pat, q):
            return True
    
    # Fuzzy match for 'about', 'on', 'regarding', 'concerning' as topic indicators
    topic_words = ['about', 'on', 'regarding', 'concerning']
    for word in topic_words:
        if f' {word} ' in q or q.startswith(word + ' '):
            return True
    
    # Fuzzy match for queries that are questions (start with what/who/which/how/why)
    if re.match(r'^(what|who|which|how|why)\b', q):
        return True
    
    # Fuzzy match for queries that are long and descriptive (more likely semantic intent)
    if len(q.split()) > 8 and any(w in q for w in ['about', 'describe', 'discuss', 'explain', 'topic']):
        return True
    
    # Fuzzy ratio for similarity to known semantic prompts
    semantic_examples = [
        'find articles about', 'find documents about', 'find items like', 'find similar', 'find related',
        'tell me about', 'what is', 'who is', 'describe', 'explain', 'discuss', 'topic of', 'subject of'
    ]
    for example in semantic_examples:
        if SequenceMatcher(None, q, example).ratio() > 0.7:
            return True
    
    return False

def fill_missing_semantic_queries(plan, user_query):
    for step in plan.get('steps', []):
        if step.get('tool') == 'semantic_search':
            params = step.get('parameters', {})
            if not params.get('query') or not isinstance(params.get('query'), str) or not params.get('query').strip():
                params['query'] = user_query
    return plan

def decompose_query_node(state: Dict[str, Any], search_tools, query_decomposer, tool_registry) -> Dict[str, Any]:
    """Decompose a user query into a multi-step execution plan using the LLM-based query_decomposer."""
    try:
        schema_fields = search_tools.get_schema_fields()
        if not isinstance(schema_fields, dict):
            logger.warning(f"Retrieved schema_fields is not a dictionary: {type(schema_fields)}, using empty dict")
            schema_fields = {}
    except Exception as e:
        logger.error(f"Failed to get schema fields: {e}")
        schema_fields = {}

    # Use the LLM-based query decomposer
    plan = query_decomposer.decompose(
        state['original_query'],
        search_tools.class_name,
        schema_fields,
        tool_registry
    )
    plan = fill_missing_semantic_queries(plan, state['original_query'])
    plan, plan_warnings = validate_and_correct_plan(plan, schema_fields)

    return {
        'plan': plan,
        'current_step': 1,
        'intermediate_results': [],
        'final_results': [],
        'original_query': state['original_query'],
        'plan_warnings': plan_warnings
    }

def resolve_prev_step_reference(ref, prev_result):
    """
    Resolves references like:
    - $PREV_STEP_RESULT
    - $PREV_STEP_RESULT[0]
    - $PREV_STEP_RESULT[0].field_name
    - $PREV_STEP_RESULT.field_name
    - $PREV_STEP_RESULT[0]['field_name']
    
    This function is schema-agnostic and dynamically handles any field structure.
    """
    if not isinstance(ref, str) or not ref.startswith("$PREV_STEP_RESULT"):
        return ref
    expr = ref[len("$PREV_STEP_RESULT"):]
    value = prev_result
    
    # Handle the case where $PREV_STEP_RESULT.field should access the first item's field
    if expr.startswith('.') and isinstance(value, list) and len(value) > 0:
        # If it's a list and we're trying to access a field directly, get the first item
        value = value[0]
        expr = expr  # Keep the field access part
    
    # Parse [index], .field, or ['field'] accesses
    pattern = r"(\[\d+\]|\.[a-zA-Z_][a-zA-Z0-9_]*|\['[a-zA-Z0-9_]+'\])"
    tokens = re.findall(pattern, expr)
    for token in tokens:
        if token.startswith('[') and token.endswith(']'):
            idx = token[1:-1]
            if idx.isdigit():
                value = value[int(idx)]
            else:
                key = idx.strip("'\"")
                value = value[key]
        elif token.startswith('.'):
            key = token[1:]
            if isinstance(value, dict):
                # Dynamic field mapping based on actual result structure
                if key in value:
                    # Direct field access
                    value = value[key]
                else:
                    # Check if this might be a group_by result and try to map fields
                    available_keys = list(value.keys())
                    
                    # Look for common group_by patterns
                    if 'group_name' in available_keys:
                        # This is likely a group_by result
                        # Try to find what the original field might have been
                        if key in ['author', 'category', 'title', 'content', 'publish_date']:
                            # Common fields that get mapped to group_name in group_by
                            value = value['group_name']
                        else:
                            # For other fields, check if they exist with different names
                            # Common mappings: count -> item_count, etc.
                            if key == 'count' and 'item_count' in available_keys:
                                value = value['item_count']
                            elif key == 'count' and 'article_count' in available_keys:
                                value = value['article_count']
                            else:
                                raise KeyError(f"Field '{key}' not found in result. Available fields: {available_keys}. "
                                             f"This might be because you're trying to access '{key}' after a group_by operation. "
                                             f"For group_by results, the original field name is mapped to 'group_name'.")
                    else:
                        # Not a group_by result, field simply doesn't exist
                        raise KeyError(f"Field '{key}' not found in result. Available fields: {available_keys}.")
            else:
                value = getattr(value, key, None)
    return value

def substitute_prev_step_refs(value, prev_result):
    if isinstance(value, str) and value.startswith("$PREV_STEP_RESULT"):
        resolved = resolve_prev_step_reference(value, prev_result)
        # --- PATCH: Prevent double-nesting of filter structures ---
        if isinstance(resolved, dict) and len(resolved) == 1:
            key, val = list(resolved.items())[0]
            # Only flatten if BOTH keys start with $
            if isinstance(val, dict) and len(val) == 1:
                inner_key, inner_val = list(val.items())[0]
                if key.startswith('$') and inner_key.startswith('$'):
                    logger.warning(f"Double-nested filter detected in reference resolution, flattening: {resolved}")
                    return {inner_key: inner_val}
        return resolved
    elif isinstance(value, dict):
        # --- PATCH: Handle nested dictionaries and prevent double-nesting ---
        result = {}
        for k, v in value.items():
            if isinstance(v, dict) and len(v) == 1:
                inner_key, inner_val = list(v.items())[0]
                if isinstance(inner_val, dict) and len(inner_val) == 1:
                    inner_inner_key, inner_inner_val = list(inner_val.items())[0]
                    # Only flatten if BOTH keys start with $
                    if k.startswith('$') and inner_key.startswith('$') and inner_inner_key.startswith('$'):
                        logger.warning(f"Double-nested filter detected in dict, flattening: {v}")
                        result[k] = {inner_inner_key: inner_inner_val}
                        continue
            result[k] = substitute_prev_step_refs(v, prev_result)
        return result
    elif isinstance(value, list):
        return [substitute_prev_step_refs(v, prev_result) for v in value]
    return value

def execute_tool_node(state: AgentState, tool_registry=None, search_tools=None) -> Dict[str, Any]:
    """Accumulate query objects for each step, and only execute the query at the final step."""
    try:
        current_step_idx = state['current_step'] - 1
        step = state['plan']['steps'][current_step_idx]
        tool_name = step['tool']
        if tool_registry is None:
            if search_tools:
                tool_registry = get_tool_registry(search_tools)
            else:
                raise ValueError("Either tool_registry or search_tools must be provided")
        tool_info = tool_registry.get(tool_name)
        if not tool_info:
            raise ValueError(f"Unknown tool: {tool_name}")
        params = step.get('parameters', {}).copy()

        # Accumulate query objects
        filter_obj = state.get('filter_obj')
        group_by_obj = state.get('group_by_obj')
        sort_params = state.get('sort_params')
        semantic_query = state.get('semantic_query', "")
        is_last_step = (state['current_step'] == len(state['plan']['steps']))
        result = None

        # Only accumulate objects/params for each step
        if tool_name == "filter_data":
            filter_obj = tool_info['func'](**{k: v for k, v in params.items() if k in tool_info['parameters']})
        elif tool_name == "group_data":
            group_by_obj = params  # Save params for client-side grouping
        elif tool_name == "sort_data":
            sort_params = params
        elif tool_name == "semantic_search":
            new_query = step.get('parameters', {}).get('query', "")
            if new_query and (not semantic_query or not semantic_query.strip()):
                semantic_query = new_query

        # --- NEW LOGIC: Push all filters/groupby into near_text if any semantic_search, else into fetch_objects ---
        plan_steps = state['plan']['steps']
        has_semantic = any(s['tool'] == 'semantic_search' for s in plan_steps)

        # Collect all filters
        filter_dicts = []
        group_field = None
        sort_field = None
        ascending = True
        semantic_queries = []
        for s in plan_steps:
            s_tool = s.get('tool')
            s_params = s.get('parameters', {})
            if s_tool == 'filter_data':
                filters = s_params.get('filters', {})
                if filters and isinstance(filters, dict) and len(filters) > 0:
                    filter_dicts.append(filters)
            elif s_tool == 'group_data':
                group_field = s_params.get('group_field')
            elif s_tool == 'sort_data':
                sort_field = s_params.get('sort_field')
                ascending = s_params.get('ascending', True)
            elif s_tool == 'semantic_search':
                q = s_params.get('query', "")
                if q and isinstance(q, str) and q.strip():
                    semantic_queries.append(q.strip())
                # Also check for group_field in semantic_search step
                if s_params.get('group_field'):
                    group_field = s_params['group_field']

        # Combine all filters with logical AND if more than one, ignore empty filters
        if filter_dicts:
            if len(filter_dicts) == 1:
                combined_filter = filter_dicts[0]
            else:
                combined_filter = {"$and": filter_dicts}
        else:
            combined_filter = None

        # Prepare group_by_obj if needed
        group_by_obj_final = create_groupby_object(group_field) if group_field else None

        # Combine all semantic queries with ' AND '
        if semantic_queries:
            semantic_query_final = ' AND '.join(semantic_queries)
        else:
            semantic_query_final = None

        if has_semantic:
            # --- If any semantic_search in plan, push all filters/groupby into near_text ---
            result = search_tools.execute_combined_query(
                query=semantic_query_final,
                filter_obj=search_tools.filter_data(filters=combined_filter) if combined_filter else None,
                group_by_obj=group_by_obj_final,
                score_threshold=0.5
            )
            # Do not do any client-side groupby/filter after
            if sort_field:
                result = search_tools.sort_data(result, sort_field=sort_field, ascending=ascending)
        else:
            # --- If no semantic_search, push all filters/sort into fetch_objects ---
            result = search_tools.objects(
                filter_obj=search_tools.filter_data(filters=combined_filter) if combined_filter else None,
                sort_field=sort_field,
                ascending=ascending,
                number_of_groups=100,
                objects_per_group=100
            )
            # If groupby is present, do client-side groupby
            if group_field:
                result = search_tools.group_data(
                    result,
                    group_field=group_field,
                    limit=None,
                    include_items=False,
                    max_items_per_group=3
                )
        return {
            **state,
            'current_step': len(plan_steps) + 1,  # Jump to end
            'intermediate_results': state['intermediate_results'] + ([result] if result is not None else []),
            'final_results': result if result is not None else state.get('final_results', []),
            'filter_obj': None,
            'group_by_obj': None,
            'sort_params': None,
            'semantic_query': ""
        }
    except Exception as e:
        error_msg = f"Failed to execute step {state.get('current_step', '?')} ({step.get('tool', 'N/A')}): {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {**state, 'error': error_msg}

def llm_tailor_response(user_query: str, raw_output: dict, query_decomposer=None, search_plan: dict = None) -> str:
    """Use the LLM to generate a user-facing summary or answer tailored to the query intent and the raw output."""
    safe_output = convert_datetimes(raw_output)
    plan_str = json.dumps(search_plan, indent=2) if search_plan else "(No plan provided)"
    prompt = f'''
You are an expert research assistant. Given the following user query, the search plan (as JSON), and the raw search output (in JSON), generate a clear, complete, and well-formatted answer that directly addresses the user's intent.

**Important:**
- Enumerate and display **ALL results** in the output, not just a sample or summary. If the result is a list of unique authors, list every author. Do not omit or summarize results.
- For articles, list every article with all available fields (title, author, publish_date, category, etc.).
- For value lists, enumerate every value.
- Use bullet points, tables, or lists for clarity.
- Do not omit or summarize results; show the full output.

User Query:
"""
{user_query}
"""

Search Plan (JSON):
{plan_str}

Raw Output (JSON):
{json.dumps(safe_output, indent=2)}

Full Output:
'''
    # Use the global query_decomposer if not provided
    if query_decomposer is None:
        raise ValueError("query_decomposer must be provided to llm_tailor_response.")
    response = query_decomposer.model.generate_content(prompt)
    return response.text.strip()

def summarize_node(state: AgentState, query_decomposer=None) -> Dict[str, Any]:
    """
    Node to summarize the final search results and provide execution feedback.
    This function is robust to any schema: it will use the last non-empty result for summary, and will choose the right formatter based on the result structure.
    """
    try:
        # Find the last step with results
        last_step_result = None
        last_step_with_results = None
        
        # Check intermediate results first
        if state.get('intermediate_results'):
            for i, result in enumerate(state['intermediate_results']):
                if result and (isinstance(result, list) and len(result) > 0 or not isinstance(result, list)):
                    last_step_result = flatten_results(result)
                    last_step_with_results = i + 1
        
        # If no intermediate results, check final results
        if not last_step_result and state.get('final_results'):
            last_step_result = flatten_results(state['final_results'])
            last_step_with_results = len(state.get('intermediate_results', [])) + 1
        # --- PATCH: If we have final_results, always treat them as coming from the final step ---
        if state.get('final_results'):
            last_step_with_results = len(state.get('plan', {}).get('steps', []))

        # --- PATCH: If group_data is present in the plan, group the data before passing to LLM ---
        plan = state.get('plan', {})
        group_field = None
        if plan and 'steps' in plan:
            for step in plan['steps']:
                if step.get('tool') == 'group_data' and 'group_field' in step.get('parameters', {}):
                    group_field = step['parameters']['group_field']
                    break
        if group_field and last_step_result and isinstance(last_step_result, list) and len(last_step_result) > 0 and isinstance(last_step_result[0], dict) and 'group_name' not in last_step_result[0]:
            from collections import defaultdict
            grouped = defaultdict(list)
            for item in last_step_result:
                grouped[item.get(group_field, None)].append(item)
            last_step_result = [{"group_name": k, "items": v, "item_count": len(v)} for k, v in grouped.items()]

        # PRINT the result that is passed onto the LLM (after all processing)
        if last_step_result:
            logger.info("RESULT PASSED TO LLM (for summary):", last_step_result)
        
        if not last_step_result:
            summary = "No results found."
            if 'plan_warnings' in state and state['plan_warnings']:
                summary += "\n" + "\n".join([f"[Warning] {w}" for w in state['plan_warnings']])
            return {"summary": summary, "status": "no_results", "results_count": 0, "results": []}
        
        # If the last step's result is a list of dicts (articles or groups), use the right formatter
        if isinstance(last_step_result, list) and len(last_step_result) > 0 and isinstance(last_step_result[0], dict):
            # Detect group-by result
            if 'group_name' in last_step_result[0] and ('item_count' in last_step_result[0] or 'article_count' in last_step_result[0]):
                raw_summary = format_group_results(state, last_step_result)
                if query_decomposer:
                    tailored = llm_tailor_response(state['original_query'], raw_summary, query_decomposer, state.get('plan'))
                else:
                    tailored = raw_summary.get('summary', 'No summary available.')
                if 'plan_warnings' in state and state['plan_warnings']:
                    tailored += "\n" + "\n".join([f"[Warning] {w}" for w in state['plan_warnings']])
                # Add note if this wasn't the final step
                if last_step_with_results and last_step_with_results < len(state.get('plan', {}).get('steps', [])):
                    tailored += f"\n\n[Note: Results shown are from step {last_step_with_results}. The final step returned no results.]"
                return {
                    'summary': tailored,
                    'status': 'success',
                    'results_count': len(last_step_result),
                    'results': last_step_result,
                    'metadata': raw_summary.get('metadata', {})
                }
            else:
                # If the last step's result is a list of dicts (articles), use that for summary
                raw_summary = format_search_results(state, last_step_result)
                # PRINT the result that is passed onto the LLM (after all processing)
                if last_step_result:
                    logger.info("RESULT PASSED TO LLM (for summary):", last_step_result)
                if query_decomposer:
                    tailored = llm_tailor_response(state['original_query'], raw_summary, query_decomposer, state.get('plan'))
                else:
                    tailored = raw_summary.get('summary', 'No summary available.')
                if 'plan_warnings' in state and state['plan_warnings']:
                    tailored += "\n" + "\n".join([f"[Warning] {w}" for w in state['plan_warnings']])
                # Add note if this wasn't the final step
                if last_step_with_results and last_step_with_results < len(state.get('plan', {}).get('steps', [])):
                    tailored += f"\n\n[Note: Results shown are from step {last_step_with_results}. The final step returned no results.]"
                return {
                    'summary': tailored,
                    'status': 'success',
                    'results_count': len(last_step_result),
                    'results': last_step_result,
                    'metadata': raw_summary.get('metadata', {})
                }
        # If the last step's result is a list of values (not dicts), show as value list
        elif isinstance(last_step_result, list) and len(last_step_result) > 0 and not isinstance(last_step_result[0], dict):
            summary = f"Found {len(last_step_result)} distinct values:\n" + "\n".join(f"- {v}" for v in last_step_result)
            
            # Add note if this wasn't the final step
            if last_step_with_results and last_step_with_results < len(state.get('plan', {}).get('steps', [])):
                summary += f"\n\n[Note: Results shown are from step {last_step_with_results}. The final step returned no results.]"
            
            result_obj = {
                "summary": summary,
                "status": "success",
                "results_count": len(last_step_result),
                "results": last_step_result,
                "metadata": {"result_type": "values"}
            }
            return result_obj
        else:
            summary = "No results found."
            if 'plan_warnings' in state and state['plan_warnings']:
                summary += "\n" + "\n".join([f"[Warning] {w}" for w in state['plan_warnings']])
            return {"summary": summary, "status": "no_results", "results_count": 0, "results": []}
    except Exception as e:
        logger.error(f"Error in summarize_node: {str(e)}", exc_info=True)
        return {
            "summary": f"An error occurred while processing the results: {str(e)}",
            "status": "error",
            "results_count": 0,
            "execution_steps": state.get('current_step', 0),
            "current_step": state.get('current_step', 0),
            "error": str(e),
            "results": [],
            "metadata": {}
        }

# Conditional Edge
def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue to the next step, summarize, or stop due to an error.
    
    Returns:
        str: Next node to transition to ('execute_tool', 'summarize', or 'error')
    """
    
    # Check for errors in the current state - if there's an error, stop immediately
    if 'error' in state:
        logger.error(f"Error in previous step: {state['error']}")
        return "summarize"  # Go to summarize to show the error to the user
    
    # Check if we've executed all steps in the plan
    if state['current_step'] > len(state['plan']['steps']):
        return "summarize"
        
    # Check if we have a valid step to execute
    if state['current_step'] <= 0 or state['current_step'] > len(state['plan']['steps']):
        logger.warning(f"Invalid current_step: {state['current_step']}, forcing summarize")
        return "summarize"
    
    # Check if we have results to work with for the next step
    next_step_idx = state['current_step'] - 1
    next_step = state['plan']['steps'][next_step_idx]
    
    # If the next step requires input but we have no results, stop
    if next_step['tool'] in ['sort_objects', 'group_by', 'get_top_n', 'get_distinct_values', 'filter_existing_objects'] and \
       not state['intermediate_results']:
        logger.warning(f"No results available for {next_step['tool']}, stopping execution")
        return "summarize"
    
    # --- PATCH: Prevent infinite recursion by checking if we've been stuck on the same step ---
    if '_step_retry_count' in state:
        state['_step_retry_count'] = state.get('_step_retry_count', 0) + 1
        if state['_step_retry_count'] > 2:  # Reduced from 3 to 2 to prevent excessive retries
            logger.error(f"Too many retries on step {state['current_step']}, stopping execution")
            return "summarize"
    else:
        state['_step_retry_count'] = 1
    
    # --- PATCH: Add additional safety check for filter_objects failures ---
    if next_step['tool'] == 'filter_objects' and state['intermediate_results']:
        # Check if the previous step was create_filter_from_group_results
        prev_step_idx = state['current_step'] - 2
        if prev_step_idx >= 0:
            prev_step = state['plan']['steps'][prev_step_idx]
            if prev_step['tool'] == 'create_filter_from_group_results':
                # If we're trying to filter after creating a filter from group results,
                # and we've already tried this step multiple times, stop
                if state.get('_step_retry_count', 0) > 1:  # Reduced from 2 to 1
                    logger.warning(f"Filter_objects step after create_filter_from_group_results failed multiple times, stopping")
                    return "summarize"
    
    # --- PATCH: Add check for empty results that would cause issues ---
    if state['intermediate_results']:
        last_result = state['intermediate_results'][-1]
        if not last_result or (isinstance(last_result, list) and len(last_result) == 0):
            logger.warning(f"Empty result from previous step, stopping execution")
            return "summarize"
    
    return "execute_tool"

def display_execution_summary(summary: Dict):
    """Display only the LLM-generated summary to the user, with no extra formatting or lines."""
    print(summary.get('summary', 'No summary available.'))

class ResearchAgent:
    """A dynamic research agent that can work with any Weaviate collection schema."""
    
    def __init__(self):
        self.client = self._connect_to_weaviate()
        self.class_name = self._select_collection()
        self.search_tools = SearchTools(client=self.client, class_name=self.class_name)
        self.query_decomposer = QueryDecomposer()
        self.tool_registry = self._get_tool_registry()
        self.app = self._build_graph()
        logger.info(f"Agent initialized for collection '{self.class_name}'")

    def _connect_to_weaviate(self):
        """Connect to the Weaviate instance."""
        try:
            client = weaviate.connect_to_local(
                host="localhost", port=8080, grpc_port=50051
            )
            logger.info("Successfully connected to Weaviate")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    def _select_collection(self):
        """Select a collection to work with."""
        try:
            # Use a direct REST call to bypass strict schema parsing in the client library.
            # This is a workaround for potential client/server version mismatches causing KeyErrors.
            response = self.client._connection.get(path="/schema")
            schema_json = response.json() # Parse the JSON from the response
            all_collections = schema_json.get("classes", [])
            
            if not all_collections:
                raise RuntimeError("No Weaviate collections found. Please add data first.")

            collection_names = [c["class"] for c in all_collections]

            if len(collection_names) == 1:
                class_name = collection_names[0]
                logger.info(f"Automatically selected the only available collection: '{class_name}'")
                return class_name
            
            print("\nAvailable collections:")
            for i, name in enumerate(collection_names, 1):
                print(f"  {i}. {name}")
            
            while True:
                try:
                    choice = int(input("Please select a collection to query: ")) - 1
                    if 0 <= choice < len(collection_names):
                        return collection_names[choice]
                    else:
                        print(f"Invalid choice. Please enter a number between 1 and {len(collection_names)}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        except Exception as e:
            logger.error(f"Failed to list or select Weaviate collections: {e}")
            logger.error("This might be due to a Weaviate server/client version mismatch or a connection issue.")
            raise

    def _get_tool_registry(self):
        """Get the tool registry for this agent instance."""
        return {
            "filter_data": {
                "func": self.search_tools.filter_data,
                "description": "Unified filtering that works on any data source.",
                "parameters": ["filters"]},
            "group_data": {
                "func": self.search_tools.group_data,
                "description": "Unified grouping that works on any data structure.",
                "parameters": ["group_field", "limit", "include_items", "max_items_per_group"]},
            "sort_data": {
                "func": self.search_tools.sort_data,
                "description": "Unified sorting that works on any data structure.",
                "parameters": ["data", "sort_field", "ascending"]},
            "semantic_search": {
                "func": self.search_tools.semantic_search,
                "description": "Performs a semantic search using the given query or reference article.",
                "parameters": ["query", "limit", "filter_dict", "reference_id", "score_threshold", "group_field", "max_items_per_group"],
            },
        }

    def _build_graph(self):
        """Build the LangGraph workflow."""
        from agent import should_continue  # Use the global function
        workflow = StateGraph(AgentState)
        workflow.add_node("decompose_query", partial(decompose_query_node, search_tools=self.search_tools, query_decomposer=self.query_decomposer, tool_registry=self.tool_registry))
        workflow.add_node("execute_tool", partial(execute_tool_node, tool_registry=self.tool_registry, search_tools=self.search_tools))
        workflow.add_node("summarize_results", partial(summarize_node, query_decomposer=self.query_decomposer))

        workflow.set_entry_point("decompose_query")
        workflow.add_edge("decompose_query", "execute_tool")
        workflow.add_conditional_edges(
            "execute_tool",
            should_continue,  # Use the global function
            {
                "execute_tool": "execute_tool",
                "summarize": "summarize_results"
            }
        )
        workflow.add_edge("summarize_results", END)
        return workflow.compile()

    def run(self):
        """Run the research agent in interactive mode."""
        print("\n" + "="*80)
        print("DYNAMIC SCHEMA RESEARCH AGENT")
        print(f"Now querying collection: '{self.class_name}'")
        print("Type 'exit' to quit\n" + "="*80)
        try:
            while True:
                try:
                    user_query = input("\nEnter your research query: ").strip()
                    if user_query.lower() in ('exit', 'quit'):
                        break
                    if not user_query:
                        print("Please enter a valid query.")
                        continue
                    print("\nProcessing your query...")
                    inputs = {"original_query": user_query}
                    final_state = self.app.invoke(inputs)
                    self.display_execution_summary(final_state)
                except KeyboardInterrupt:
                    print("\nOperation cancelled by user.")
                    break
                except Exception as e:
                    print(f"\nAn error occurred during query execution: {str(e)}")
                    if input("View detailed error? (y/n): ").lower() == 'y':
                        import traceback
                        traceback.print_exc()
        finally:
            print("\nShutting down...")
            self.client.close()
            print("Goodbye!")

    def display_execution_summary(self, summary: Dict):
        """Display the execution summary to the user."""
        print("\n" + "="*80)
        print("SEARCH RESULTS")
        print("="*80)
        print("\n" + summary.get('summary', 'No summary available.'))
        print("\n" + "="*80 + "\n")


# Conditional Edge

def validate_tool_parameters(tool_name, parameters, schema_fields):
    """
    Validate and normalize tool parameters based on the schema.
    Also, move 'limit' out of filters if present (LLM sometimes puts it there).
    """
    # Ensure schema_fields is a dictionary - handle all edge cases
    if not isinstance(schema_fields, dict):
        logger.warning(f"schema_fields is not a dictionary in validate_tool_parameters: {type(schema_fields)}, using empty dict")
        schema_fields = {}
    valid_fields = set(schema_fields.keys())
    validated_params = parameters.copy()

    # Move 'limit' out of filters if present
    if 'filters' in validated_params and isinstance(validated_params['filters'], dict) and 'limit' in validated_params['filters']:
        validated_params['limit'] = validated_params['filters'].pop('limit')

    # Tool-specific validation for unified tools
    if tool_name == "filter_data":
        if "filters" in validated_params:
            filters = validated_params["filters"]
            # Handle filters that are strings
            if isinstance(filters, str):
                import json
                if filters.strip().startswith("$PREV_STEP_RESULT"):
                    # Leave for substitution
                    return validated_params
                try:
                    filters = json.loads(filters)
                    validated_params["filters"] = filters
                except Exception:
                    logger.warning(f"Filters is a string but not valid JSON: {filters}, skipping validation")
                    return validated_params
            if not isinstance(filters, dict):
                logger.warning(f"Filters is not a dictionary: {type(filters)}, skipping validation")
                return validated_params
            for field in list(filters.keys()):
                if field not in valid_fields and field.lower() not in ["$and", "$or", "$not"]:
                    logger.warning(f"Filter field '{field}' not in schema, removing filter.")
                    del filters[field]

    elif tool_name == "group_data":
        if "group_field" in validated_params:
            field = validated_params["group_field"]
            if field not in valid_fields and field not in ["value", "group_name", "item_count"]:
                logger.warning(f"Group field '{field}' not in schema, using first available field.")
                validated_params["group_field"] = list(valid_fields)[0] if valid_fields else None

    elif tool_name == "sort_data":
        if "sort_field" in validated_params:
            field = validated_params["sort_field"]
            # Allow special fields for group results
            if field not in valid_fields and field not in ["item_count", "article_count", "group_name"]:
                logger.warning(f"Sort field '{field}' not in schema, removing sort.")
                validated_params["sort_field"] = None

    elif tool_name == "get_distinct_values":
        if "field" in validated_params:
            field = validated_params["field"]
            if field not in valid_fields:
                logger.warning(f"Field '{field}' not in schema, using first available field.")
                validated_params["field"] = list(valid_fields)[0] if valid_fields else None

    return validated_params

def handle_chaining_fallback(tool_name: str, params: Dict, prev_result: Any, schema_fields: Dict) -> Dict:
    """
    Handle fallback strategies for tool chaining when the primary approach fails.
    
    Args:
        tool_name: Name of the tool that failed
        params: Original parameters
        prev_result: Previous step result
        schema_fields: Available schema fields
    
    Returns:
        Modified parameters for retry
    """
    # Ensure schema_fields is a dictionary - handle all edge cases
    if not isinstance(schema_fields, dict):
        logger.warning(f"schema_fields is not a dictionary in handle_chaining_fallback: {type(schema_fields)}, using empty dict")
        schema_fields = {}
    
    fallback_params = params.copy()
    
    if tool_name == "filter_objects":
        # If filter_objects fails, try to extract field values and create a simpler filter
        if "filters" in fallback_params and prev_result:
            # Try to extract field values from previous result
            for field, condition in list(fallback_params["filters"].items()):
                if isinstance(condition, dict) and "$in" in condition:
                    # If it's an $in filter, try to extract the values from prev_result
                    try:
                        if isinstance(prev_result, list) and prev_result:
                            # Extract values from the first few items
                            extracted_values = []
                            for item in prev_result[:5]:  # Limit to first 5 items
                                if isinstance(item, dict) and field in item:
                                    extracted_values.append(item[field])
                                elif hasattr(item, field):
                                    extracted_values.append(getattr(item, field))
                            
                            if extracted_values:
                                # Create a simpler filter with extracted values - ensure no double-nesting
                                fallback_params["filters"] = {field: {"$in": extracted_values}}
                                logger.info(f"Created fallback filter with extracted values: {extracted_values}")
                                break
                    except Exception as e:
                        logger.warning(f"Failed to extract field values for fallback: {e}")
    
    elif tool_name == "group_by":
        # If group_by fails, try with a different field or fallback to get_distinct_values
        if "group_field" in fallback_params:
            group_field = fallback_params["group_field"]
            if group_field not in schema_fields:
                # Try to find a suitable alternative field
                for field in ["category", "author", "title", "type"]:
                    if field in schema_fields:
                        fallback_params["group_field"] = field
                        logger.info(f"Changed group_field from '{group_field}' to '{field}'")
                        break
    
    elif tool_name == "sort_objects":
        # If sort_objects fails, try with a different field
        if "sort_field" in fallback_params:
            sort_field = fallback_params["sort_field"]
            if sort_field not in schema_fields and sort_field not in ["item_count", "article_count", "group_name"]:
                # Try to find a suitable alternative field
                for field in ["title", "author", "category", "publish_date"]:
                    if field in schema_fields:
                        fallback_params["sort_field"] = field
                        logger.info(f"Changed sort_field from '{sort_field}' to '{field}'")
                        break
    
    return fallback_params

def validate_chain_compatibility(tool_name: str, params: Dict, prev_result: Any) -> bool:
    """
    Validate if a tool can be chained with the previous result.
    
    Args:
        tool_name: Name of the tool
        params: Tool parameters
        prev_result: Previous step result
    
    Returns:
        True if compatible, False otherwise
    """
    if not prev_result:
        return True  # No previous result, so no compatibility issues
    
    # For most tools, we can handle different data types through normalization
    # Only return False for clearly incompatible cases
    
    if tool_name in ["sort_data", "group_data", "get_top_n", "get_distinct_values"]:
        # These tools work with the SearchTools._normalize_data_for_operation method
        # which can handle various input types, so we don't need to be strict here
        return True
    
    elif tool_name == "filter_data":
        # filter_data can work with any previous result through its source_type parameter
        return True
    
    elif tool_name == "extract_field_values":
        # extract_field_values can work with any data structure
        return True
    
    elif tool_name == "create_filter_from_values":
        # create_filter_from_values can work with any value list
        return True
    
    elif tool_name == "combine_results":
        # combine_results can work with any result lists
        return True
    
    elif tool_name == "semantic_search":
        # semantic_search can work with any previous result
        return True
    
    return True

def main():
    """Main entry point for the research agent."""
    try:
        agent = ResearchAgent()
        agent.run()
    except Exception as e:
        logger.critical(f"A fatal error occurred during agent initialization or execution: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        print("Please check your Weaviate connection and Google API key.")
    finally:
        print("\nExiting application.")

# --- Main Execution ---
if __name__ == "__main__":
    main()

