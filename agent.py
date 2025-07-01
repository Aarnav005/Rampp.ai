import google.generativeai as genai
import os
from typing import Dict, Optional, List, Any, TypedDict
import json
from dotenv import load_dotenv
from datetime import datetime
import weaviate
from weaviate.classes.query import Filter
import logging
from langgraph.graph import StateGraph, END
import re
from functools import partial
import urllib.parse
from search_tools import SearchTools, normalize_filter_dict
from type import AgentState
from query_decomposer import QueryDecomposer
from formatting import log_tool_execution, flatten_results, convert_datetimes, format_search_results, format_group_results

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
        "semantic_search": {
            "func": search_tools.semantic_search,
            "description": "Performs a semantic search using the given query or reference article.",
            "parameters": ["query", "limit", "filter_dict", "reference_id", "score_threshold"],
        },
        "filter_data": {
            "func": search_tools.filter_data,
            "description": "Unified filtering that works on any data source (database or existing objects).",
            "parameters": ["data_source", "filters", "limit", "source_type"]},
        "group_data": {
            "func": search_tools.group_data,
            "description": "Unified grouping that works on any data structure.",
            "parameters": ["data", "group_field", "limit", "include_items", "max_items_per_group"]},
        "sort_data": {
            "func": search_tools.sort_data,
            "description": "Unified sorting that works on any data structure.",
            "parameters": ["data", "sort_field", "ascending"]},
        "extract_field_values": {
            "func": search_tools.extract_field_values,
            "description": "Extract values for a specific field from any data structure.",
            "parameters": ["data", "field_name", "limit"]},
        "create_filter_from_values": {
            "func": search_tools.create_filter_from_values,
            "description": "Create a filter from a list of values (unified method).",
            "parameters": ["values", "target_field", "limit"]},
        "combine_results": { 
            "func": search_tools.combine_results, 
            "description": "Combines multiple result lists into one. Useful for merging results from different operations.", 
            "parameters": ["results_list", "deduplicate"]},
        "get_top_n": {
            "func": search_tools.get_top_n,
            "description": "Gets the top N objects from any data structure.",
            "parameters": ["data", "n"]},
        "find_item_by_field": {
            "func": search_tools.find_item_by_field,
            "description": "Finds an item by a specific field value and returns its Weaviate ID (UUID).",
            "parameters": ["field_name", "field_value"]},
        "transform_results": { 
            "func": search_tools.transform_results, 
            "description": "Transforms results into different formats. Useful for data preparation.", 
            "parameters": ["objects", "transform_type"]},
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
        
        # --- PATCH: Fix incorrect author counting pattern ---
        for i in range(len(steps) - 1):
            if (i + 1 < len(steps) and 
                steps[i]['tool'] == 'extract_field_values' and 
                steps[i+1]['tool'] == 'group_data' and
                steps[i]['parameters'].get('field_name') == 'author' and
                steps[i+1]['parameters'].get('group_field') == 'value'):
                
                # This is the incorrect pattern - replace with direct grouping
                logger.info("Fixing incorrect author counting pattern")
                
                # Remove the extract_field_values step
                steps.pop(i)
                
                # Update the group_data step to group by author directly
                steps[i]['parameters']['group_field'] = 'author'
                steps[i]['parameters']['data'] = '$PREV_STEP_RESULT'
                steps[i]['description'] = 'Group by author to count articles'
                
                # Update step IDs
                for j in range(i, len(steps)):
                    steps[j]['step_id'] = j + 1
                
                warnings.append(f"Fixed author counting pattern: replaced extract_field_values + group_data with direct group_data by author")
                break
        
        for i, step in enumerate(steps):
            try:
                params = step.get('parameters', {})
                
                # Validate group_field
                if 'group_field' in params and params['group_field'] not in valid_fields:
                    # Check if this step follows extract_field_values
                    if i > 0 and steps[i-1]['tool'] == 'extract_field_values':
                        # For extract_field_values results, the field gets normalized to 'value'
                        if params['group_field'] == steps[i-1]['parameters'].get('field_name'):
                            params['group_field'] = 'value'
                            warnings.append(f"Step {step['step_id']}: group_field '{steps[i-1]['parameters'].get('field_name')}' replaced with 'value' for extract_field_values results.")
                        elif params['group_field'] == 'value':
                            # LLM already correctly set it to 'value', this is valid
                            warnings.append(f"Step {step['step_id']}: group_field 'value' is valid for extract_field_values results.")
                        else:
                            warnings.append(f"Step {step['step_id']}: group_field '{params['group_field']}' not in schema, removing group_by.")
                            step['parameters']['group_field'] = None
                    else:
                        warnings.append(f"Step {step['step_id']}: group_field '{params['group_field']}' not in schema, removing group_by.")
                        step['parameters']['group_field'] = None
                        
                # Validate sort_field
                if 'sort_field' in params:
                    # Handle group_by result sorting
                    prev_tool = steps[i-1]['tool'] if i > 0 else None
                    prev_prev_tool = steps[i-2]['tool'] if i > 1 else None
                    
                    if params['sort_field'] == 'count' and (prev_tool == 'group_by' or (prev_tool == 'get_top_n' and prev_prev_tool == 'group_by')):
                        params['sort_field'] = 'item_count'
                        warnings.append(f"Step {step['step_id']}: sort_field 'count' replaced with 'item_count' for group_by results.")
                    elif params['sort_field'] not in valid_fields and params['sort_field'] not in ['item_count', 'article_count']:
                        warnings.append(f"Step {step['step_id']}: sort_field '{params['sort_field']}' not in schema, removing sort.")
                        step['parameters']['sort_field'] = None
                        
                # Validate and fix filters
                if 'filters' in params:
                    # --- PATCH: Handle filters that are strings ---
                    if isinstance(params['filters'], str):
                        filters_str = params['filters']
                        if filters_str.strip().startswith('$PREV_STEP_RESULT'):
                            # Leave for substitution, skip validation
                            continue
                        else:
                            try:
                                import json
                                params['filters'] = json.loads(filters_str)
                            except Exception:
                                logger.warning(f"Step {step['step_id']}: filters is a string but not valid JSON: {filters_str}, skipping filter validation")
                                continue
                    
                    # Ensure filters is a dictionary before processing
                    if not isinstance(params['filters'], dict):
                        logger.warning(f"Step {step['step_id']}: filters is not a dictionary: {type(params['filters'])}, skipping filter validation")
                        continue
                        
                    to_remove = []
                    for f in list(params['filters'].keys()):
                        # Don't remove logical operators - they are valid at the top level
                        if f not in valid_fields and f not in logical_operators:
                            warnings.append(f"Step {step['step_id']}: filter field '{f}' not in schema, removing filter.")
                            to_remove.append(f)
                    for f in to_remove:
                        del step['parameters']['filters'][f]
                        
                    # Fix field references after group_by operations
                    for k, v in params['filters'].items():
                        if isinstance(v, dict):
                            for op, val in v.items():
                                if isinstance(val, str) and val.startswith('$PREV_STEP_RESULT'):
                                    # Check if this step follows a group_by operation
                                    prev_tool = steps[i-1]['tool'] if i > 0 else None
                                    prev_prev_tool = steps[i-2]['tool'] if i > 1 else None
                                    
                                    # If the previous step was group_by or get_top_n after group_by
                                    if prev_tool == 'group_by' or (prev_tool == 'get_top_n' and prev_prev_tool == 'group_by'):
                                        # Get the group field from the group_by step
                                        group_field = None
                                        if prev_tool == 'group_by':
                                            group_field = steps[i-1]['parameters'].get('group_field')
                                        elif prev_prev_tool == 'group_by':
                                            group_field = steps[i-2]['parameters'].get('group_field')
                                        
                                        # Dynamically fix field references
                                        if group_field and f'.{group_field}' in val:
                                            new_val = val.replace(f'.{group_field}', '.group_name')
                                            params['filters'][k][op] = new_val
                                            warnings.append(f"Step {step['step_id']}: changed filter reference from '.{group_field}' to '.group_name' after group_by.")
                                    
                # Validate and fix filter_dict (for semantic_search)
                if 'filter_dict' in params:
                    # Ensure filter_dict is a dictionary before processing
                    if not isinstance(params['filter_dict'], dict):
                        logger.warning(f"Step {step['step_id']}: filter_dict is not a dictionary: {type(params['filter_dict'])}, skipping filter_dict validation")
                        continue
                        
                    to_remove = []
                    for f in list(params['filter_dict'].keys()):
                        # Don't remove logical operators - they are valid at the top level
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

def decompose_query_node(state: Dict[str, Any], search_tools, query_decomposer, tool_registry) -> Dict[str, Any]:
    """Decompose a user query into a multi-step execution plan."""
    try:
        schema_fields = search_tools.get_schema_fields()
        # Ensure schema_fields is a dictionary - handle all edge cases
        if not isinstance(schema_fields, dict):
            logger.warning(f"Retrieved schema_fields is not a dictionary: {type(schema_fields)}, using empty dict")
            schema_fields = {}
    except Exception as e:
        logger.error(f"Failed to get schema fields: {e}")
        schema_fields = {}
    
    try:
        plan = query_decomposer.decompose(state['original_query'], search_tools.class_name, schema_fields, tool_registry)
        
        plan, plan_warnings = validate_and_correct_plan(plan, schema_fields)
        
        # Auto-append filter_objects step if last step is get_distinct_values
        steps = plan.get('steps', [])
        # Only auto-append if previous step is not extract_field_values or get_distinct_values
        if steps and steps[-1]['tool'] == 'get_distinct_values':
            prev_tool = steps[-2]['tool'] if len(steps) > 1 else None
            if prev_tool not in ['extract_field_values', 'get_distinct_values']:
                last_field = steps[-1]['parameters'].get('field', None)
                if last_field:
                    steps.append({
                        "step_id": steps[-1]['step_id'] + 1,
                        "tool": "filter_objects",
                        "parameters": {
                            "filters": {last_field: {"$in": "$PREV_STEP_RESULT"}},
                            "limit": 1000
                        },
                        "description": f"Fetch all articles where {last_field} is in the previous result."
                    })
                    plan['steps'] = steps
                    plan_warnings = plan_warnings or []
                    plan_warnings.append("Auto-appended filter_objects step to fetch articles by values from get_distinct_values.")
    except Exception as e:
        logger.error(f"LLM-based planning failed with error: {e}", exc_info=True)
        logger.warning(f"LLM-based planning failed: {e}, falling back to simple search.")
        plan = {
            "steps": [{
                "step_id": 1,
                "tool": "semantic_search",
                "parameters": {"query": state['original_query'], "limit": 10},
                "description": f"Fallback: Perform a search for '{state['original_query']}'."
            }]
        }
        plan_warnings = ["LLM-based planning failed, using fallback plan."]

    # Display the search plan
    print("\n" + "="*60)
    print("SEARCH PLAN")
    print("="*60)
    for step in plan['steps']:
        print(f"Step {step['step_id']}: {step.get('description', 'No description available')}")
    if plan_warnings:
        print("\n[PLAN WARNINGS]")
        for warning in plan_warnings:
            print(f"  • {warning}")
    print("="*60 + "\n")
    
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
    """Execute a single tool step in the research plan."""
    try:
        current_step_idx = state['current_step'] - 1
        step = state['plan']['steps'][current_step_idx]
        tool_name = step['tool']
        
        # Use the provided tool_registry or fall back to global TOOL_REGISTRY
        if tool_registry is None:
            if search_tools:
                tool_registry = get_tool_registry(search_tools)
            else:
                raise ValueError("Either tool_registry or search_tools must be provided")
        tool_info = tool_registry.get(tool_name)
        if not tool_info:
            raise ValueError(f"Unknown tool: {tool_name}")
        params = step.get('parameters', {}).copy()
        # Get schema fields for validation - with fallback
        try:
            if search_tools:
                schema_fields = search_tools.get_schema_fields()
            else:
                schema_fields = {}
            if not isinstance(schema_fields, dict):
                logger.warning(f"Retrieved schema_fields is not a dictionary: {type(schema_fields)}, using empty dict")
                schema_fields = {}
        except Exception as e:
            logger.warning(f"Failed to get schema fields: {e}, using empty schema")
            schema_fields = {}
        # Get previous result for chaining validation
        prev_result = None
        if state['intermediate_results']:
            prev_result = state['intermediate_results'][-1]
        # --- PATCH: Prevent get_distinct_values from running on a list of values ---
        if tool_name == "get_distinct_values":
            if prev_result is not None and (not isinstance(prev_result, list) or (prev_result and not isinstance(prev_result[0], dict))):
                logger.warning("get_distinct_values called on non-object list, skipping step.")
                # Skip this step, just pass through previous result
                return {
                    **state,
                    'current_step': state['current_step'] + 1,
                    'intermediate_results': state['intermediate_results'] + [prev_result],
                    'final_results': prev_result if isinstance(prev_result, list) else [prev_result]
                }
        # FIRST: Substitute $PREV_STEP_RESULT and indexed/field references BEFORE validation
        if state['intermediate_results']:
            last_result = state['intermediate_results'][-1]
            
            # --- PATCH: Special handling for combine_results tool ---
            if tool_name == "combine_results" and "results_list" in params:
                # For combine_results, we need to substitute references to multiple previous results
                results_list = params["results_list"]
                if isinstance(results_list, list):
                    substituted_results = []
                    for i, ref in enumerate(results_list):
                        if isinstance(ref, str) and ref.startswith("$PREV_STEP_RESULT"):
                            # Extract the index from the reference
                            if "[" in ref and "]" in ref:
                                try:
                                    # Parse index like [0], [1], etc.
                                    start_idx = ref.find("[") + 1
                                    end_idx = ref.find("]")
                                    if start_idx > 0 and end_idx > start_idx:
                                        idx_str = ref[start_idx:end_idx]
                                        if idx_str.isdigit():
                                            idx = int(idx_str)
                                            # Get the result at the specified index
                                            if 0 <= idx < len(state['intermediate_results']):
                                                substituted_results.append(state['intermediate_results'][idx])
                                            else:
                                                logger.warning(f"Index {idx} out of range for intermediate_results, using empty list")
                                                substituted_results.append([])
                                        else:
                                            logger.warning(f"Invalid index '{idx_str}' in reference '{ref}', using empty list")
                                            substituted_results.append([])
                                    else:
                                        logger.warning(f"Could not parse index from reference '{ref}', using empty list")
                                        substituted_results.append([])
                                except Exception as e:
                                    logger.warning(f"Error parsing reference '{ref}': {e}, using empty list")
                                    substituted_results.append([])
                            else:
                                # No index specified, use the last result
                                substituted_results.append(last_result)
                        else:
                            # Not a reference, use as-is
                            substituted_results.append(ref)
                    params["results_list"] = substituted_results
                else:
                    # Fallback to normal substitution
                    params = substitute_prev_step_refs(params, last_result)
            else:
                # Normal substitution for other tools
                params = substitute_prev_step_refs(params, last_result)
            
            if 'filters' in params and isinstance(params['filters'], str):
                import json
                filters_str = params['filters']
                if filters_str.strip().startswith('$PREV_STEP_RESULT'):
                    pass  # Leave for substitution
                else:
                    try:
                        params['filters'] = json.loads(filters_str)
                    except Exception:
                        logger.warning(f"filters is a string but not valid JSON after substitution: {filters_str}, skipping parse")
            if tool_name == "filter_existing_objects" and "objects" not in params:
                params["objects"] = last_result
        # SECOND: Validate chain compatibility
        # Since validate_chain_compatibility now returns True for all valid cases,
        # we don't need the fallback logic here
        # if not validate_chain_compatibility(tool_name, params, prev_result):
        #     logger.warning(f"Chain compatibility issue detected for {tool_name}, attempting fallback")
        #     params = handle_chaining_fallback(tool_name, params, prev_result, schema_fields)
        #     if state['intermediate_results']:
        #         params = substitute_prev_step_refs(params, state['intermediate_results'][-1])
        
        # THIRD: Validate and normalize parameters
        params = validate_tool_parameters(tool_name, params, schema_fields)
        # FOURTH: Filter out unsupported parameters for each tool
        supported_params = tool_info.get('parameters', [])
        filtered_params = {k: v for k, v in params.items() if k in supported_params}
        if filtered_params != params:
            logger.warning(f"Removed unsupported parameters for {tool_name}: {set(params.keys()) - set(supported_params)}")
            params = filtered_params
        
        # --- PATCH: Handle parameter name mismatches for unified tools ---
        if tool_name == "extract_field_values" and "data" not in params and "objects" in params:
            # Convert old parameter name to new unified parameter name
            params["data"] = params.pop("objects")
        
        if tool_name == "group_data" and "data" not in params and "objects" in params:
            # Convert old parameter name to new unified parameter name
            params["data"] = params.pop("objects")
        
        if tool_name == "sort_data" and "data" not in params and "objects" in params:
            # Convert old parameter name to new unified parameter name
            params["data"] = params.pop("objects")
        
        if tool_name == "filter_data" and "data_source" not in params and "objects" in params:
            # Convert old parameter name to new unified parameter name
            params["data_source"] = params.pop("objects")
        
        if tool_name == "get_top_n" and "data" not in params and "objects" in params:
            # Convert old parameter name to new unified parameter name
            params["data"] = params.pop("objects")
        
        if tool_name == "get_distinct_values" and "data" not in params and "objects" in params:
            # Convert old parameter name to new unified parameter name
            params["data"] = params.pop("objects")
        
        if tool_name == "create_filter_from_values" and "values" not in params and "group_results" in params:
            # Convert old parameter name to new unified parameter name
            params["values"] = params.pop("group_results")
        
        # --- PATCH: Ensure required parameters are present ---
        required_params = {
            "extract_field_values": ["data", "field_name"],
            "group_data": ["data", "group_field"],
            "sort_data": ["data", "sort_field"],
            "filter_data": ["data_source", "filters"],
            "get_top_n": ["data", "n"],
            "get_distinct_values": ["data", "field"],
            "create_filter_from_values": ["values", "target_field"],
        }
        
        if tool_name in required_params:
            missing_params = [p for p in required_params[tool_name] if p not in params]
            if missing_params:
                logger.error(f"Missing required parameters for {tool_name}: {missing_params}")
                # Try to add missing parameters with defaults or from previous results
                if "data" in missing_params and state['intermediate_results']:
                    params["data"] = state['intermediate_results'][-1]
                elif "data_source" in missing_params:
                    params["data_source"] = None  # Default to database query
                elif "values" in missing_params and state['intermediate_results']:
                    params["values"] = state['intermediate_results'][-1]
        # Execute tool with retry logic
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                result = tool_info['func'](**params)
                result_count = len(result) if hasattr(result, '__len__') else 1
                log_tool_execution(tool_name, result_count, state['current_step'])
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Tool {tool_name} failed on attempt {attempt + 1}: {str(e)}")
                    params = handle_chaining_fallback(tool_name, params, prev_result, schema_fields)
                    if state['intermediate_results']:
                        params = substitute_prev_step_refs(params, state['intermediate_results'][-1])
                else:
                    error_msg = f"Failed to execute step {state.get('current_step', '?')} ({tool_name}): {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    return {**state, 'error': error_msg}
        
        # --- PATCH: Prevent infinite recursion by checking retry count ---
        if '_step_retry_count' in state:
            state['_step_retry_count'] = state.get('_step_retry_count', 0) + 1
            if state['_step_retry_count'] > 3:  # Maximum 3 retries per step
                error_msg = f"Too many retries on step {state['current_step']} ({tool_name}), stopping execution"
                logger.error(error_msg)
                return {**state, 'error': error_msg}
        else:
            state['_step_retry_count'] = 1
        
        if '_step_retry_count' in state:
            del state['_step_retry_count']
        return {
            'current_step': state['current_step'] + 1,
            'intermediate_results': state['intermediate_results'] + [result],
            'final_results': result if isinstance(result, list) else [result],
            'plan_warnings': state.get('plan_warnings', [])
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
You are an expert research assistant. Given the following user query, the search plan (as JSON), and the raw search output (in JSON), generate a clear, complete, and well-formatted answer or summary that directly addresses the user's intent.

**Important:**
- Use the search plan to understand the context and the steps taken.
- Enumerate and display **all results** in the output, not just a sample or summary.
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
                    tailored += f"\n\n[Note: Results shown are from step {last_step_with_results}. The final filter step returned no results.]"
                
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
                if query_decomposer:
                    tailored = llm_tailor_response(state['original_query'], raw_summary, query_decomposer, state.get('plan'))
                else:
                    tailored = raw_summary.get('summary', 'No summary available.')
                if 'plan_warnings' in state and state['plan_warnings']:
                    tailored += "\n" + "\n".join([f"[Warning] {w}" for w in state['plan_warnings']])
                
                # Add note if this wasn't the final step
                if last_step_with_results and last_step_with_results < len(state.get('plan', {}).get('steps', [])):
                    tailored += f"\n\n[Note: Results shown are from step {last_step_with_results}. The final filter step returned no results.]"
                
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
                summary += f"\n\n[Note: Results shown are from step {last_step_with_results}. The final filter step returned no results.]"
            
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
            "semantic_search": { "func": self.search_tools.semantic_search, "description": "Performs a semantic search.", "parameters": ["query", "limit", "filter_dict", "reference_id", "score_threshold"]},
            "filter_data": { "func": self.search_tools.filter_data, "description": "Unified filtering that works on any data source.", "parameters": ["data_source", "filters", "limit", "source_type"]},
            "group_data": { "func": self.search_tools.group_data, "description": "Unified grouping that works on any data structure.", "parameters": ["data", "group_field", "limit", "include_items", "max_items_per_group"]},
            "sort_data": { "func": self.search_tools.sort_data, "description": "Unified sorting that works on any data structure.", "parameters": ["data", "sort_field", "ascending"]},
            "extract_field_values": { "func": self.search_tools.extract_field_values, "description": "Extract values for a specific field from any data structure.", "parameters": ["data", "field_name", "limit"]},
            "create_filter_from_values": { "func": self.search_tools.create_filter_from_values, "description": "Create a filter from a list of values (unified method).", "parameters": ["values", "target_field", "limit"]},
            "combine_results": { "func": self.search_tools.combine_results, "description": "Combines multiple result lists into one. Useful for merging results from different operations.", "parameters": ["results_list", "deduplicate"]},
            "get_top_n": { "func": self.search_tools.get_top_n, "description": "Gets the top N objects from any data structure.", "parameters": ["data", "n"]},
            "find_item_by_field": { "func": self.search_tools.find_item_by_field, "description": "Finds an item by a specific field value and returns its Weaviate ID (UUID). Use this only when you have the exact field value.", "parameters": ["field_name", "field_value"]},
            "transform_results": { "func": self.search_tools.transform_results, "description": "Transforms results into different formats. Useful for data preparation.", "parameters": ["objects", "transform_type"]},
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
