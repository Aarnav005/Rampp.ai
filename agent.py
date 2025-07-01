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

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose HTTP requests and other noise
logging.getLogger('weaviate').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

def log_tool_execution(tool_name: str, result_count: int, step_id: int = None):
    """Log tool execution with minimal information."""
    step_info = f"Step {step_id}: " if step_id else ""
    print(f"{step_info}{tool_name} → {result_count} results")

load_dotenv()

# --- FILTER NORMALIZATION PATCH ---
def normalize_filter_dict(filter_dict):
    """
    Recursively normalize filter dicts from {'operator': ..., 'value': ...} to the expected {'$gt': ...} etc.
    Also handles nested $and/$or/$not logic.
    """
    if not isinstance(filter_dict, dict):
        return filter_dict
    
    # --- PATCH: Handle empty dictionaries ---
    if not filter_dict:
        return {}
    
    normalized = {}
    for k, v in filter_dict.items():
        if k.lower() in ["$and", "$or"] and isinstance(v, list):
            normalized[k.lower()] = [normalize_filter_dict(item) for item in v]
        elif k.lower() == "$not":
            # Handle NOT operations by converting to property-level negations where possible
            if isinstance(v, dict):
                # If NOT contains a single field with a single operator, convert to property-level negation
                if len(v) == 1:
                    field, field_value = list(v.items())[0]
                    if isinstance(field_value, dict) and len(field_value) == 1:
                        operator, operand = list(field_value.items())[0]
                        # Convert to property-level negation
                        if operator in ["$eq", "equal"]:
                            normalized[field] = {"$ne": operand}
                        elif operator in ["$ne", "notequal", "not_equal"]:
                            normalized[field] = {"$eq": operand}
                        elif operator in ["$gt", "greaterthan", "greater_than"]:
                            normalized[field] = {"$lte": operand}
                        elif operator in ["$gte", "greaterthanequal", "greater_or_equal"]:
                            normalized[field] = {"$lt": operand}
                        elif operator in ["$lt", "lessthan", "less_than"]:
                            normalized[field] = {"$gte": operand}
                        elif operator in ["$lte", "lessthanequal", "less_or_equal"]:
                            normalized[field] = {"$gt": operand}
                        elif operator in ["$in", "contains_any"]:
                            normalized[field] = {"$nin": operand}
                        elif operator in ["$nin", "notin", "not_contains_any"]:
                            normalized[field] = {"$in": operand}
                        elif operator in ["$contains", "$like"]:
                            # --- PATCH: For NOT contains/like, skip this filter since Weaviate doesn't support it ---
                            logger.warning(f"NOT contains/like operations are not supported for field '{field}', skipping filter")
                            continue
                        else:
                            # For other operators, skip this filter
                            logger.warning(f"Unsupported NOT operator '{operator}' for field '{field}', skipping filter")
                            continue
                    else:
                        # Multiple operators or complex structure, keep as NOT
                        normalized[k.lower()] = normalize_filter_dict(v)
                else:
                    # Multiple fields, keep as NOT
                    normalized[k.lower()] = normalize_filter_dict(v)
            else:
                # Non-dict value, keep as NOT
                normalized[k.lower()] = normalize_filter_dict(v)
        elif isinstance(v, dict) and set(v.keys()) == {"operator", "value"}:
            op = v["operator"].lower()
            val = v["value"]
            # Map to Weaviate-style operator keys
            op_map = {
                "equal": "$eq", "eq": "$eq",
                "notequal": "$ne", "not_equal": "$ne", "ne": "$ne",
                "greaterthan": "$gt", "gt": "$gt",
                "greaterthanequal": "$gte", "gte": "$gte",
                "lessthan": "$lt", "lt": "$lt",
                "lessthanequal": "$lte", "lte": "$lte",
                "in": "$in",
                "notin": "$nin",
                "like": "$like",
                "contains": "$contains",
            }
            mapped = op_map.get(op, op)
            normalized[k] = {mapped: val}
        elif isinstance(v, dict):
            # --- PATCH: Handle empty dictionaries ---
            if not v:
                logger.warning(f"Empty filter value for field '{k}', skipping")
                continue
                
            # --- PATCH: Enhanced double-nested filter detection ---
            # Only flatten if BOTH outer and inner keys start with $ (i.e., both are operators)
            if len(v) == 1:
                operator, operand = list(v.items())[0]
                # --- PATCH: Don't flatten if outer key is a field name (doesn't start with $) ---
                if not k.startswith('$'):
                    normalized[k] = normalize_filter_dict(v)
                    continue
                # --- PATCH: Don't flatten logical operators like $not, $and, $or ---
                if operator.lower() in ["$not", "$and", "$or"]:
                    normalized[k] = normalize_filter_dict(v)
                    continue
                if isinstance(operand, dict) and len(operand) == 1:
                    inner_operator, inner_operand = list(operand.items())[0]
                    if inner_operator.startswith('$') and k.startswith('$'):
                        # This is a double-nested filter, flatten it
                        logger.warning(f"Double-nested filter detected for operator '{k}': {v}, flattening to {inner_operator}: {inner_operand}")
                        normalized[k] = {inner_operator: inner_operand}
                        continue
                # Check for more complex double-nesting patterns
                if isinstance(operand, dict):
                    for inner_key, inner_val in operand.items():
                        if isinstance(inner_val, dict) and len(inner_val) == 1:
                            inner_inner_key, inner_inner_val = list(inner_val.items())[0]
                            if inner_inner_key.startswith('$') and k.startswith('$'):
                                logger.warning(f"Complex double-nested filter detected for operator '{k}': {v}, flattening")
                                normalized[k] = {inner_key: {inner_inner_key: inner_inner_val}}
                                break
                    else:
                        normalized[k] = normalize_filter_dict(v)
                else:
                    normalized[k] = normalize_filter_dict(v)
            else:
                normalized[k] = normalize_filter_dict(v)
        else:
            normalized[k] = v
    return normalized

# --- TOOL DEFINITIONS ---

class SearchTools:
    """A collection of tools for interacting with the Weaviate database with unified, consistent operations."""
    def __init__(self, client: weaviate.WeaviateClient, class_name: str):
        self.client = client
        self.class_name = class_name
        self._cached_schema_fields = None

    def _normalize_filter_value(self, field: str, value: Any) -> Any:
        """Normalizes a filter value based on the schema field type, especially for dates."""
        schema = self.get_schema_fields()
        field_type = schema.get(field)
        
        # Weaviate expects RFC3339 format for dates (e.g., "2024-01-01T00:00:00Z")
        if field_type == 'date' or 'date' in field.lower():
            if isinstance(value, str):
                # Handle various date formats
                if 'T' not in value.upper():
                    # If it's a date-only string, validate and format it
                    if len(value) == 4 and value.isdigit():
                        # It's a year, convert to start of year
                        return f"{value}-01-01T00:00:00Z"
                    elif len(value) == 2 and value.isdigit():
                        # It's a 2-digit year, assume 20xx
                        return f"20{value}-01-01T00:00:00Z"
                    elif '-' in value:
                        # It's already a date string, just add time and timezone
                        return f"{value}T00:00:00Z"
                    else:
                        # Invalid date format, skip this filter
                        logger.warning(f"Invalid date format '{value}' for field '{field}', skipping filter")
                        return None
                else:
                    # Already has time, ensure it has timezone
                    if value.endswith('Z'):
                        return value  # Already in correct format
                    elif '+' in value or '-' in value[-6:]:
                        # Has timezone, convert Z to +00:00 for consistency
                        return value.replace('Z', '+00:00') if 'Z' in value else value
                    else:
                        # No timezone, add UTC
                        return value + 'Z'
        
        # For string/text fields, convert to lowercase for case-insensitive comparison
        if isinstance(value, str) and (field_type in ['text', 'string'] or field_type is None):
            return value.lower()
        
        # Return value as-is for other types or if format is already correct
        return value

    def get_schema_fields(self):
        if self._cached_schema_fields is None:
            try:
                # Direct REST call to get the schema for a specific class, bypassing client parsing issues
                response = self.client._connection.get(path=f"/schema/{self.class_name}")
                schema_json = response.json()
                props = schema_json.get('properties', [])
                self._cached_schema_fields = {prop['name']: prop['dataType'][0] for prop in props if 'dataType' in prop and prop['dataType']}
                # Ensure we always return a dictionary
                if not isinstance(self._cached_schema_fields, dict):
                    logger.warning(f"Schema fields is not a dictionary: {type(self._cached_schema_fields)}, using empty dict")
                    self._cached_schema_fields = {}
            except Exception as e:
                logger.error(f"Failed to get schema for '{self.class_name}': {e}", exc_info=True)
                self._cached_schema_fields = {}
        # Ensure we always return a dictionary
        if not isinstance(self._cached_schema_fields, dict):
            logger.warning(f"Returned schema_fields is not a dictionary: {type(self._cached_schema_fields)}, using empty dict")
            return {}
        return self._cached_schema_fields

    def _get_collection(self):
        """Get the Weaviate collection."""
        return self.client.collections.get(self.class_name)

    def _create_filters(self, filter_data: Dict[str, Any]) -> Optional[Filter]:
        """Create a Weaviate Filter object from a more standard query format, supporting nested logic."""
        if not filter_data:
            return None
        # --- PATCH: Normalize filter dicts before processing ---
        filter_data = normalize_filter_dict(filter_data)
        OPERATOR_MAP = {
            "$eq": "equal", "equal": "equal",
            "$ne": "not_equal", "notequal": "not_equal",
            "$gt": "greater_than", "greaterthan": "greater_than",
            "$gte": "greater_or_equal", "greaterthanequal": "greater_or_equal",
            "$lt": "less_than", "lessthan": "less_than",
            "$lte": "less_or_equal", "lessthanequal": "less_or_equal",
            "$in": "contains_any", "in": "contains_any",
            "$nin": "not_contains_any", "notin": "not_contains_any",
            "$like": "like", "like": "like",
            "$contains": "like",  # Map $contains to 'like' for broad text matching
        }
        filter_objects = []
        for field, value in filter_data.items():
            if field.lower() in ["$or", "$and", "$not"]:
                logical_op = field.lower()
                if logical_op == "$or":
                    nested_filters = [self._create_filters(f) for f in value if f]
                    if nested_filters: filter_objects.append(Filter.any_of(nested_filters))
                elif logical_op == "$and":
                    # --- PATCH: Handle $and operations more carefully ---
                    if isinstance(value, list):
                        nested_filters = [self._create_filters(f) for f in value if f]
                        if nested_filters: 
                            filter_objects.append(Filter.all_of(nested_filters))
                    elif isinstance(value, dict):
                        # If $and contains a dict, treat it as a single filter
                        nested_filter = self._create_filters(value)
                        if nested_filter:
                            filter_objects.append(nested_filter)
                    else:
                        logger.warning(f"Invalid $and value type: {type(value)}, skipping")
                elif logical_op == "$not":
                    # Handle NOT operations by converting to property-level negations
                    nested_filter = self._create_filters(value)
                    if nested_filter:
                        # For NOT operations, we need to handle them differently
                        # Since Filter.not_() doesn't exist, we'll skip NOT operations for now
                        # and log a warning
                        logger.warning("NOT operations in filters are not fully supported in this version. Skipping NOT filter.")
                        continue
                continue
            if isinstance(value, dict):
                # --- PATCH: Handle empty dictionaries ---
                if not value:
                    logger.warning(f"Empty filter value for field '{field}', skipping")
                    continue
                
                # Support multiple operators for a single field (e.g., date range)
                if len(value) > 1:
                    # Combine all operator filters for this field with AND
                    sub_filters = []
                    for operator, operand in value.items():
                        weaviate_op_name = OPERATOR_MAP.get(operator.lower())
                        if not weaviate_op_name:
                            raise ValueError(f"Unsupported filter operator: '{operator}'")
                        normalized_operand = self._normalize_filter_value(field, operand)
                        if normalized_operand is None:
                            # Skip this filter if normalization failed
                            continue
                        prop_filter = Filter.by_property(field)
                        if weaviate_op_name == "like":
                            # For case-insensitive text matching, convert to lowercase
                            if isinstance(normalized_operand, str):
                                normalized_operand = normalized_operand.lower()
                            sub_filters.append(prop_filter.like(f"*{normalized_operand}*"))
                        else:
                            filter_method = getattr(prop_filter, weaviate_op_name)
                            sub_filters.append(filter_method(normalized_operand))
                    if sub_filters:  # Only add if we have valid filters
                        filter_objects.append(Filter.all_of(sub_filters))
                else:
                    # --- PATCH: Ensure we have at least one item before accessing ---
                    if not value:
                        logger.warning(f"Empty filter value for field '{field}', skipping")
                        continue
                    
                    operator, operand = list(value.items())[0]
                    weaviate_op_name = OPERATOR_MAP.get(operator.lower())
                    if not weaviate_op_name:
                        raise ValueError(f"Unsupported filter operator: '{operator}'")
                    
                    # --- PATCH: Handle double-nested filters ---
                    if isinstance(operand, dict) and len(operand) == 1:
                        # This might be a double-nested filter like {'$in': {'$in': [...]}}
                        inner_operator, inner_operand = list(operand.items())[0]
                        if inner_operator.lower() in OPERATOR_MAP:
                            # Use the inner operator and operand instead
                            operator = inner_operator
                            operand = inner_operand
                            weaviate_op_name = OPERATOR_MAP.get(operator.lower())
                    
                    normalized_operand = self._normalize_filter_value(field, operand)
                    if normalized_operand is None:
                        # Skip this filter if normalization failed
                        continue
                    prop_filter = Filter.by_property(field)
                    if weaviate_op_name == "like":
                        # For case-insensitive text matching, convert to lowercase
                        if isinstance(normalized_operand, str):
                            normalized_operand = normalized_operand.lower()
                        filter_objects.append(prop_filter.like(f"*{normalized_operand}*"))
                    else:
                        filter_method = getattr(prop_filter, weaviate_op_name)
                        filter_objects.append(filter_method(normalized_operand))
            else:
                # Handles simple equality {'field': 'value'}
                normalized_value = self._normalize_filter_value(field, value)
                if normalized_value is None:
                    # Skip this filter if normalization failed
                    continue
                # For case-insensitive string comparison, convert to lowercase
                if isinstance(normalized_value, str):
                    normalized_value = normalized_value.lower()
                filter_objects.append(Filter.by_property(field).equal(normalized_value))
        if not filter_objects:
            return None
        return Filter.all_of(filter_objects) if len(filter_objects) > 1 else filter_objects[0]

    def _normalize_data_for_operation(self, data: Any, operation: str) -> List[Dict]:
        """
        Unified data normalization that automatically handles different data types
        for any operation (filter, group, sort, etc.).
        
        Args:
            data: Input data (list of dicts, list of strings, single items, etc.)
            operation: The operation being performed ("filter", "group", "sort", etc.)
            
        Returns:
            Normalized list of dictionaries ready for the operation
        """
        if data is None:
            return []
        
        # Handle string inputs that might be JSON
        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except (json.JSONDecodeError, ValueError):
                # If it's not JSON, treat as a single item
                data = [data]
        
        # Ensure data is a list
        if not isinstance(data, list):
            data = [data]
        
        # Handle different data types
        normalized = []
        for item in data:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, str):
                # Convert strings to dict format for operations
                if operation == "group":
                    # For grouping, create a dict with a generic field
                    normalized.append({"value": item})
                else:
                    # For other operations, try to use the string as-is
                    normalized.append({"value": item})
            else:
                # Convert other types to string representation
                normalized.append({"value": str(item)})
        
        return normalized

    def filter_data(self, data_source: Any, filters: Dict[str, Any], limit: int = 1000, source_type: str = "auto") -> List[Dict]:
        """
        Unified filtering that works on any data source (database or existing objects).
        Automatically detects the data source type and applies appropriate filtering.
        
        Args:
            data_source: Data to filter (can be None for database queries, or existing objects)
            filters: Filter criteria
            limit: Maximum number of results
            source_type: "database", "objects", or "auto" (auto-detect)
            
        Returns:
            Filtered results
        """
        
        # Auto-detect source type if not specified
        if source_type == "auto":
            if data_source is None:
                source_type = "database"
            else:
                source_type = "objects"
        
        if source_type == "database":
            # Filter from database
            return self._filter_from_database(filters, limit)
        else:
            # Filter existing objects
            return self._filter_existing_objects(data_source, filters, limit)

    def _filter_from_database(self, filters: Dict[str, Any], limit: int = 1000) -> List[Dict]:
        """Filter objects from the database based on the given criteria."""
        
        # Normalize filters
        filters = normalize_filter_dict(filters)
        collection = self._get_collection()
        weaviate_filter = self._create_filters(filters)
        
        response = collection.query.fetch_objects(
            filters=weaviate_filter,
            limit=limit
        )
        
        return [obj.properties for obj in response.objects]

    def _filter_existing_objects(self, objects: List[Dict], filters: Dict[str, Any], limit: int = 1000) -> List[Dict]:
        """Filter existing objects based on the given criteria."""
        # Normalize inputs
        objects = self._normalize_data_for_operation(objects, "filter")
        
        if not objects:
            return []
        
        # Normalize filters
        filters = normalize_filter_dict(filters)
        filtered_objects = []
        
        import re
        import urllib.parse
        
        for obj in objects:
            # Check if object matches all filter conditions
            matches = True
            for field, condition in filters.items():
                if field.lower() in ["$and", "$or", "$not"]:
                    # Handle logical operators for existing objects
                    if field.lower() == "$not":
                        # For NOT operations, we need to handle them differently
                        if isinstance(condition, dict):
                            # Simple NOT case - check if field doesn't match
                            for not_field, not_condition in condition.items():
                                if isinstance(not_condition, dict):
                                    # Prefer $like > $contains > $eq if multiple present
                                    op_priority = ['$like', '$contains', '$eq']
                                    op_to_use = None
                                    for op in op_priority:
                                        if op in not_condition:
                                            op_to_use = op
                                            val = not_condition[op]
                                            break
                                    if not op_to_use:
                                        # Fallback to first operator
                                        op_to_use, val = list(not_condition.items())[0]
                                    # Convert both values to lowercase for case-insensitive comparison
                                    obj_value = str(obj.get(not_field, "")).lower()
                                    val = str(val).lower()
                                    if op_to_use == "$eq" and obj_value == val:
                                        matches = False
                                        break
                                    elif op_to_use == "$contains" and val in obj_value:
                                        matches = False
                                        break
                                    elif op_to_use == "$like":
                                        value_decoded = urllib.parse.unquote(str(val))
                                        pattern = '^' + re.escape(value_decoded).replace('%', '.*').replace('_', '.') + '$'
                                        if re.match(pattern, obj_value):
                                            matches = False
                                            break
                                    elif op_to_use == "$in" and obj_value in [str(v).lower() for v in val]:
                                        matches = False
                                        break
                                    elif op_to_use in ["$gt", "$gte", "$lt", "$lte"]:
                                        # Handle date/numeric comparisons for NOT operations
                                        try:
                                            # For date comparisons, use original values without lowercase conversion
                                            original_obj_value = obj.get(not_field, "")
                                            original_value = not_condition[op_to_use]
                                            
                                            if 'date' in not_field.lower() or 'time' in not_field.lower() or (isinstance(original_obj_value, str) and 'T' in original_obj_value):
                                                from datetime import datetime
                                                if isinstance(original_obj_value, str):
                                                    if 'T' in original_obj_value:
                                                        if original_obj_value.endswith('Z'):
                                                            obj_value_parsed = datetime.fromisoformat(original_obj_value.replace('Z', '+00:00'))
                                                        elif '+' in original_obj_value or '-' in original_obj_value[-6:]:
                                                            obj_value_parsed = datetime.fromisoformat(original_obj_value)
                                                        else:
                                                            obj_value_parsed = datetime.fromisoformat(original_obj_value + '+00:00')
                                                    else:
                                                        obj_value_parsed = datetime.fromisoformat(original_obj_value + 'T00:00:00+00:00')
                                                else:
                                                    obj_value_parsed = original_obj_value
                                                    
                                                if isinstance(original_value, str):
                                                    if 'T' in original_value:
                                                        if original_value.endswith('Z'):
                                                            value_parsed = datetime.fromisoformat(original_value.replace('Z', '+00:00'))
                                                        elif '+' in original_value or '-' in original_value[-6:]:
                                                            value_parsed = datetime.fromisoformat(original_value)
                                                        else:
                                                            value_parsed = datetime.fromisoformat(original_value + '+00:00')
                                                    else:
                                                        value_parsed = datetime.fromisoformat(original_value + 'T00:00:00+00:00')
                                                else:
                                                    value_parsed = original_value
                                            else:
                                                # For non-date fields, use lowercase string comparison
                                                obj_value_parsed = obj_value
                                                value_parsed = val
                                                
                                            if op_to_use == "$gt" and obj_value_parsed > value_parsed:
                                                matches = False
                                                break
                                            elif op_to_use == "$gte" and obj_value_parsed >= value_parsed:
                                                matches = False
                                                break
                                            elif op_to_use == "$lt" and obj_value_parsed < value_parsed:
                                                matches = False
                                                break
                                            elif op_to_use == "$lte" and obj_value_parsed <= value_parsed:
                                                matches = False
                                                break
                                        except (TypeError, ValueError) as e:
                                            logger.warning(f"Date comparison failed for field '{not_field}': {e}")
                                            matches = False
                                            break
                    continue
                if isinstance(condition, dict):
                    # Prefer $like > $contains > $eq if multiple present
                    op_priority = ['$like', '$contains', '$eq']
                    op_to_use = None
                    for op in op_priority:
                        if op in condition:
                            op_to_use = op
                            value = condition[op]
                            break
                    if not op_to_use:
                        # Fallback to first operator
                        op_to_use, value = list(condition.items())[0]
                    
                    # Convert both values to lowercase for case-insensitive comparison
                    obj_value = str(obj.get(field, "")).lower()
                    value = str(value).lower()
                    
                    if op_to_use == "$eq":
                        if obj_value != value:
                            matches = False
                    elif op_to_use == "$ne":
                        if obj_value == value:
                            matches = False
                    elif op_to_use == "$contains":
                        if value not in obj_value:
                            matches = False
                    elif op_to_use == "$like":
                        value_decoded = urllib.parse.unquote(str(value))
                        pattern = '^' + re.escape(value_decoded).replace('%', '.*').replace('_', '.') + '$'
                        if not re.match(pattern, obj_value):
                            matches = False
                    elif op_to_use == "$in":
                        if obj_value not in [str(v).lower() for v in value]:
                            matches = False
                    elif op_to_use == "$nin":
                        if obj_value in [str(v).lower() for v in value]:
                            matches = False
                    elif op_to_use in ["$gt", "$gte", "$lt", "$lte"]:
                        # Handle date/numeric comparisons
                        try:
                            # For date comparisons, use original values without lowercase conversion
                            original_obj_value = obj.get(field, "")
                            original_value = condition[op_to_use]
                            
                            if 'date' in field.lower() or 'time' in field.lower() or (isinstance(original_obj_value, str) and 'T' in original_obj_value):
                                from datetime import datetime
                                if isinstance(original_obj_value, str):
                                    if 'T' in original_obj_value:
                                        if original_obj_value.endswith('Z'):
                                            obj_value_parsed = datetime.fromisoformat(original_obj_value.replace('Z', '+00:00'))
                                        elif '+' in original_obj_value or '-' in original_obj_value[-6:]:
                                            obj_value_parsed = datetime.fromisoformat(original_obj_value)
                                        else:
                                            obj_value_parsed = datetime.fromisoformat(original_obj_value + '+00:00')
                                    else:
                                        obj_value_parsed = datetime.fromisoformat(original_obj_value + 'T00:00:00+00:00')
                                else:
                                    obj_value_parsed = original_obj_value
                                    
                                if isinstance(original_value, str):
                                    if 'T' in original_value:
                                        if original_value.endswith('Z'):
                                            value_parsed = datetime.fromisoformat(original_value.replace('Z', '+00:00'))
                                        elif '+' in original_value or '-' in original_value[-6:]:
                                            value_parsed = datetime.fromisoformat(original_value)
                                        else:
                                            value_parsed = datetime.fromisoformat(original_value + '+00:00')
                                    else:
                                        value_parsed = datetime.fromisoformat(original_value + 'T00:00:00+00:00')
                                else:
                                    value_parsed = original_value
                            else:
                                # For non-date fields, try numeric comparison first, then fall back to string
                                try:
                                    obj_value_parsed = float(obj_value) if obj_value.replace('.', '').replace('-', '').isdigit() else obj_value
                                    value_parsed = float(value) if value.replace('.', '').replace('-', '').isdigit() else value
                                except (ValueError, TypeError):
                                    # If numeric conversion fails, use string comparison
                                    obj_value_parsed = obj_value
                                    value_parsed = value
                            if op_to_use == "$gt":
                                if obj_value_parsed <= value_parsed:
                                    matches = False
                            elif op_to_use == "$gte":
                                if obj_value_parsed < value_parsed:
                                    matches = False
                            elif op_to_use == "$lt":
                                if obj_value_parsed >= value_parsed:
                                    matches = False
                            elif op_to_use == "$lte":
                                if obj_value_parsed > value_parsed:
                                    matches = False
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Date comparison failed for field '{field}': {e}")
                            matches = False
                else:
                    # Simple equality - convert both to lowercase for case-insensitive comparison
                    obj_value = str(obj.get(field, "")).lower()
                    condition_value = str(condition).lower()
                    if obj_value != condition_value:
                        matches = False
                if not matches:
                    break
            if matches:
                filtered_objects.append(obj)
                if len(filtered_objects) >= limit:
                    break
        return filtered_objects

    def group_data(self, data: Any, group_field: str, limit: Optional[int] = None, 
                  include_items: bool = False, max_items_per_group: int = 3) -> List[Dict]:
        """
        Unified grouping that works on any data structure.
        Automatically normalizes data and handles different input types.
        
        Args:
            data: Data to group (list of dicts, list of strings, etc.)
            group_field: Field name to group by
            limit: Maximum number of groups to return
            include_items: Whether to include item details in the results
            max_items_per_group: Maximum number of items to include per group
            
        Returns:
            List of dictionaries with group information
        """
        # Normalize data for grouping
        objects = self._normalize_data_for_operation(data, "group")
        
        if not objects:
            return []
        
        groups = {}
        
        for obj in objects:
            group_key = obj.get(group_field)
            if not group_key:  # Skip objects without the group field
                continue
                
            if group_key not in groups:
                groups[group_key] = {
                    'group_name': group_key,
                    'item_count': 0,
                    'items': [],
                    'field_values': {},  # Track values of other fields
                    'latest_date': None
                }
            
            # Update group information
            group = groups[group_key]
            group['item_count'] += 1
            
            # Add item information if requested
            if include_items and len(group['items']) < max_items_per_group:
                # Create a generic item info with all available fields
                item_info = {}
                for field, value in obj.items():
                    if field != group_field:  # Don't duplicate the group field
                        item_info[field] = value
                
                # Add search metadata if available
                if '_search_metadata' in obj:
                    item_info['_search_metadata'] = obj['_search_metadata']
                
                group['items'].append(item_info)
            
            # Track values of other fields for diversity analysis
            for field, value in obj.items():
                if field != group_field and value is not None:
                    if field not in group['field_values']:
                        group['field_values'][field] = set()
                    group['field_values'][field].add(str(value))
            
            # Track latest date (try to find a date field)
            for field, value in obj.items():
                if isinstance(value, str) and ('date' in field.lower() or 'time' in field.lower()):
                    try:
                        # Try to parse as date
                        from datetime import datetime
                        if 'T' in value:
                            parsed_date = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        else:
                            parsed_date = datetime.fromisoformat(value)
                        
                        if group['latest_date'] is None or parsed_date > group['latest_date']:
                            group['latest_date'] = parsed_date
                    except (ValueError, TypeError):
                        # Not a valid date, continue
                        continue
        
        # Convert to list and sort by item count (descending)
        result = list(groups.values())
        result.sort(key=lambda x: x['item_count'], reverse=True)
        
        # Convert sets to lists for JSON serialization
        for group in result:
            for field in group['field_values']:
                group['field_values'][field] = list(group['field_values'][field])
            
            # Sort items by date if available
            if 'items' in group and group['items']:
                # Try to find a date field in the items
                date_field = None
                for item in group['items']:
                    for field in item:
                        if 'date' in field.lower() or 'time' in field.lower():
                            date_field = field
                            break
                    if date_field:
                        break
                
                if date_field:
                    group['items'].sort(
                        key=lambda x: x.get(date_field) or '', 
                        reverse=True
                    )
        
        # Apply limit if specified
        if limit is not None:
            result = result[:limit]
            
        return result

    def sort_data(self, data: Any, sort_field: str, ascending: bool = True) -> List[Dict]:
        """
        Unified sorting that works on any data structure.
        Automatically normalizes data and handles different input types.
        
        Args:
            data: Data to sort (list of dicts, list of strings, etc.)
            sort_field: Field name to sort by
            ascending: Whether to sort in ascending order
            
        Returns:
            Sorted list of objects
        """
        # Normalize data for sorting
        objects = self._normalize_data_for_operation(data, "sort")
        
        if not objects:
            return []
            
        return sorted(
            objects,
            key=lambda x: x.get(sort_field, ""),
            reverse=not ascending
        )

    def extract_field_values(self, data: Any, field_name: str, limit: int = 10) -> List[Any]:
        """
        Extract values for a specific field from any data structure.
        Automatically handles different input types.
        
        Args:
            data: Input data (list of dicts, single dict, or other)
            field_name: Name of the field to extract
            limit: Maximum number of values to return
            
        Returns:
            List of extracted values (not wrapped in dictionaries)
        """
        # Normalize data
        objects = self._normalize_data_for_operation(data, "extract")
        
        if not objects:
            return []
        
        values = []
        for obj in objects:
            if isinstance(obj, dict) and field_name in obj:
                values.append(obj[field_name])
            elif hasattr(obj, field_name):
                values.append(getattr(obj, field_name))
        
        # Return unique values up to limit
        unique_values = []
        seen = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                unique_values.append(value)
                if len(unique_values) >= limit:
                    break
        
        return unique_values

    def create_filter_from_values(self, values: List[Any], target_field: str, limit: int = 5) -> Dict[str, Any]:
        """
        Create a filter from a list of values.
        Unified method that works with any value list (from groups, extracted values, etc.).
        
        Args:
            values: List of values to filter by
            target_field: Field name to filter on in the target collection
            limit: Maximum number of values to include in the filter
            
        Returns:
            Filter dictionary ready for filter_data
        """
        if not values:
            logger.warning("No values provided to create_filter_from_values")
            return {}
        
        # Normalize values
        if not isinstance(values, list):
            values = [values]
        
        # Extract actual values (handle group objects, strings, etc.)
        actual_values = []
        for value in values[:limit]:
            if isinstance(value, dict):
                # If it's a group object, extract the group name
                if 'group_name' in value:
                    actual_values.append(value['group_name'])
                else:
                    # Try to find any string value in the dict
                    for k, v in value.items():
                        if isinstance(v, str) and k != target_field:  # Don't use the field name itself
                            actual_values.append(v)
                            break
            elif isinstance(value, str):
                # Don't add the field name itself as a value
                if value != target_field:
                    actual_values.append(value)
            else:
                str_value = str(value)
                if str_value != target_field:
                    actual_values.append(str_value)
        
        if not actual_values:
            logger.warning("No valid values found to create filter")
            return {}
        
        # Create filter
        if len(actual_values) == 1:
            filter_result = {target_field: {"$eq": actual_values[0]}}
        else:
            filter_result = {target_field: {"$in": actual_values}}
        
        return filter_result

    def combine_results(self, results_list: List[List[Dict]], deduplicate: bool = True) -> List[Dict]:
        """
        Combine multiple result lists into one. Useful for merging results from different operations.
        
        Args:
            results_list: List of result lists to combine
            deduplicate: Whether to remove duplicate items based on a unique field
            
        Returns:
            Combined list of results
        """
        if not results_list:
            logger.warning("combine_results: No results_list provided")
            return []
        
        # Flatten the list
        combined = []
        for i, results in enumerate(results_list):
            if results:
                normalized_results = self._normalize_data_for_operation(results, "combine")
                combined.extend(normalized_results)
        
        if not deduplicate:
            return combined
        
        # Try to deduplicate based on common unique fields
        seen = set()
        unique_results = []
        
        for item in combined:
            if isinstance(item, dict):
                # Try to create a unique key
                unique_key = None
                if 'id' in item:
                    unique_key = item['id']
                elif 'uuid' in item:
                    unique_key = item['uuid']
                elif 'title' in item and 'author' in item:
                    unique_key = f"{item['title']}_{item['author']}"
                elif 'title' in item:
                    unique_key = item['title']
                else:
                    # Use the entire item as key (less efficient but works)
                    unique_key = str(item)
                
                if unique_key not in seen:
                    seen.add(unique_key)
                    unique_results.append(item)
            else:
                # For non-dict items, just add them
                unique_results.append(item)
        
        return unique_results

    def get_top_n(self, data: Any, n: int = 5) -> List[Dict]:
        """Get the top N objects from any data structure."""
        objects = self._normalize_data_for_operation(data, "top_n")
        return objects[:n] if objects else []

    def get_distinct_values(self, data: Any, field: str, limit: int = 10) -> List[Any]:
        """Get distinct values for a field from any data structure."""
        return self.extract_field_values(data, field, limit)

    def find_item_by_field(self, field_name: str, field_value: str) -> Optional[str]:
        """Find an item by a specific field value and return its Weaviate ID (UUID)."""
        collection = self._get_collection()
        response = collection.query.fetch_objects(
            filters=Filter.by_property(field_name).equal(field_value),
            limit=1
        )
        if response.objects:
            return response.objects[0].uuid
        return None

    def semantic_search(self, query: str, limit: int = 10, filter_dict: Optional[Dict] = None, 
                      after_date: str = "2023-01-01T00:00:00Z", include_metadata: bool = True, reference_id: Optional[str] = None, score_threshold: float = 0.3) -> List[Dict]:
        """
        Perform a semantic search with enhanced filtering and date range support.
        If reference_id is provided, use nearObject for similarity search.
        Only include results with score >= score_threshold (default 0.3).
        """
        try:
            collection = self._get_collection()
            date_field = None
            # Dynamically find a date field if present
            for field, dtype in self.get_schema_fields().items():
                if dtype == 'date' or 'date' in field.lower():
                    date_field = field
                    break
            if date_field:
                date_filter = Filter.by_property(date_field).greater_than(after_date)
            else:
                date_filter = None
            if filter_dict:
                filter_dict = normalize_filter_dict(filter_dict)
            if filter_dict and date_filter:
                search_filter = Filter.all_of([date_filter, self._create_filters(filter_dict)])
            elif filter_dict:
                search_filter = self._create_filters(filter_dict)
            elif date_filter:
                search_filter = date_filter
            else:
                search_filter = None
            # Dynamically build query_properties from all text fields
            schema_fields = self.get_schema_fields()
            text_fields = [k for k, v in schema_fields.items() if v in ['text', 'string']]
            query_properties = [f"{text_fields[0]}^2"] + text_fields[1:] if text_fields else []
            # --- PATCH: Use nearObject if reference_id is provided ---
            if reference_id:
                response = collection.query.near_object(
                    id=reference_id,
                    limit=limit,
                    filters=search_filter
                )
                results = [{
                    **obj.properties,
                    "_search_metadata": {
                        "score": getattr(obj.metadata, 'score', None),
                        "distance": getattr(obj.metadata, 'distance', None),
                        "explain_score": getattr(obj.metadata, 'explain_score', None)
                    }
                } for obj in response.objects]
                if include_metadata and score_threshold is not None:
                    results = [r for r in results if r.get('_search_metadata', {}).get('score', 0) is not None and r['_search_metadata']['score'] >= score_threshold]
                if results:
                    return results
            try:
                response = collection.query.hybrid(
                    query=query,
                    limit=limit,
                    query_properties=query_properties,
                    return_metadata=["score", "distance"],
                    filters=search_filter
                )
                if include_metadata:
                    results = [{
                        **obj.properties,
                        "_search_metadata": {
                            "score": obj.metadata.score,
                            "distance": getattr(obj.metadata, 'distance', None),
                            "explain_score": getattr(obj.metadata, 'explain_score', None)
                        }
                    } for obj in response.objects]
                    if score_threshold is not None:
                        results = [r for r in results if r.get('_search_metadata', {}).get('score', 0) is not None and r['_search_metadata']['score'] >= score_threshold]
                else:
                    results = [obj.properties for obj in response.objects]
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to BM25: {str(e)}")
            response = collection.query.bm25(
                query=query,
                limit=limit,
                query_properties=query_properties,
                return_metadata=["score"],
                filters=search_filter
            )
            if include_metadata:
                results = [{
                    **obj.properties,
                    "_search_metadata": {
                        "score": obj.metadata.score,
                        "distance": None,
                        "explain_score": None
                    }
                } for obj in response.objects]
                if score_threshold is not None:
                    results = [r for r in results if r.get('_search_metadata', {}).get('score', 0) is not None and r['_search_metadata']['score'] >= score_threshold]
            else:
                results = [obj.properties for obj in response.objects]
            return results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            return []

    def transform_results(self, objects: List[Dict], transform_type: str = "flatten", **kwargs) -> List[Dict]:
        """
        Transform results into different formats. Useful for data preparation.
        
        Args:
            objects: List of objects to transform
            transform_type: Type of transformation ("flatten", "select_fields", "rename_fields", "convert_to_groups")
            **kwargs: Additional parameters for the transformation
            
        Returns:
            Transformed results
        """
        # Normalize input
        objects = self._normalize_data_for_operation(objects, "transform")
        if not objects:
            return []
        
        if transform_type == "flatten":
            # Flatten nested structures
            flattened = []
            for obj in objects:
                if isinstance(obj, dict):
                    flat_obj = {}
                    for key, value in obj.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                flat_obj[f"{key}_{sub_key}"] = sub_value
                        elif isinstance(value, list):
                            flat_obj[f"{key}_count"] = len(value)
                            if value and isinstance(value[0], dict):
                                # Add first item's fields
                                for sub_key, sub_value in value[0].items():
                                    flat_obj[f"{key}_{sub_key}"] = sub_value
                        else:
                            flat_obj[key] = value
                    flattened.append(flat_obj)
                else:
                    flattened.append(obj)
            return flattened
        
        elif transform_type == "select_fields":
            # Select only specific fields
            fields = kwargs.get('fields', [])
            if not fields:
                return objects
            
            selected = []
            for obj in objects:
                if isinstance(obj, dict):
                    selected_obj = {field: obj.get(field) for field in fields if field in obj}
                    selected.append(selected_obj)
                else:
                    selected.append(obj)
            return selected
        
        elif transform_type == "rename_fields":
            # Rename fields
            field_mapping = kwargs.get('field_mapping', {})
            if not field_mapping:
                return objects
            
            renamed = []
            for obj in objects:
                if isinstance(obj, dict):
                    renamed_obj = {}
                    for old_name, new_name in field_mapping.items():
                        if old_name in obj:
                            renamed_obj[new_name] = obj[old_name]
                        else:
                            # Keep original field if mapping doesn't exist
                            renamed_obj[old_name] = obj.get(old_name)
                    renamed.append(renamed_obj)
                else:
                    renamed.append(obj)
            return renamed
        
        elif transform_type == "convert_to_groups":
            # Convert a list of strings to group format for create_filter_from_group_results
            field_name = kwargs.get('field_name', 'value')
            if not objects:
                return []
            
            # Convert each string to a group object
            groups = []
            for value in objects:
                if isinstance(value, dict):
                    # If it's already a dict, try to extract the value
                    if field_name in value:
                        groups.append({
                            'group_name': value[field_name],
                            'item_count': 1,
                            'items': [],
                            'field_values': {}
                        })
                elif isinstance(value, str):
                    groups.append({
                        'group_name': value,
                        'item_count': 1,  # Each value represents one item
                        'items': [],
                        'field_values': {}
                    })
            
            return groups
        
        else:
            logger.warning(f"Unknown transform_type: {transform_type}")
            return objects

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
        "get_distinct_values": {
            "func": search_tools.get_distinct_values,
            "description": "Gets distinct values for a field from any data structure.",
            "parameters": ["data", "field", "limit"]},
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

class QueryDecomposer:
    """Decomposes a user query into a multi-step execution plan using an LLM."""
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key must be provided")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")

    def decompose(self, query: str, class_name: str, schema_fields: dict, tool_registry: dict) -> Dict[str, Any]:
        try:
            # Ensure schema_fields is a dictionary - handle all edge cases
            if not isinstance(schema_fields, dict):
                logger.warning(f"schema_fields is not a dictionary in decompose: {type(schema_fields)}, using empty dict")
                schema_fields = {}
            
            # Ensure tool_registry is a dictionary - handle all edge cases
            if not isinstance(tool_registry, dict):
                logger.warning(f"tool_registry is not a dictionary in decompose: {type(tool_registry)}, using empty dict")
                tool_registry = {}
            
            schema_str = "\n".join([f"- {k}: {v}" for k, v in schema_fields.items()])
            tool_list_str = "\n".join([f"- {name}: {meta['description']} (Parameters: {', '.join(meta['parameters'])})" for name, meta in tool_registry.items()])
            operator_hints = """
Operator hints:
- For text fields: use $contains, $like, $eq
- For string fields: use $eq, $in
- For date fields: use $gt, $lt, $gte, $lte, $eq
- For numeric fields: use $gt, $lt, $gte, $lte, $eq, $in
- For all fields: $and, $or, $not can be used for combining conditions
"""
            # --- PATCH: Add explicit instructions and examples for both semantic and pure filter/group queries ---
            prompt = f"""
You are a query planner for a research database. For each user query, output a JSON plan with a sequence of steps.

- If the user query is about meaning, similarity, relatedness, or is subjective (e.g., 'about', 'related to', 'similar to', 'discuss', 'find items about ...'), start your plan with a 'semantic_search' step.
- If the user query is a pure field-based, grouping, or counting query (e.g., 'group all items by field', 'count how many match criteria', 'list all items where ...'), use 'filter_data', 'group_data', and other structured tools as appropriate. Do NOT use 'semantic_search' for these queries.
- Only use 'semantic_search' if the query requires semantic understanding or similarity.
- Use 'filter_data' for all filtering operations (both database and existing objects).
- Use 'group_data' for grouping operations.
- Use 'sort_data' for sorting operations.
- IMPORTANT: Do NOT use 'find_article_by_title' unless you have the exact title. Instead, use 'semantic_search' to find items and then use 'filter_data' to filter by any criteria.

**CRITICAL: Use the new unified tool names and parameters:**
- Use 'filter_data' instead of 'filter_objects' or 'filter_existing_objects'
- Use 'group_data' instead of 'group_by'
- Use 'sort_data' instead of 'sort_objects'
- Use 'create_filter_from_values' instead of 'create_filter_from_group_results'
- Parameter names: 'data' instead of 'objects', 'data_source' instead of 'objects' for filters

**CRITICAL: For author counting queries, use direct grouping instead of extract_field_values:**
- Instead of: filter_data → extract_field_values → group_data (which counts each author as 1)
- Use: filter_data → group_data (which counts actual article counts per author)
- Example: "Get all articles, group by author to count articles" should use:
  1. filter_data (get all articles)
  2. group_data (group by author field directly)

**CRITICAL: For complex filtering, use multiple filter_data steps instead of complex filters:**
- If you need to filter by multiple criteria, use separate filter_data steps
- Example: Instead of {{"$and": [{{"field1": {{"$eq": "value1"}}}}, {{"field2": {{"$gt": "2020-01-01"}}}}]}}, use:
  1. filter_data with {{"field1": {{"$eq": "value1"}}}} (source_type: "database")
  2. filter_data with {{"field2": {{"$gt": "2020-01-01"}}}} (source_type: "objects")
- This prevents complex nested filter issues and is more reliable.

**CRITICAL: Avoid NOT operations in filters:**
- Do NOT use $not operations in filters as they are not well supported
- Instead of filtering out items with "review" in title, use semantic_search to find relevant items
- Or use filter_data to get all items, then use filter_data again to remove unwanted ones
- Example: Instead of {{"title": {{"$not": {{"$contains": "review"}}}}}}, use:
  1. filter_data to get all articles (source_type: "database")
  2. filter_data to remove articles with "review" in title (source_type: "objects")

**CRITICAL: After group_data operations, field names change:**
- When you group by any field 'X', the result has 'group_name' instead of 'X'
- The count field is called 'item_count', not 'count'
- So after group_data, use '$PREV_STEP_RESULT[0].group_name' not '$PREV_STEP_RESULT[0].original_field_name'

**CRITICAL: After extract_field_values, use 'value' as group_field:**
- When you extract field values and then group them, use 'group_field': 'value'
- This is because extract_field_values returns strings that get normalized to dicts with a 'value' field
- Example: extract_field_values with field_name: "author" → group_data with group_field: "value"

**CRITICAL: Tool parameters format:**
- Only include the actual tool parameters in the "parameters" object
- Do NOT include "description" in the parameters - that's only for documentation
- Each tool's parameters are listed above in the tool descriptions
- Example: filter_data only accepts: data_source, filters, limit, source_type

Available step types:
- semantic_search: for semantic search (optionally using reference_id)
- filter_data: for all filtering operations (database or existing objects)
- group_data: for grouping/aggregation
- sort_data: for sorting
- get_top_n: for limiting results
- find_item_by_field: for fetching an item's ID by any field value (use sparingly)
- extract_field_values: for extracting specific field values from results
- create_filter_from_values: for creating filters from any value list
- combine_results: for merging multiple result sets
- transform_results: for transforming data formats

Schema for collection '{class_name}':
{schema_str}

{operator_hints}

The available tools are:
{tool_list_str}

Decompose the user's query into a JSON object with a 'steps' array. Each step is a tool call. Use '$PREV_STEP_RESULT' to pass results between steps.

---
Example Query 1: "Find space missions similar to Apollo 11 but only if they were successful"
Example JSON Output:
```json
{{
  "steps": [
    {{ "step_id": 1, "tool": "find_item_by_field", "parameters": {{"field_name": "mission_name", "field_value": "Apollo 11"}}, "description": "Find the Apollo 11 mission by name." }},
    {{ "step_id": 2, "tool": "semantic_search", "parameters": {{"reference_id": "$PREV_STEP_RESULT", "limit": 100}}, "description": "Find missions similar to Apollo 11." }},
    {{ "step_id": 3, "tool": "filter_data", "parameters": {{"data_source": "$PREV_STEP_RESULT", "filters": {{"status": {{"$eq": "success"}}}}, "source_type": "objects"}}, "description": "Filter for successful missions only." }}
  ]
}}
```

Example Query 2: "Get all products, filter out those with 'test' in the name, then group by category"
Example JSON Output:
```json
{{
  "steps": [
    {{ "step_id": 1, "tool": "filter_data", "parameters": {{"data_source": null, "filters": {{}}, "source_type": "database"}}, "description": "Get all products" }},
    {{ "step_id": 2, "tool": "filter_data", "parameters": {{"data_source": "$PREV_STEP_RESULT", "filters": {{"name": {{"$not": {{"$contains": "test"}}}}}}, "source_type": "objects"}}, "description": "Filter out products with 'test' in name" }},
    {{ "step_id": 3, "tool": "group_data", "parameters": {{"data": "$PREV_STEP_RESULT", "group_field": "category"}}, "description": "Group by category" }}
  ]
}}
```

Example Query 3: "Find users who registered between 2020 and 2023 and are active"
Example JSON Output:
```json
{{
  "steps": [
    {{ "step_id": 1, "tool": "filter_data", "parameters": {{"data_source": null, "filters": {{"registration_date": {{"$gte": "2020-01-01"}}}}, "source_type": "database"}}, "description": "Filter users registered after 2020" }},
    {{ "step_id": 2, "tool": "filter_data", "parameters": {{"data_source": "$PREV_STEP_RESULT", "filters": {{"registration_date": {{"$lte": "2023-12-31"}}}}, "source_type": "objects"}}, "description": "Filter users registered before 2024" }},
    {{ "step_id": 3, "tool": "filter_data", "parameters": {{"data_source": "$PREV_STEP_RESULT", "filters": {{"status": {{"$eq": "active"}}}}, "source_type": "objects"}}, "description": "Filter for active users only" }}
  ]
}}
```

Example Query 4: "Find articles about AI and machine learning, combine the results, group by author, then find similar articles to the most productive author"
Example JSON Output:
```json
{{
  "steps": [
    {{ "step_id": 1, "tool": "semantic_search", "parameters": {{"query": "AI", "limit": 50}}, "description": "Find articles about AI" }},
    {{ "step_id": 2, "tool": "semantic_search", "parameters": {{"query": "machine learning", "limit": 50}}, "description": "Find articles about machine learning" }},
    {{ "step_id": 3, "tool": "combine_results", "parameters": {{"results_list": ["$PREV_STEP_RESULT[0]", "$PREV_STEP_RESULT[1]"], "deduplicate": true}}, "description": "Combine AI and ML results" }},
    {{ "step_id": 4, "tool": "group_data", "parameters": {{"data": "$PREV_STEP_RESULT", "group_field": "author"}}, "description": "Group by author" }},
    {{ "step_id": 5, "tool": "sort_data", "parameters": {{"data": "$PREV_STEP_RESULT", "sort_field": "item_count", "ascending": false}}, "description": "Sort by article count" }},
    {{ "step_id": 6, "tool": "get_top_n", "parameters": {{"data": "$PREV_STEP_RESULT", "n": 1}}, "description": "Get the most productive author" }},
    {{ "step_id": 7, "tool": "extract_field_values", "parameters": {{"data": "$PREV_STEP_RESULT", "field_name": "group_name", "limit": 1}}, "description": "Extract the author name" }},
    {{ "step_id": 8, "tool": "semantic_search", "parameters": {{"query": "$PREV_STEP_RESULT[0]", "limit": 20}}, "description": "Find similar articles to the most productive author" }}
  ]
}}
```

Example Query 5: "Get all space missions, group by agency to count missions, and show the top 3 agencies with most missions"
Example JSON Output:
```json
{{
  "steps": [
    {{ "step_id": 1, "tool": "filter_data", "parameters": {{"data_source": null, "filters": {{}}, "source_type": "database"}}, "description": "Get all space missions" }},
    {{ "step_id": 2, "tool": "group_data", "parameters": {{"data": "$PREV_STEP_RESULT", "group_field": "agency"}}, "description": "Group by agency to count missions" }},
    {{ "step_id": 3, "tool": "sort_data", "parameters": {{"data": "$PREV_STEP_RESULT", "sort_field": "item_count", "ascending": false}}, "description": "Sort by mission count" }},
    {{ "step_id": 4, "tool": "get_top_n", "parameters": {{"data": "$PREV_STEP_RESULT", "n": 3}}, "description": "Get top 3 agencies" }}
  ]
}}
```

Example Query 6: "Find products with price between $100 and $500, group by brand, then find similar products to the most popular brand"
Example JSON Output:
```json
{{
  "steps": [
    {{ "step_id": 1, "tool": "filter_data", "parameters": {{"data_source": null, "filters": {{"price": {{"$gte": 100}}}}, "source_type": "database"}}, "description": "Filter products with price >= $100" }},
    {{ "step_id": 2, "tool": "filter_data", "parameters": {{"data_source": "$PREV_STEP_RESULT", "filters": {{"price": {{"$lte": 500}}}}, "source_type": "objects"}}, "description": "Filter products with price <= $500" }},
    {{ "step_id": 3, "tool": "group_data", "parameters": {{"data": "$PREV_STEP_RESULT", "group_field": "brand"}}, "description": "Group by brand" }},
    {{ "step_id": 4, "tool": "sort_data", "parameters": {{"data": "$PREV_STEP_RESULT", "sort_field": "item_count", "ascending": false}}, "description": "Sort by product count" }},
    {{ "step_id": 5, "tool": "get_top_n", "parameters": {{"data": "$PREV_STEP_RESULT", "n": 1}}, "description": "Get the most popular brand" }},
    {{ "step_id": 6, "tool": "extract_field_values", "parameters": {{"data": "$PREV_STEP_RESULT", "field_name": "group_name", "limit": 1}}, "description": "Extract the brand name" }},
    {{ "step_id": 7, "tool": "semantic_search", "parameters": {{"query": "$PREV_STEP_RESULT[0]", "limit": 20}}, "description": "Find similar products to the most popular brand" }}
  ]
}}
```

---
Query: "{query}"
JSON Output:
"""
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```", 1)[1].strip()
            plan = json.loads(response_text)
            logger.info(f"Decomposed query into plan: {json.dumps(plan, indent=2)}")
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error decomposing query with LLM: {e}")
            raise  # Re-raise to be caught by the node's fallback logic
        except Exception as e:
            logger.error(f"Unexpected error in decompose: {e}", exc_info=True)
            raise

# --- LANGGRAPH AGENT SETUP ---

def flatten_results(results):
    """Recursively flatten a list of results (to handle nested lists)."""
    flat = []
    for item in results:
        if isinstance(item, list):
            flat.extend(flatten_results(item))
        else:
            flat.append(item)
    return flat

class AgentState(TypedDict):
    original_query: str
    plan: Dict[str, Any]
    current_step: int
    intermediate_results: List[Any]
    final_results: List[Dict]
    summary: str


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
            "get_distinct_values": { "func": self.search_tools.get_distinct_values, "description": "Gets distinct values for a field from any data structure.", "parameters": ["data", "field", "limit"]},
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

# Conditional Edge
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
