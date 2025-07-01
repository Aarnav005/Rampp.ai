import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import urllib.parse
import re
import weaviate
from weaviate.classes.query import Filter

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                            logging.warning(f"NOT contains/like operations are not supported for field '{field}', skipping filter")
                            continue
                        else:
                            # For other operators, skip this filter
                            logging.warning(f"Unsupported NOT operator '{operator}' for field '{field}', skipping filter")
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
                logging.warning(f"Empty filter value for field '{k}', skipping")
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
                        logging.warning(f"Double-nested filter detected for operator '{k}': {v}, flattening to {inner_operator}: {inner_operand}")
                        normalized[k] = {inner_operator: inner_operand}
                        continue
                # Check for more complex double-nesting patterns
                if isinstance(operand, dict):
                    for inner_key, inner_val in operand.items():
                        if isinstance(inner_val, dict) and len(inner_val) == 1:
                            inner_inner_key, inner_inner_val = list(inner_val.items())[0]
                            if inner_inner_key.startswith('$') and k.startswith('$'):
                                logging.warning(f"Complex double-nested filter detected for operator '{k}': {v}, flattening")
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