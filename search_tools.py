import logging
from typing import Dict, Any, Optional
from weaviate.classes.query import Filter, GroupBy
from weaviate.collections.classes.grpc import Sort, GroupBy

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
    def __init__(self, client, class_name: str):
        self.client = client
        self.class_name = class_name
        self._cached_schema_fields = None

    def _get_collection(self):
        """Get the Weaviate collection."""
        return self.client.collections.get(self.class_name)

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

    def filter_data(self, filters: Dict[str, Any]) -> Any:
        """
        Build and return the filter_obj for later use in a combined query.
        """
        filter_obj = self._create_filters(filters) if filters else None
        return filter_obj

    def group_data(self, data: Any, group_field: str, limit: Optional[int] = None, 
                  include_items: bool = False, max_items_per_group: int = 3) -> list:
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

    def sort_data(self, data: Any, sort_field: str, ascending: bool = True) -> list:
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
        from datetime import datetime
        def get_sort_value(x):
            # Try common variants
            for field in [sort_field, 'publish_date', 'publishing_date', 'date']:
                val = x.get(field)
                if val is not None:
                    # Try to parse as datetime if string
                    if isinstance(val, datetime):
                        return val
                    if isinstance(val, str):
                        try:
                            # Handle both with and without timezone
                            if 'T' in val:
                                return datetime.fromisoformat(val.replace('Z', '+00:00'))
                            else:
                                return datetime.strptime(val, '%Y-%m-%d')
                        except Exception:
                            continue
                    return val
            return datetime.min  # fallback for missing values
        return sorted(
            objects,
            key=get_sort_value,
            reverse=not ascending
        )

    def execute_combined_query(self, query: str = "", filter_obj=None, group_by_obj=None, limit: int = 100, sort_field: str = None, ascending: bool = True, query_properties: list = None, objects_per_group: int = 3, number_of_groups: int = 100, score_threshold: float = None) -> list:
        """
        Execute a single .hybrid() query with all built query objects, then apply sort in Python if needed.
        Only uses .hybrid() if a non-empty query string is provided; otherwise raises an error.
        """
        collection = self._get_collection()
        if query_properties is None:
            schema_fields = self.get_schema_fields()
            text_fields = [k for k, v in schema_fields.items() if v in ['text', 'string']]
            query_properties = [f"{text_fields[0]}^2"] + text_fields[1:] if text_fields else []
        # Use .hybrid() only if query is a non-empty string
        if not (query and isinstance(query, str) and query.strip()):
            raise ValueError("Cannot execute .hybrid() without a non-empty semantic query string. Pure filter/group/sort queries are not supported in this configuration.")
        response = collection.query.near_text(
            query=query,
            filters=filter_obj,
            group_by=group_by_obj,
            distance=0.5,
            limit=limit
        )
        results = []
        # If group_by, return grouped results
        if group_by_obj and hasattr(response, 'groups') and response.groups:
            group_results = []
            for group_name, items in response.groups.items():
                # If items is a Group object, get its .objects attribute; otherwise, use as is
                if hasattr(items, 'objects'):
                    items_list = items.objects
                else:
                    items_list = items if isinstance(items, list) else list(items)
                group = {
                    'group_name': group_name,
                    'item_count': len(items_list),
                    'items': [item.properties for item in items_list[:objects_per_group]],
                    'field_values': {},
                    'latest_date': None
                }
                group_results.append(group)
            results = group_results[:number_of_groups]
        else:
            results = [obj.properties for obj in response.objects]
            # Attach score if available
            for i, obj in enumerate(response.objects):
                if hasattr(obj, '_search_metadata') and hasattr(obj._search_metadata, 'score'):
                    results[i]['_search_metadata'] = {'score': obj._search_metadata.score}
        if sort_field:
            results = self.sort_data(results, sort_field, ascending)
        return results

    def semantic_search(self, query: str = "", **kwargs) -> str:
        """
        Return the semantic query string for use in final execution.
        Ignores extra arguments.
        """
        return query

    def objects(self, filter_obj=None, group_by_obj=None, limit: int = 100, sort_field: str = None, ascending: bool = True, objects_per_group: int = 3, number_of_groups: int = 100) -> list:
        """
        Execute a standard fetch query (non-hybrid) with optional filter, group_by, and sort.
        """
        collection = self._get_collection()
        results = []
        try:
            # Build the sorting object
            sort_obj = None
            if sort_field:
                sort_obj = Sort.by_property(
                    sort_field,
                    ascending=ascending
                )
            query = collection.query.fetch_objects(
                filters=filter_obj,
                sort=sort_obj,
                limit=limit
            )
            response = query.objects
            if response:
                results = [obj.properties for obj in response]
                if group_by_obj and results:
                    group_field = None
                    if hasattr(group_by_obj, 'path'):
                        group_field = group_by_obj.path[0] if isinstance(group_by_obj.path, list) else group_by_obj.path
                    if group_field:
                        from collections import defaultdict
                        grouped = defaultdict(list)
                        for item in results:
                            group_value = item.get(group_field)
                            grouped[group_value].append(item)
                        group_results = []
                        for group_name, items in grouped.items():
                            if group_name is not None:
                                group = {
                                    'group_name': group_name,
                                    'item_count': len(items),
                                    'items': items[:objects_per_group],
                                    'field_values': {group_field: group_name},
                                    'latest_date': None
                                }
                                group_results.append(group)
                        results = group_results[:number_of_groups]
            return results
        except Exception as e:
            logger.error(f"Error in objects query: {str(e)}")
            raise

    def _normalize_data_for_operation(self, data: Any, operation: str) -> list:
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

def create_groupby_object(group_field: str, objects_per_group: int = 10, number_of_groups: int = 10):
    """
    Utility to create a Weaviate GroupBy object from a group_field string.
    Uses all required fields: prop (string), objects_per_group, number_of_groups.
    
    """
    if not group_field or not isinstance(group_field, str) or not group_field.strip():
        return None
    try:
        obj = GroupBy(prop=group_field, objects_per_group=objects_per_group, number_of_groups=number_of_groups)
        logging.info(f"Created GroupBy object for field '{group_field}' with objects_per_group={objects_per_group}, number_of_groups={number_of_groups}")
        return obj
    except Exception as e:
        logging.warning(f"Failed to create GroupBy object for field '{group_field}': {e}")
        return None