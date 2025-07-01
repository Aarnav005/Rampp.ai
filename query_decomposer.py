import logging
import google.generativeai as genai
import json
from typing import Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

**CRITICAL: For queries asking for all unique/distinct values of a field (e.g., 'all unique authors', 'distinct case names', 'list all X values'), use group_data to group by that field and return the group names. Do NOT use get_distinct_values or extract_field_values for these queries.**

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

Example Query 7: "Give me all unique authors"
Example JSON Output:
```json
{{
  "steps": [
    {{ "step_id": 1, "tool": "filter_data", "parameters": {{"data_source": null, "filters": {{}}, "source_type": "database"}}, "description": "Get all articles" }},
    {{ "step_id": 2, "tool": "group_data", "parameters": {{"data": "$PREV_STEP_RESULT", "group_field": "author"}}, "description": "Group by author to get all unique authors" }}
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