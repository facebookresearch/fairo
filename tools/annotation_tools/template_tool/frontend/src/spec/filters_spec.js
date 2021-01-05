const autocompleteMatches = [
    {match: /"GET_MEMORY":  /gi, replacement: `{"dialogue_type": "GET_MEMORY",\n"filters":"",\n"replace":""\n}`},
    {match: /"filters":  /gi, replacement: `"filters": { \n"triples":"", \n"output":"", \n"contains_coreference":"", \n"memory_type":"", \n"argval":"", \n"comparator":"", \n"author":"", \n"location":"" } `},
    {match: /"triples":  /gi, replacement: `"triples":  \n[{"pred_text":"", "obj_text":"", "subj_text":""}]`},
    {match: /"location":  /gi, replacement: `"location": { \n"text_span":"",\n"steps":"",\n"has_measure":"",\n"contains_coreference":"",\n"relative_direction":"",\n"reference_object":""\n}`},
    {match: /"reference_object":  /gi, replacement: `"reference_object": { \n"text_span":"",\n"repeat":"",\n"special_reference":"",\n"filters":""\n}`},
    {match: /"comparator":  /gi, replacement: `"comparator": [{\n"input_left":"",\n"comparison_type":"",\n"input_right":"",\n"comparison_measure":"",\n"set_comparison":""\n}]`},
    {match: /"argval":  /gi, replacement: `"argval": {\n"polarity":"", \n"ordinal":"",\n"quantity":""\n}`},
    {match: /"linear_extent":  /gi, replacement: `"linear_extent": {\n"relative_direction":"",\n"frame":"",\n"has_measure":"",\n"source":"",\n"destination":""\n}`},
    {match: /"output":  /gi, replacement: `"output": {\n"attribute":""\n}`},
    {match: /"num_blocks":  /gi, replacement: `"num_blocks": {\n"block_filters": {\n"triples":""\n}\n}`},
    {match: /"task_info":  /gi, replacement: `"task_info": {\n"reference_object" : {"attribute":""\n}}`},
    {match: /"attribute":  /gi, replacement: `"attribute": {\n"num_blocks":"", \n"linear_extent":"", \n"task_info":""}`},
    {match: /"frame":  /gi, replacement: `"frame": {\n"player_span":""\n}`},
    {match: /"source":  /gi, replacement: `"source": {\n"reference_object":""\n}`},
    {match: /"destination":  /gi, replacement: `"destination": {\n"reference_object":""\n}`},
    {match: /"value_extractor":  /gi, replacement: `"value_extractor": {\n"filters":"", \n"attribute":"", \n"span":""\n}`},
    {match: /"quantity":  /gi, replacement: `"quantity": {\n"attribute":""\n}`},
    {match: /"comparison_type":  /gi, replacement: `"comparison_type": {\n"close_tolerance":"", \n"modulus":""\n}`},
    {match: /"obj_text":  /gi, replacement: `"obj_text": {\n"filters":""\n}`},
    {match: /"subj_text":  /gi, replacement: `"subj_text": {\n"filters":""\n}`},
    {match: /"input_left":  /gi, replacement: `"input_left": {"value_extractor":""\n}`},
    {match: /"input_right":  /gi, replacement: `"input_right": {"value_extractor":""\n}`},
    {match: /("pred_text":(.|\n)*},)  /gi, replacement: `$1{"pred_text":"", "obj_text":"", "subj_text":""}`}
  ];

  export default autocompleteMatches;