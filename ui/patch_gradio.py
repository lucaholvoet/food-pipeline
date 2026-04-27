import pathlib

p = pathlib.Path('/usr/local/lib/python3.11/site-packages/gradio_client/utils.py')
src = p.read_text()

# gradio_client 1.3.0 bug: _json_schema_to_python_type() is called recursively
# with schema['additionalProperties'] which Pydantic emits as a boolean (True/False).
# Inserting an isinstance guard at the top of the function makes it return "Any"
# for any non-dict schema value, short-circuiting all further processing.
old = '    type_ = get_type(schema)'
new = '    if not isinstance(schema, dict): return "Any"\n    type_ = get_type(schema)'

if old not in src:
    print("patch_gradio: target line not found — already patched or version mismatch")
else:
    p.write_text(src.replace(old, new, 1))
    print("patch_gradio: gradio_client/utils.py patched successfully")
