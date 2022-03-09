# Build and view documentation

```
pip install -r requirements.txt
make html
pushd build/html
python -m http.server
# go to localhost:8000
```
