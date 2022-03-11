# Documentation

Modify code in [source](./source) to change documentation. They will be published at [Github Pages](https://facebookresearch.github.io/fairo/polymetis/) when merged to `main`.

You can start a webserver in [_build/html](_build/html) to see local changes: `python -m http.server`

# Auto-generated documentation

## Generating Python documentation

Simply `make html` in this directory.

### Adding a new module

Use [sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) to automatically generate a new `.rst` file:

```bash
cd docs/
sphinx-apidoc -o . ../path/to/new/python/module
```

## Generating C++ documentation

While [building locally](../polymetis/README.md), simply add the `BUILD_DOCS` flag to your `cmake` command:

```bash
mdkir -p ./polymetis/build/
cd ./polymetis/build/

cmake .. -DBUILD_DOCS=ON
cmake --build .
```
