# Repo documentation

We use GitHub Pages to publish our documentation for all projects in this repository. The entrypoint for the overall Fairo repository is the [`index.html`](./index.html) in this folder.

## Add project to documentation page

To add your project's documentation to the repo:

1. In your CircleCI job, generate your documentation in the required environment (e.g. using [Sphinx](https://www.sphinx-doc.org/en/master/)). Each project should be responsible for generating its own documentation, including any required environment setup.
2. Use [CircleCI Workspace](https://circleci.com/docs/2.0/concepts/#workspaces) to persist the generated documentation under `workspace/docs/your-project-name`. It will automatically be pushed to GitHub Pages when merged to `main` branch at [facebookresearch.github.io/fairo](https://facebookresearch.github.io/fairo).
3. When ready, add a link/logo to `your-project-name` in [`index.html`](./index.html).
