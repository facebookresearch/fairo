This project contains a suite of tools that are used to create annotated and generated labelled datasets.

For more information on the Parse Tree Annotation Tool (Autocomplete Tool), see `AUTOCOMPLETE_README.md`.

# Blockly Template Tool

The goal of this tool is to enable users to efficiently create templates for the CraftAssist bot. A [Blockly](https://developers.google.com/blockly) interface is used. This project is in active development.

# Installation

Open a terminal in the `template_tool` folder and do

```
cd frontend
npm install
npm start
```

This will spawn a `localhost` window.

Open another terminal in the `template_tool` folder and do

```
cd backend
npm install
npm start
```

If either installation fails, do ` npm rebuild`.

The backend is serving the app on `localhost:9000` and the frontend on `localhost:3000` (by default). The frontend may be run on another port, but the backend must remain to be at `localhost:9000`.

# Usage

For instructions on using the tool, refer to [usage doc](./usage.md)

For developer instructions, refer to [developer doc](./developers.md)

## License

CraftAssist is [MIT licensed](./LICENSE).

<sup>1</sup> Minecraft features: © Mojang Synergies AB included courtesy of Mojang AB
