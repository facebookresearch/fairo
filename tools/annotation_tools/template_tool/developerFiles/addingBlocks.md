# Adding blocks

To add blocks to the toolbar, refer to

1. Use [block factory](https://blockly-demo.appspot.com/static/demos/blockfactory/index.html) to create new blocks.
2. Copy the block definition and generator stub (both in `Javascript`).
3. Create a new `.js` file in `src/block` and paste the copied code in it.

4. To retain the custom save and tag options that have been created, make the following changes to the file:
5. Add this import to the top of your file ` import customInit from "./customInit"`
6. In the `init: function () {}` of the block, add `customInit(this)`.

7. In `app.js` write `import "./block/ + fileName`.

8. In the `render` in `app.js` there is a `Blockly component`. In that, add `<Block type="block name" />`.
