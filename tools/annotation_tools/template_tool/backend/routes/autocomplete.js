/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
var express = require("express");
var router = express.Router();

/* GET autocomplete page. */
router.get("/", function (req, res, next) {
  res.render("autocomplete", { title: "Autocompletion Tool" });
});

module.exports = router;