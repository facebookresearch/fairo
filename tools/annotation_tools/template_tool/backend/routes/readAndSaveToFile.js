/**
 *
 * @license
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 */

/**
 * @fileoverview This file contains the definition of a router for the template tool. This allows us to read and write from the local templateToolInfo.txt file.
 */

var express = require("express");
var fs = require("fs");
var router = express.Router();

router.get("/", function (req, res, next) {
  if (fs.existsSync("templates.txt")) {
    // the file exists
    fs.readFile("templates.txt", function (err, data) {
      // err is an error other than fileNotExists
      // we already checked for existence
      if (err) throw err;
      res.writeHead(200, { "Content-Type": "text/html" });
      res.write(data);
      return res.end();
    });
  }
});

router.post("/", function (req, res, next) {
  console.log(req.body);

  fs.writeFile("templates.txt", JSON.stringify(req.body), function (err) {
    // err is an error other than fileNotExists
    // if file does not exist, writeFile will create it
    if (err) throw err;
    console.log("Saved template information to file!");
  });
  res.send("post is working properly");
});

module.exports = router;
