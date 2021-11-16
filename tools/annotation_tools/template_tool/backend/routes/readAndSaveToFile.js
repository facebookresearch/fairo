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
const execSync = require('child_process').execSync;


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

/***
 * Fetch the commands we want to label
 */
router.get("/get_commands", function (req, res, next) {
  if (fs.existsSync("commands.txt")) {
    // the file exists
    fs.readFile("commands.txt", function (err, data) {
      if (err) throw err;
      // console.log(data.toString())
      res.writeHead(200, { "Content-Type": "text/html" });
      res.write(data.toString());
      return res.end();
    });
  }
});

/***
 * Fetch the commands we want to label
 */
router.get("/get_fragments", function (req, res, next) {
  if (fs.existsSync("fragments.txt")) {
    // the file exists
    fs.readFile("fragments.txt", function (err, data) {
      if (err) throw err;
      res.writeHead(200, { "Content-Type": "text/html" });
      res.write(data.toString());
      return res.end();
    });
  }
});

/***
 * Fetch progress on labels
 */
router.get("/get_labels_progress", function (req, res, next) {
  if (fs.existsSync("../frontend/src/command_dict_pairs.json")) {
    // the file exists
    fs.readFile("../frontend/src/command_dict_pairs.json", function (err, data) {
      if (err) throw err;
      res.writeHead(200, { "Content-Type": "application/json" });
      res.write(data);
      return res.end();
    });
  }
});

/***
 * Fetch previously labelled templates
 */
router.get("/get_templates", function (req, res, next) {
  if (fs.existsSync("templates_autocomplete.json")) {
    // the file exists
    fs.readFile("templates_autocomplete.json", function (err, data) {
      if (err) throw err;
      res.writeHead(200, { "Content-Type": "application/json" });
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

/**
 * Write labelled pairs
 */
router.post("/writeLabels", function (req, res, next) {
  console.log(req.body);
  fs.writeFile("../frontend/src/command_dict_pairs.json", JSON.stringify(req.body, undefined, 4), function (err) {
    // err is an error other than fileNotExists
    // if file does not exist, writeFile will create it
    if (err) throw err;
    console.log("Saved template information to file!");
  });
  res.send("post is working properly");
});

/**
 * Write Templates
 */
router.post("/writeTemplates", function (req, res, next) {
  console.log(req.body);
  fs.writeFile("templates_autocomplete.json", JSON.stringify(req.body, undefined, 4), function (err) {
    // err is an error other than fileNotExists
    // if file does not exist, writeFile will create it
    if (err) throw err;
    console.log("Saved template information to file!");
  });
  res.send("post is working properly");
});

/**
 * Write labelled pairs
 */
router.post("/uploadDataToS3", function (req, res, next) {
  try {
    console.log(req.body);
    const execSync = require('child_process').execSync;
    const postprocessing_output = execSync('python ../../../data_processing/autocomplete_postprocess.py --source_path ../frontend/src/command_dict_pairs.json');
    console.log('Postprocessing Output was:\n', postprocessing_output);
  }
  catch (error) {
    return res.status(500).json({ error: error.toString() });
  }

  res.send("Saved processed dataset to ~/droidlet/artifacts/datasets/full_data/");
});

module.exports = router;
