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
 * Fetch progress on labels
 */
router.get("/get_labels_progress", function (req, res, next) {
  if (fs.existsSync("command_dict_pairs.json")) {
    // the file exists
    fs.readFile("command_dict_pairs.json", function (err, data) {
      if (err) throw err;
      console.log(data.toString())
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

router.post("/append", function (req, res, next) {
  console.log(req.body);
  fs.appendFile("command_dict_pairs.txt", "hi" + JSON.stringify(req.body) + "\n", function (err) {
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
  fs.writeFile("command_dict_pairs.json", JSON.stringify(req.body, undefined, 4), function (err) {
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
    const postprocessing_output = execSync('python ../../../data_processing/autocomplete_postprocess.py');
    console.log('Postprocessing Output was:\n', postprocessing_output);
    const s3_output = execSync('./../../../data_scripts/tar_and_hash_datasets.sh');
    console.log('S3 Output was:\n', postprocessing_output);
  }
  catch (error) {
    return res.status(500).json({ error: error.toString() });
  }

  res.send("Uploaded data to S3!");
});

module.exports = router;
