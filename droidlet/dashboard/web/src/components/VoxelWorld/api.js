/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
const World = require("./world");
const world = new World();

function updateAgents(agentsInfo) {
  world.updateAgents(agentsInfo);
}

function updateBlocks(blocksInfo) {
  world.updateBlocks(blocksInfo);
}

function setBlock(x, y, z, idm) {
  world.setBlock(x, y, z, idm);
}

module.exports.updateAgents = updateAgents;
module.exports.updateBlocks = updateBlocks;
module.exports.setBlock = setBlock;
