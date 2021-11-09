/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
var createGame = require("voxel-engine");
var highlight = require("voxel-highlight");
var player = require("voxel-player");
var voxel = require("voxel");
var extend = require("extend");
var fly = require("voxel-fly");
var walk = require("voxel-walk");
var texturePath = require("programmerart-textures")("");

var BLOCK_MAP = require("./blockmap.js").BLOCK_MAP;
var MATERIAL_MAP = require("./blockmap.js").MATERIAL_MAP;

TEXTURE_PATH = "./textures/";
SPEAKER_SKIN_PATH = TEXTURE_PATH + "speaker.png";
AGENT_SKIN_PATH = TEXTURE_PATH + "agent.png";
DEFAULT_BLOCK_ID = 66;

function enableFly(game, target) {
  var makeFly = fly(game);
  game.flyer = makeFly(target);
}

function defaultSetup(game, avatar) {
  var makeFly = fly(game);
  var target = game.controls.target();
  game.flyer = makeFly(target);

  // highlight blocks when you look at them, hold <Ctrl> for block placement
  var blockPosPlace, blockPosErase;
  var hl = (game.highlighter = highlight(game, { color: 0xff0000 }));
  hl.on("highlight", function (voxelPos) {
    blockPosErase = voxelPos;
  });
  hl.on("remove", function (voxelPos) {
    blockPosErase = null;
  });
  hl.on("highlight-adjacent", function (voxelPos) {
    blockPosPlace = voxelPos;
  });
  hl.on("remove-adjacent", function (voxelPos) {
    blockPosPlace = null;
  });

  // toggle between first and third person modes
  window.addEventListener("keydown", function (ev) {
    if (ev.keyCode === "R".charCodeAt(0)) avatar.toggle();
  });

  window.addEventListener("keydown", function (ev) {
    if (ev.keyCode === "1".charCodeAt(0)) {
      game.createAdjacent(game.raycastVoxels(), 57);
      console.log(window.parent);
      window.parent.postMessage("TESTINGGG", "*");
    }
  });

  // block interaction stuff, uses highlight data
  var currentMaterial = 1;

  game.on("fire", function (target, state) {
    var position = blockPosPlace;
    if (position) {
      game.createBlock(position, currentMaterial);
    } else {
      position = blockPosErase;
      if (position) game.setBlock(position, 0);
    }
  });

  game.on("tick", function () {
    walk.render(target.playerSkin);
    var vx = Math.abs(target.velocity.x);
    var vz = Math.abs(target.velocity.z);
    if (vx > 0.001 || vz > 0.001) walk.stopWalking();
    else walk.startWalking();
  });
}

function World() {
  var vals = (function (opts, setup) {
    setup = setup || defaultSetup;
    var defaults = {
      generate: function (x, y, z) {
        return y < 63 ? 1 : 0; // flat world
      },
      chunkDistance: 2,
      materials: MATERIAL_MAP,
      worldOrigin: [0, 63, 0],
      controls: { discreteFire: true },
      texturePath: TEXTURE_PATH,
    };
    opts = extend({}, defaults, opts || {});

    // setup the game and add some trees
    var game = createGame(opts);
    var container = opts.container || document.body;
    window.game = game; // for debugging
    game.appendTo(container);
    if (game.notCapable()) return game;

    var createPlayer = player(game);

    // create the player from a minecraft skin file and tell the
    // game to use it as the main player
    var avatar = createPlayer(opts.playerSkin || SPEAKER_SKIN_PATH);
    avatar.possess();
    avatar.yaw.position.set(2, 14, 4);
    avatar.position.set(0, 63, 0);
    avatar.toggle();
    setup(game, avatar);
    return [avatar, game];
  })();
  this.speaker = vals[0];
  this.game = vals[1];

  // create agent
  var createPlayer = player(vals[1]);
  this.agent = createPlayer(AGENT_SKIN_PATH);
  this.agent.yaw.position.set(2, 14, 4);
  this.agent.position.set(5, 63, 5);
  enableFly(this.game, this.agent);

  this.avatars = {};
  this.avatars["player"] = this.speaker;
  this.avatars["agent"] = this.agent;
  this.game.control(this.speaker);

  /* private */
  this.getGame = function () {
    return this.game;
  };

  this.getAvatar = function (name) {
    return this.avatars[name];
  };

  this.setAgentLocation = function (name, x, y, z) {
    this.avatars[name].position.set(x, y, z);
  };

  /* public */
  this.updateAgents = function (agentsInfo) {
    for (var i = 0; i < agentsInfo.length; i++) {
      const name = agentsInfo[i]["name"];
      const x = agentsInfo[i]["x"];
      const y = agentsInfo[i]["y"];
      const z = agentsInfo[i]["z"];
      this.setAgentLocation(name, x, y, z);
    }
  };

  this.updateBlocks = function (blocksInfo) {
    for (var i = 0; i < blocksInfo.length; i++) {
      let [xyz, idm] = blocksInfo[i];
      var id = BLOCK_MAP[idm.toString()];
      if (id == undefined) {
        id = DEFAULT_BLOCK_ID;
      }
      if (id == 0) {
        this.game.setBlock(xyz, id);
      } else {
        this.game.createBlock(xyz, id);
      }
    }
  };

  this.setBlock = function (x, y, z, idm) {
    this.game.setBlock([x, y, z], idm);
  };
}

module.exports = World;
