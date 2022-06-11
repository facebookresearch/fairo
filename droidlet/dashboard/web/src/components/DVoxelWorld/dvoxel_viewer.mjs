// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

import { DVoxelEngine, SL } from './dvoxel_engine.mjs';

let max = 5, min = -5
let dVoxelEngine = new DVoxelEngine({});
console.log(document.getElementById("dvoxel_viewer"))
// dVoxelEngine.appendTo(document.getElementById("dvoxel_viewer"));
dVoxelEngine.appendTo(document.body);
window.addEventListener("keydown", function (ev) {
    if (ev.keyCode === "2".charCodeAt(0)) {
        let hitBlock = dVoxelEngine.raycastVoxels(dVoxelEngine);
        let hitX = Math.floor(hitBlock[0])
        let hitY = Math.floor(hitBlock[1]) + 1
        let hitZ = Math.floor(hitBlock[2])
        if (dVoxelEngine.getBlock(hitX, hitY, hitZ) == 0) {
          dVoxelEngine.setVoxel([hitX, hitY, hitZ], 46);
        }
    }
  });

  window.addEventListener("keydown", function (ev) {
    if (ev.keyCode === "3".charCodeAt(0)) {
        let hitBlock = dVoxelEngine.raycastVoxels(dVoxelEngine);
        let hitX = Math.floor(hitBlock[0])
        let hitY = Math.floor(hitBlock[1]) + 1
        let hitZ = Math.floor(hitBlock[2])
        if (dVoxelEngine.getBlock(hitX, hitY, hitZ) == 0) {
          dVoxelEngine.setVoxel([hitX, hitY, hitZ], 47);
        }
    }
  });

  window.addEventListener("keydown", function (ev) {
    if (ev.keyCode === "4".charCodeAt(0)) {
        let hitBlock = dVoxelEngine.raycastVoxels(dVoxelEngine);
        let hitX = Math.floor(hitBlock[0])
        let hitY = Math.floor(hitBlock[1]) + 1
        let hitZ = Math.floor(hitBlock[2])
        if (dVoxelEngine.getBlock(hitX, hitY, hitZ) == 0) {
          dVoxelEngine.setVoxel([hitX, hitY, hitZ], 48);
        }
    }
  });

  window.addEventListener("keydown", function (ev) {
    if (ev.keyCode === "5".charCodeAt(0)) {
        let hitBlock = dVoxelEngine.raycastVoxels(dVoxelEngine);
        let hitX = Math.floor(hitBlock[0])
        let hitY = Math.floor(hitBlock[1]) + 1
        let hitZ = Math.floor(hitBlock[2])
        if (dVoxelEngine.getBlock(hitX, hitY, hitZ) == 0) {
          dVoxelEngine.setVoxel([hitX, hitY, hitZ], 49);
        }
    }
  });

  window.addEventListener("keydown", function (ev) {
    if (ev.keyCode === "6".charCodeAt(0)) {
        let hitBlock = dVoxelEngine.raycastVoxels(dVoxelEngine);
        let hitX = Math.floor(hitBlock[0])
        let hitY = Math.floor(hitBlock[1]) + 1
        let hitZ = Math.floor(hitBlock[2])
        if (dVoxelEngine.getBlock(hitX, hitY, hitZ) == 0) {
          dVoxelEngine.setVoxel([hitX, hitY, hitZ], 50);
        }
    }
  });
  window.addEventListener("keydown", function (ev) {
    if (ev.key === "g") {
      let hitBlock = dVoxelEngine.raycastVoxels(dVoxelEngine);
      let hitX = Math.floor(hitBlock[0])
      let hitY = Math.floor(hitBlock[1])
      let hitZ = Math.floor(hitBlock[2])
      console.log("hitting: ")
      console.log(hitX + ", " + hitY + ', ' + hitZ)
      if (dVoxelEngine.getBlock(hitX, hitY, hitZ) != 0) {
        dVoxelEngine.setVoxel([hitX, hitY, hitZ], 0);
      }
    }
  });

// for (let ix = -SL/2; ix < SL/2; ix++) {
//     for (let iz = -SL/2; iz < SL/2; iz++) {
//         for (let iy = 2; iy < 4; iy++) {
//             dVoxelEngine.setVoxel([ix,iy,iz], 9);
//         }
//     }
// }
// for (let ix = -SL/2; ix < SL/2; ix++) {
//     for (let iz = -SL/2; iz < SL/2; iz++) {
//         dVoxelEngine.setVoxel([ix,4,iz], 8);
//     }
// }

dVoxelEngine.render();

function updateAgents(agentsInfo) {
  dVoxelEngine.updateAgents(agentsInfo);
}

function updateMobs(mobsInfo) {
  dVoxelEngine.updateMobs(mobsInfo);
}

function updateItemStacks(itemStacksInfo) {
  dVoxelEngine.updateItemStacks(itemStacksInfo);
}

function updateBlocks(blocksInfo) {
  dVoxelEngine.updateBlocks(blocksInfo);
}

function setBlock(x, y, z, idm) {
  dVoxelEngine.setBlock(x, y, z, idm);
}

function flashBlocks(bbox) {
  dVoxelEngine.flashBlocks(bbox);
}

module.exports.updateAgents = updateAgents;
module.exports.updateMobs = updateMobs;
module.exports.updateItemStacks = updateItemStacks;
module.exports.updateBlocks = updateBlocks;
module.exports.setBlock = setBlock;
module.exports.flashBlocks = flashBlocks;