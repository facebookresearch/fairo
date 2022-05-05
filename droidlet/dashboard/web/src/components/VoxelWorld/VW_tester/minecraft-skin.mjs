var THREE

class Skin {
    constructor (three, image, opts) {
        if (opts) opts.image = opts.image || image
        else opts = { image: image }
        if (typeof image === 'object' && !(image instanceof HTMLElement)) opts = image
        THREE = three // hack until three.js fixes multiple instantiation
        this.sizeRatio = opts.sizeRatio || 8
        this.scale = opts.scale || new three.Vector3(1, 1, 1)
        this.fallbackImage = opts.fallbackImage || 'skin.png'
        this.createCanvases()
        this.charMaterial = this.getMaterial(this.skin, false)
        this.charMaterialTrans = this.getMaterial(this.skin, true)
        if (typeof opts.image === "string") this.fetchImage(opts.image)
        if (opts.image instanceof HTMLElement) this.setImage(opts.image)
        this.mesh = this.createPlayerObject()
    }
}

Skin.prototype.createCanvases = function() {
  this.skinBig = document.createElement('canvas')
  this.skinBigContext = this.skinBig.getContext('2d')
  this.skinBig.width = 64 * this.sizeRatio
  this.skinBig.height = 32 * this.sizeRatio
  
  this.skin = document.createElement('canvas')
  this.skinContext = this.skin.getContext('2d')
  this.skin.width = 64
  this.skin.height = 32
}

Skin.prototype.fetchImage = function(imageURL) {
  var self = this
  this.image = new Image()
  this.image.crossOrigin = 'anonymous'
  this.image.src = imageURL
  this.image.onload = function() {
    self.setImage(self.image)
  }
}

Skin.prototype.setImage = function (skin) {
  this.image = skin
  this.skinContext.clearRect(0, 0, 64, 32);
  
  this.skinContext.drawImage(skin, 0, 0);
  
  var imgdata = this.skinContext.getImageData(0, 0, 64, 32);
  var pixels = imgdata.data;

  this.skinBigContext.clearRect(0, 0, this.skinBig.width, this.skinBig.height);
  this.skinBigContext.save();
  
  var isOnecolor = true;
  
  var colorCheckAgainst = [40, 0];
  var colorIndex = (colorCheckAgainst[0]+colorCheckAgainst[1]*64)*4;
  
  var isPixelDifferent = function (x, y) {
    if(pixels[(x+y*64)*4+0] !== pixels[colorIndex+0] || pixels[(x+y*64)*4+1] !== pixels[colorIndex+1] || pixels[(x+y*64)*4+2] !== pixels[colorIndex+2] || pixels[(x+y*64)*4+3] !== pixels[colorIndex+3]) {
      return true;
    }
    return false;
  };
  
  // Check if helmet/hat is a solid color
  // Bottom row
  for(var i=32; i < 64; i+=1) {
    for(var j=8; j < 16; j+=1) {
      if(isPixelDifferent(i, j)) {
        isOnecolor = false;
        break;
      }
    }
    if(!isOnecolor) {
      break;
    }
  }
  if(!isOnecolor) {
    // Top row
    for(var i=40; i < 56; i+=1) {
      for(var j=0; j < 8; j+=1) {
        if(isPixelDifferent(i, j)) {
          isOnecolor = false;
          break;
        }
      }
      if(!isOnecolor) {
        break;
      }
      
    }
  }
  
  for(var i=0; i < 64; i+=1) {
    for(var j=0; j < 32; j+=1) {
      if(isOnecolor && ((i >= 32 && i < 64 && j >= 8 && j < 16) || (i >= 40 && i < 56 && j >= 0 && j < 8))) {
        pixels[(i+j*64)*4+3] = 0
      }
      this.skinBigContext.fillStyle = 'rgba('+pixels[(i+j*64)*4+0]+', '+pixels[(i+j*64)*4+1]+', '+pixels[(i+j*64)*4+2]+', '+pixels[(i+j*64)*4+3]/255+')';
      this.skinBigContext.fillRect(i * this.sizeRatio, j * this.sizeRatio, this.sizeRatio, this.sizeRatio);
    }
  }
  
  this.skinBigContext.restore();
  
  this.skinContext.putImageData(imgdata, 0, 0);
  
  this.charMaterial.map.needsUpdate = true;
  this.charMaterialTrans.map.needsUpdate = true;
  
};

Skin.prototype.getMaterial = function(img, transparent) {
  var texture    = new THREE.Texture(img);
  texture.magFilter  = THREE.NearestFilter;
  texture.minFilter  = THREE.NearestFilter;
  texture.format    = transparent ? THREE.RGBAFormat : THREE.RGBFormat;
  texture.needsUpdate  = true;
  var material  = new THREE.MeshBasicMaterial({
    map    : texture,
    transparent  : transparent ? true : false
  });
  return material;
}

Skin.prototype.UVMap = function(mesh, face, x, y, w, h, rotateBy) {
  if (!rotateBy) rotateBy = 0;
  var uvs = mesh.geometry.faceVertexUvs[0][face];
  var tileU = x;
  var tileV = y;
  var tileUvWidth = 1/64;
  var tileUvHeight = 1/32;
  uvs[ (0 + rotateBy) % 4 ].x = (tileU * tileUvWidth)
  uvs[ (0 + rotateBy) % 4 ].y = 1 - (tileV * tileUvHeight)
  uvs[ (1 + rotateBy) % 4 ].x = (tileU * tileUvWidth)
  uvs[ (1 + rotateBy) % 4 ].y = 1 - (tileV * tileUvHeight + h * tileUvHeight)
  uvs[ (2 + rotateBy) % 4 ].x = (tileU * tileUvWidth + w * tileUvWidth)
  uvs[ (2 + rotateBy) % 4 ].y = 1 - (tileV * tileUvHeight + h * tileUvHeight)
  uvs[ (3 + rotateBy) % 4 ].x = (tileU * tileUvWidth + w * tileUvWidth)
  uvs[ (3 + rotateBy) % 4 ].y = 1 - (tileV * tileUvHeight)
}

Skin.prototype.cubeFromPlanes = function (size, mat) {
  var cube = new THREE.Object3D();
  var meshes = [];
  for(var i=0; i < 6; i++) {
    var mesh = new THREE.Mesh(new THREE.PlaneGeometry(size, size), mat);
    mesh.doubleSided = true;
    cube.add(mesh);
    meshes.push(mesh);
  }
  // Front
  meshes[0].rotation.x = Math.PI/2;
  meshes[0].rotation.z = -Math.PI/2;
  meshes[0].position.x = size/2;
  
  // Back
  meshes[1].rotation.x = Math.PI/2;
  meshes[1].rotation.z = Math.PI/2;
  meshes[1].position.x = -size/2;
  
  // Top
  meshes[2].position.y = size/2;
  
  // Bottom
  meshes[3].rotation.y = Math.PI;
  meshes[3].rotation.z = Math.PI;
  meshes[3].position.y = -size/2;
  
  // Left
  meshes[4].rotation.x = Math.PI/2;
  meshes[4].position.z = size/2;
  
  // Right
  meshes[5].rotation.x = -Math.PI/2;
  meshes[5].rotation.y = Math.PI;
  meshes[5].position.z = -size/2;
  
  return cube;
}

//exporting these meshes for manipulation:
//leftLeg
//rightLeg
//leftArm
//rightArm
//body
//head

Skin.prototype.createPlayerObject = function() {
  var headgroup = new THREE.Object3D();
  var upperbody = this.upperbody = new THREE.Object3D();
  
  // Left leg
  var leftleggeo = new THREE.BoxGeometry(4, 12, 4);
//   for(var i=0; i < 8; i+=1) {
//     leftleggeo.vertices[i].y -= 6;
//   }
  var leftleg = this.leftLeg = new THREE.Mesh(leftleggeo, this.charMaterial);
  leftleg.position.z = -2;
  leftleg.position.y = -6;
//   this.UVMap(leftleg, 0, 8, 20, -4, 12);
//   this.UVMap(leftleg, 1, 16, 20, -4, 12);
//   this.UVMap(leftleg, 2, 4, 16, 4, 4, 3);
//   this.UVMap(leftleg, 3, 8, 20, 4, -4, 1);
//   this.UVMap(leftleg, 4, 12, 20, -4, 12);
//   this.UVMap(leftleg, 5, 4, 20, -4, 12);

  // Right leg
  var rightleggeo = new THREE.BoxGeometry(4, 12, 4);
//   for(var i=0; i < 8; i+=1) {
//     rightleggeo.vertices[i].y -= 6;
//   }
  var rightleg = this.rightLeg =new THREE.Mesh(rightleggeo, this.charMaterial);
  rightleg.position.z = 2;
  rightleg.position.y = -6;
//   this.UVMap(rightleg, 0, 4, 20, 4, 12);
//   this.UVMap(rightleg, 1, 12, 20, 4, 12);
//   this.UVMap(rightleg, 2, 8, 16, -4, 4, 3);
//   this.UVMap(rightleg, 3, 12, 20, -4, -4, 1);
//   this.UVMap(rightleg, 4, 0, 20, 4, 12);
//   this.UVMap(rightleg, 5, 8, 20, 4, 12);
  
  // Body
  var bodygeo = new THREE.BoxGeometry(4, 12, 8);
  var bodymesh = this.body = new THREE.Mesh(bodygeo, this.charMaterial);
//   this.UVMap(bodymesh, 0, 20, 20, 8, 12);
//   this.UVMap(bodymesh, 1, 32, 20, 8, 12);
//   this.UVMap(bodymesh, 2, 20, 16, 8, 4, 1);
//   this.UVMap(bodymesh, 3, 28, 16, 8, 4, 3);
//   this.UVMap(bodymesh, 4, 16, 20, 4, 12);
//   this.UVMap(bodymesh, 5, 28, 20, 4, 12);
  upperbody.add(bodymesh);
  
  
  // Left arm
  var leftarmgeo = new THREE.BoxGeometry(4, 12, 4);
//   for(var i=0; i < 8; i+=1) {
//     leftarmgeo.vertices[i].y -= 4;
//   }
  var leftarm = this.leftArm = new THREE.Mesh(leftarmgeo, this.charMaterial);
  leftarm.position.z = -6;
  leftarm.position.y = 4;
  leftarm.rotation.x = Math.PI/32;
//   this.UVMap(leftarm, 0, 48, 20, -4, 12);
//   this.UVMap(leftarm, 1, 56, 20, -4, 12);
//   this.UVMap(leftarm, 2, 48, 16, -4, 4, 1);
//   this.UVMap(leftarm, 3, 52, 16, -4, 4, 3);
//   this.UVMap(leftarm, 4, 52, 20, -4, 12);
//   this.UVMap(leftarm, 5, 44, 20, -4, 12);
  upperbody.add(leftarm);
  
  // Right arm
  var rightarmgeo = new THREE.BoxGeometry(4, 12, 4);
//   for(var i=0; i < 8; i+=1) {
//     rightarmgeo.vertices[i].y -= 4;
//   }
  var rightarm =this.rightArm = new THREE.Mesh(rightarmgeo, this.charMaterial);
  rightarm.position.z = 6;
  rightarm.position.y = 4;
  rightarm.rotation.x = -Math.PI/32;
//   this.UVMap(rightarm, 0, 44, 20, 4, 12);
//   this.UVMap(rightarm, 1, 52, 20, 4, 12);
//   this.UVMap(rightarm, 2, 44, 16, 4, 4, 1);
//   this.UVMap(rightarm, 3, 48, 16, 4, 4, 3);
//   this.UVMap(rightarm, 4, 40, 20, 4, 12);
//   this.UVMap(rightarm, 5, 48, 20, 4, 12);
  upperbody.add(rightarm);
  
  //Head
  var headgeo = new THREE.BoxGeometry(8, 8, 8);
  var headmesh = this.head = new THREE.Mesh(headgeo, this.charMaterial);
  headmesh.position.y = 2;
//   this.UVMap(headmesh, 0, 8, 8, 8, 8);
//   this.UVMap(headmesh, 1, 24, 8, 8, 8);
  
//   this.UVMap(headmesh, 2, 8, 0, 8, 8, 1);
//   this.UVMap(headmesh, 3, 16, 0, 8, 8, 3);
  
//   this.UVMap(headmesh, 4, 0, 8, 8, 8);
//   this.UVMap(headmesh, 5, 16, 8, 8, 8);

  var unrotatedHeadMesh = new THREE.Object3D();
  unrotatedHeadMesh.rotation.y = Math.PI / 2;
  unrotatedHeadMesh.add(headmesh);

  headgroup.add(unrotatedHeadMesh);

  var helmet = this.cubeFromPlanes(9, this.charMaterialTrans);
  helmet.position.y = 2;
//   this.UVMap(helmet.children[0], 0, 32+8, 8, 8, 8);
//   this.UVMap(helmet.children[1], 0, 32+24, 8, 8, 8);
//   this.UVMap(helmet.children[2], 0, 32+8, 0, 8, 8, 1);
//   this.UVMap(helmet.children[3], 0, 32+16, 0, 8, 8, 3);
//   this.UVMap(helmet.children[4], 0, 32+0, 8, 8, 8);
//   this.UVMap(helmet.children[5], 0, 32+16, 8, 8, 8);
  
  headgroup.add(helmet);
  
  var ears = new THREE.Object3D();
  
  var eargeo = new THREE.BoxGeometry(1, (9/8)*6, (9/8)*6);
  var leftear = new THREE.Mesh(eargeo, this.charMaterial);
  var rightear = new THREE.Mesh(eargeo, this.charMaterial);
  
  leftear.position.y = 2+(9/8)*5;
  rightear.position.y = 2+(9/8)*5;
  leftear.position.z = -(9/8)*5;
  rightear.position.z = (9/8)*5;
  
  // Right ear share same geometry, same uv-maps
  
//   this.UVMap(leftear, 0, 25, 1, 6, 6); // Front side
//   this.UVMap(leftear, 1, 32, 1, 6, 6); // Back side
  
//   this.UVMap(leftear, 2, 25, 0, 6, 1, 1); // Top edge
//   this.UVMap(leftear, 3, 31, 0, 6, 1, 1); // Bottom edge
  
//   this.UVMap(leftear, 4, 24, 1, 1, 6); // Left edge
//   this.UVMap(leftear, 5, 31, 1, 1, 6); // Right edge
  
  ears.add(leftear);
  ears.add(rightear);
  
  leftear.visible = rightear.visible = false;
  
  headgroup.add(ears);
  headgroup.position.y = 8;
  
  var playerModel = this.playerModel = new THREE.Object3D();
  
  playerModel.add(leftleg);
  playerModel.add(rightleg);
  
  playerModel.add(upperbody);
  
  var playerRotation = new THREE.Object3D();
  playerRotation.rotation.y = Math.PI / 2
  playerRotation.position.y = 12
  playerRotation.add(playerModel)

  var rotatedHead = new THREE.Object3D();
  rotatedHead.rotation.y = -Math.PI/2;
  rotatedHead.add(headgroup);

  playerModel.add(rotatedHead);
  playerModel.position.y = 6;
  
  var playerGroup = new THREE.Object3D();
  playerGroup.cameraInside = new THREE.Object3D()
  playerGroup.cameraOutside = new THREE.Object3D()

  playerGroup.cameraInside.position.x = 0;
  playerGroup.cameraInside.position.y = 2;
  playerGroup.cameraInside.position.z = 0; 

  playerGroup.head = headgroup
  headgroup.add(playerGroup.cameraInside)
  playerGroup.cameraInside.add(playerGroup.cameraOutside)

  playerGroup.cameraOutside.position.z = 100

  
  playerGroup.add(playerRotation);
  playerGroup.scale.set(this.scale);
  return playerGroup
}

export {Skin};