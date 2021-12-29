import * as THREE from 'https://cdn.skypack.dev/three';
import { OrbitControls } from './OrbitControls.mjs';
import { GLTFLoader } from './GLTFLoader.mjs';

const BLOCK_MAP = {
46: 0xffffff, // White Wool
47: 0xffa500, // Orange Wool
48: 0xff00ff, // Magenta Wool
49: 0x75bdff, // Light Blue Wool
50: 0xffff00, // Yellow Wool
51: 0x00ff00, // Lime Wool
52: 0xffc0cb, // Pink Wool
53: 0x5b5b5b, // Gray Wool
54: 0xbcbcbc, // Light Gray Wool
55: 0x00ffff, // Cyan Wool
56: 0x800080, // Purple Wool
57: 0x2986cc, // Blue Wool
58: 0xa52a2a, // Brown Wool
59: 0x8fce00, // Green Wool
60: 0xff0000, // Red Wool
61: 0x000000, // Black Wool
};

let cameras = {1: [], 2: []};
let controls = {1: [], 2: []};
let scenes = {1: [], 2: []};
let renderers = {1: [], 2: []};
let planes = {1: [], 2: []};
let pointers = {1: [], 2: []};
let raycasters = {1: [], 2: []};
let user_raycasters = {1: [], 2: []};
let objects = {1: [], 2: []};

let marked_blocks = [];
let actions_taken = []; // [original_block, new_block, action_type]

let isADown = false;
let isSDown = false;
let isDDown = false;
var startedHIT = false;

const geo = new THREE.BoxGeometry( 50, 50, 50 );
const rollOverMaterial = new THREE.MeshBasicMaterial( { color: 0xff0000, opacity: 0.3, transparent: true } );
const cubeMaterial_mark = new THREE.MeshBasicMaterial( { color: 0x089000, opacity: 0.6, transparent: true } );
let rollOverMesh = new THREE.Mesh( geo, rollOverMaterial );

// Agent will always be looking at user, transform relative location into yaw
//let user_agent_angle = Math.atan((user[0][0] - agent[0][0]) / (user[0][2] - agent[0][2]));

let starting_cube = [];
let block_id = 0;
for (let x=-2; x<2; x++){
    for (let y=0; y<4; y++){
        for (let z=-2; z<2; z++){
            starting_cube.push([x,y,z,block_id++])
        }
    }
}

// TODO download data from S3 based on data.csv?
// In the meantime, here's some dummy data

fetch('./scene1_v2.json')
    .then(response => {
        return response.json();
    })
    .then(jsondata => {
        return jsondata[0];  // Pull the first scene (of 1 in this case)
    })
    .then(scene => {
        init(scene, 1);
        init(scene, 2);
        addEventListeners();
        render();
        var canvii = document.getElementsByTagName("canvas");
        Array.from(canvii).forEach(canv => canv.style.display = "inline");
        canvii[1].style.float = "right";
    })

function lookRadsToVec(raw_vals) {
    // Look direction comes in as [yaw, pitch] floats
    // referenced s.t. pos pitch is down, 0 yaw = pos-z, and pos yaw is CCW
    // Three.js has pos pitch up, 0 yaw = pos-x, and pos yaw is CW
    const look_angles = [(-1)*raw_vals[0], ((-1)*raw_vals[1]) + Math.PI/2]
    // Convert pitch and yaw radian values to a Vector3 look direction
    let look_dir = new THREE.Vector3( 
        Math.cos(look_angles[0]) * Math.cos(look_angles[1]),
        Math.sin(look_angles[1]),
        Math.sin(look_angles[0]) * Math.cos(look_angles[1])
        );
    look_dir.normalize();

    return look_dir;
}

function init(scene, idx) {
    let user_pos = scene.avatarInfo.pos;
    let agent_pos = scene.agentInfo.pos;
    let user_look = lookRadsToVec(scene.avatarInfo.look);
    //let agent_look = lookRadsToVec(scene.agentInfo.look);
    const head_position = new THREE.Vector3((user_pos[0]*50)+25, (user_pos[1]*50)+100, (user_pos[2]*50)+25);
    const userMaterial = new THREE.MeshLambertMaterial( { color: 0xffff00 } );
    const agentMaterial = new THREE.MeshLambertMaterial( { color: 0x0000ff } );
    const groundMaterial = new THREE.MeshBasicMaterial( { color: 0xffffff, opacity: 1.0} );
    const targetGeo = new THREE.BoxGeometry( 0, 50, 50 );
    const targetMaterial = new THREE.MeshLambertMaterial( { color: 0xfeb74c, map: new THREE.TextureLoader().load( 'target.png' ), transparent: true } );

    cameras[idx] = new THREE.PerspectiveCamera( 50, (window.innerWidth/2) / (window.innerHeight - 50), 1, 10000 );
    cameras[idx].position.set( 400, 640, 1040 );
    cameras[idx].lookAt( 0, 0, 0 );

    scenes[idx] = new THREE.Scene();
    scenes[idx].background = new THREE.Color( 0xf0f0f0 );

    if (idx === 2) {
        // some things only for scene #2
        scenes[idx].add( rollOverMesh );

        // pointer for clicking
        raycasters[idx] = new THREE.Raycaster();
        pointers[idx] = new THREE.Vector2();
    }

    // grid
    // const gridHelper = new THREE.GridHelper( scene.length*50 , scene.length );
    // scenes[idx].add( gridHelper );

    // plane
    const geometry = new THREE.PlaneGeometry( 1000, 1000 );
    geometry.rotateX( - Math.PI / 2 );
    planes[idx] = new THREE.Mesh( geometry, new THREE.MeshBasicMaterial( { visible: false } ) );
    scenes[idx].add( planes[idx] );
    objects[idx].push( planes[idx] );

    // find origin offset so that scene is centerd on 0,0
    let Xs = scene.blocks.map(function(x) { return x[0]; });
    let origin_offset = Math.floor(Math.max(...Xs) / 2)

    // load scene
    for (let i=0; i<scene.blocks.length; i++) {
        let cubeMaterial;
        if (scene.blocks[i][3] === 46) {  // if it's the ground, skip the texture and add lines instead
            cubeMaterial = new THREE.MeshBasicMaterial( { color: 0xffffff, opacity: 1.0 } );
            const edges = new THREE.EdgesGeometry( geo );  // outline the white blocks for visibility
            const line = new THREE.LineSegments( edges, new THREE.LineBasicMaterial( { color: 0x000000 } ) );
            line.position.set(((scene.blocks[i][0] - origin_offset)*50)+25, (scene.blocks[i][1]*50)+25, ((scene.blocks[i][2] - origin_offset)*50)+25);
            scenes[idx].add( line );
        }
        else {
            cubeMaterial = new THREE.MeshLambertMaterial( { 
                color: BLOCK_MAP[scene.blocks[i][3]], 
                map: new THREE.TextureLoader().load( 'square-outline-textured.png' ) 
            });
        }
        const voxel = new THREE.Mesh( geo, cubeMaterial );
        voxel.position.set(((scene.blocks[i][0] - origin_offset)*50)+25, (scene.blocks[i][1]*50)+25, ((scene.blocks[i][2] - origin_offset)*50)+25);
        scenes[idx].add( voxel );
        objects[idx].push( voxel );
    }

    // add axes helper
    //The X axis is red. The Y axis is green. The Z axis is blue.
    scenes[idx].add( new THREE.AxesHelper( 10000 ) );

    // add look direction and init look raycaster - MUST BE AFTER SCENE, BEFORE USER
    scenes[idx].updateMatrixWorld();  // Must call this for raycasting to work before render
    scenes[idx].add( new THREE.ArrowHelper( user_look, head_position, 150, 0xff0000, 40, 20 ) );
    user_raycasters[idx] = new THREE.Raycaster(head_position, user_look);
    const intersects = user_raycasters[idx].intersectObjects( objects[idx] );
    if ( intersects.length > 0 ) {
        const intersect = intersects[ 0 ];
        let target = new THREE.Mesh( targetGeo, targetMaterial );
        target.position.set(intersect.point.x, intersect.point.y, intersect.point.z);
        target.rotation.y += user_look.y;
        target.rotation.z -= user_look.z;
        scenes[idx].add(target);
    }

    // add user and agent avatars
    addAvatar(scenes[idx], user_pos, userMaterial, scene.avatarInfo.look);
    addAvatar(scenes[idx], agent_pos, agentMaterial, scene.agentInfo.look);

    // lights
    const ambientLight = new THREE.AmbientLight( 0x606060 );
    scenes[idx].add( ambientLight );
    const directionalLight = new THREE.DirectionalLight( 0xffffff );
    directionalLight.position.set( 1, 0.75, 0.5 ).normalize();
    scenes[idx].add( directionalLight );

    renderers[idx] = new THREE.WebGLRenderer( { antialias: true } );
    renderers[idx].setPixelRatio( window.devicePixelRatio );
    renderers[idx].setSize( (window.innerWidth/2.1), (window.innerHeight - 50) );
    let cont = document.getElementById("voxel_painter");
    cont.appendChild( renderers[idx].domElement );

    // controls
    controls[idx] = new OrbitControls( cameras[idx], renderers[idx].domElement );
    controls[idx].listenToKeyEvents( window );
    controls[idx].addEventListener( 'change', render );

    controls[idx].enableZoom = false;
    controls[idx].minPolarAngle = (0.5 * Math.PI) / 4;
    controls[idx].maxPolarAngle = (2.0 * Math.PI) / 4;
}

function addAvatar(scene, position, material, look_dir) {
    const loader = new GLTFLoader();
    loader.load( './body.glb', function ( gltf ) {
        let model = gltf.scene;
        model.scale.multiplyScalar(75.0);
        model.position.set((position[0]*50)+25, (position[1]*50), (position[2]*50)+25)
        model.rotation.y += look_dir[0]; //yaw, referenced a la the raw vals for some reason
        scene.add( model );
        model.traverse( function ( object ) {
            if ( object.isMesh ) {
                object.castShadow = false;
                object.material = material;
            }
        } );
    } );
}

function addEventListeners() {
    document.addEventListener( 'pointermove', onPointerMove );
    document.addEventListener( 'pointerdown', onPointerDown );
    document.addEventListener( 'keydown', onDocumentKeyDown );
    document.addEventListener( 'keyup', onDocumentKeyUp );
    window.addEventListener( 'resize', onWindowResize );
}

function onWindowResize() {
    for (let idx in cameras) {
        cameras[idx].aspect = (window.innerWidth/2) / (window.innerHeight - 50);
        cameras[idx].updateProjectionMatrix();
        renderers[idx].setSize( (window.innerWidth/2.1), (window.innerHeight - 50) );
    }
    render();
}

function onPointerMove( event ) {
    pointers[2].set( ( (event.clientX - (window.innerWidth/1.9)) / (window.innerWidth/2.1) ) * 2 - 1, - ( event.clientY / (window.innerHeight - 50) ) * 2 + 1 );
    raycasters[2].setFromCamera( pointers[2], cameras[2] );
    // Select which blocks the raycaster should intersect with
    let objs_to_intersect = objects[2];
    if (isSDown) objs_to_intersect = objects[2].concat(marked_blocks);
    else if (isDDown) objs_to_intersect = marked_blocks;
    const intersects = raycasters[2].intersectObjects( objs_to_intersect, false );

    if ( intersects.length > 0 ) {
        const intersect = intersects[ 0 ];
        if (isADown || isDDown){ // overlap existing cubes or marked blocks
            rollOverMesh.position.copy( intersect.object.position );
        } else {  // S down is also the default, with is to show the rollover on top of existing blocks
            rollOverMesh.position.copy( intersect.point ).add( intersect.face.normal );
            rollOverMesh.position.divideScalar( 50 ).floor().multiplyScalar( 50 ).addScalar( 25 );
        }
        
    }
    cameras[2].position.set( cameras[1].position.x, cameras[1].position.y, cameras[1].position.z );
    cameras[2].lookAt( 0, 0, 0 );
    render();
}

function onPointerDown( event ) {
    pointers[2].set( ( (event.clientX - (window.innerWidth/1.9)) / (window.innerWidth/2.1) ) * 2 - 1, - ( event.clientY / (window.innerHeight - 50) ) * 2 + 1 );
    raycasters[2].setFromCamera( pointers[2], cameras[2] );
    
    // Select which blocks the raycaster should intersect with
    let objs_to_intersect = [];
    if (isADown) objs_to_intersect = objects[2];
    else if (isSDown) objs_to_intersect = objects[2].concat(marked_blocks);
    else if (isDDown) objs_to_intersect = marked_blocks;
    const intersects = raycasters[2].intersectObjects( objs_to_intersect, false );

    if ( intersects.length > 0 ) {
        const intersect = intersects[ 0 ];

        // mark cube
        if ( isADown ) {
            if ( intersect.object !== planes[2] ) {
                // Remove the old block from the scene
                scenes[2].remove( intersect.object );
                objects[2].splice( objects2.indexOf( intersect.object ), 1 );
                // Add in a marked block in the same spot
                const voxel = new THREE.Mesh( geo, cubeMaterial_mark );
                voxel.position.set(intersect.object.position.x,intersect.object.position.y,intersect.object.position.z);
                scenes[2].add( voxel );
                marked_blocks.push( voxel );
                // Record the action to be able to undo
                actions_taken.push( [intersect.object, voxel, "mark_cube"] );
            }

        // mark air
        } else if (isSDown) {
            // Create new marked block and place on the surface
            const voxel = new THREE.Mesh( geo, cubeMaterial_mark );
            voxel.position.copy( intersect.point ).add( intersect.face.normal );
            voxel.position.divideScalar( 50 ).floor().multiplyScalar( 50 ).addScalar( 25 );
            scenes[2].add( voxel );
            marked_blocks.push( voxel );
            // Record the action to be able to undo
            actions_taken.push( [null, voxel, "mark_air"] );

        // unmark block
        } else if (isDDown) {
            if ( intersect.object !== planes[2] ) {
                // Remove the marked block from the scene
                scenes[2].remove( intersect.object );
                marked_blocks.splice( marked_blocks.indexOf( intersect.object ), 1 );
                // If it existed originally, replace with a the original object
                let voxel = null;
                objects[1].forEach(obj => {
                    if (obj.position.equals(intersect.object.position)) {
                        voxel = obj.clone()
                        scenes[2].add( voxel );
                        objects[2].push( voxel );
                    }
                });
                // Record the action to be able to undo
                actions_taken.push( [intersect.object, voxel, "unmark_block"] );
            }
        }
        render();
    }
}

function onDocumentKeyDown( event ) {
    switch ( event.keyCode ) {
        case 65: isADown = true; break;
        case 83: isSDown = true; break;
        case 68: isDDown = true; break;
        case 90:  // ctrl-z to undo
            let action = actions_taken.pop();
            if (action){
                switch (action[2]) {
                    case "mark_cube":
                        // Remove the marked block
                        scenes[2].remove( action[1] );
                        marked_blocks.splice( marked_blocks.indexOf( action[1] ), 1 );
                        // Put back the original cube
                        scenes[2].add( action[0] );
                        objects[2].push( action[0] );
                        break;
                    case "mark_air":
                        // Remove the marked block
                        scenes[2].remove( action[1] );
                        marked_blocks.splice( marked_blocks.indexOf( action[1] ), 1 );
                        break;
                    case "unmark_block":
                        // If there's a cube there, remove it
                        if (action[1]){
                            scenes[2].remove( action[1] );
                            objects[2].splice( objects2.indexOf( action[1] ), 1 );
                        }
                        // Put back the marked block
                        scene[2].add( action[0] );
                        marked_blocks.push( action[0] );
                        break;
                }
                render();
            }
            break;
    }
}

function onDocumentKeyUp( event ) {
    switch ( event.keyCode ) {
        case 65: isADown = false; break;
        case 83: isSDown = false; break;
        case 68: isDDown = false; break;
    }
}

function render() {
    renderers[1].render( scenes[1], cameras[1] );
    renderers[2].render( scenes[2], cameras[2] );

    if ((marked_blocks.length > 0) && (!startedHIT)) {
        startedHIT = true;
        window.parent.postMessage(JSON.stringify({ msg: "block_marked" }), "*");
    }

    let output_list = [];
    marked_blocks.forEach((block) => {
        let positionArray = block.position.toArray();
        let scaledArray = positionArray.map(function(item) { return (item-25)/50 });
        output_list.push(scaledArray);
    })
    document.getElementById("markedBlocks").value = JSON.stringify(output_list);
}