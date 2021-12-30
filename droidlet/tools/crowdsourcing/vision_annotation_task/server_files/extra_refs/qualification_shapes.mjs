class qualShapes {

    constructor (qnum) {

        this.starting_cube = function () {  // Will need to fix this later to match the scene format
			let starting_cube = [];
            let material = 58; // Brown wool
            for (let x=-2; x<2; x++){
                for (let y=0; y<4; y++){
                    for (let z=-2; z<2; z++){
                        starting_cube.push([x,y,z,material])
                    }
                }
            }
            return starting_cube
		};

        switch (qnum) {
            case 1:
                this.scene = {
                    "avatarInfo": null,
                    "agentInfo": null, 
                    "blocks": 
                        [[2,0,2,60], [2,1,2,60], [2,2,2,60],  //red tower
                        [2,0,-2,51], [2,1,-2,51], [2,2,-2,51],  //green tower
                        [-2,0,2,57], [-2,1,2,57], [-2,2,2,57],  //blue tower
                        [-2,0,-2,61], [-2,1,-2,61], [-2,2,-2,61]]  //black tower
                };
                break;
            case 2:
                this.scene = {
                    "avatarInfo": {
                        "pos": [-5, 0, -3], 
                        "look": [0.0, 0.0]
                    },
                    "agentInfo": {
                        "pos": [6, 0, 2], 
                        "look": [0.0, 0.0]
                    },
                    "blocks": 
                        [[5,0,0,58],  // three cubes
                        [5,1,0,58],
                        [4,0,0,58],
                        [4,1,0,58],
                        [5,0,-1,58],
                        [5,1,-1,58],
                        [4,0,-1,58],
                        [4,1,-1,58],
                        [0,0,0,58],
                        [0,1,0,58],
                        [-1,0,0,58],
                        [-1,1,0,58],
                        [0,0,-1,58],
                        [0,1,-1,58],
                        [-1,0,-1,58],
                        [-1,1,-1,58],
                        [-5,0,0,58],
                        [-5,1,0,58],
                        [-4,0,0,58],
                        [-4,1,0,58],
                        [-5,0,-1,58],
                        [-5,1,-1,58],
                        [-4,0,-1,58],
                        [-4,1,-1,58]]
                };
                break;
            case 3:
                this.scene = {
                    "avatarInfo": null,
                    "agentInfo": null, 
                    "blocks": 
                        [[2,0,2,58],  // Cube and L
                        [2,1,2,58],
                        [2,2,2,58],
                        [1,0,2,58],
                        [0,0,2,58],
                        [-1,0,2,58],
                        [-2,0,-2,58],
                        [-2,1,-2,58],
                        [-2,0,-1,58],
                        [-2,1,-1,58],
                        [-1,0,-2,58],
                        [-1,1,-2,58],
                        [-1,0,-1,58],
                        [-1,1,-1,58]]
                };
                break;
            case 4:
                this.scene = {
                    "avatarInfo": null,
                    "agentInfo": null, 
                    "blocks": 
                        [[0,0,0,58],  // Small square on ground
                        [-1,0,0,58],
                        [0,0,-1,58],
                        [-1,0,-1,58]]
                };
                break;
            default:
                this.scene = null;
                console.log("Invalid qualification question number received, something is wrong.");
        }
    }
}

export { qualShapes };