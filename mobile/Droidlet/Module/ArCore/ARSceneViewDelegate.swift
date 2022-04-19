import Foundation
import ARKit
import CoreLocation

class ARSceneViewDelegate: NSObject, ARSCNViewDelegate {
    //Singleton
    static let ARSCNViewDelegateInstance = ARSceneViewDelegate()
    //MARK: - Properties
    
    //Referenced beacon that is scanned. All beacons located inside Assets.xcassets/ARResources
    var beaconImageName: String?
    //Serves two functions. One as a helper to build custom maps. One as a tracker of all plotted nodes. It is either one or the other and is reinitialized depending on if the user is navigating or creating a custom map.
    var nodeList: Array<SCNNode>?
    //Delegate = ViewController.swift
    var delegate: ARScnViewDelegate?
    //Centralized datasource
    var dataModelSharedInstance: DataModel?
    
    //MARK: - Init
    override init() {
        super.init()
        self.dataModelSharedInstance = DataModel.dataModelSharedInstance
    }
    //MARK: - HelperFunctions
    
    private func resetNodeList(){
        self.nodeList = Array<SCNNode>()
    }

    func reset(){
        self.beaconImageName = nil
        self.nodeList = nil
    }

    func setUpNavigation(renderedBeaconNode: SCNNode) -> SCNNode{
        //Reinitializes nodeList = nil
        resetNodeList()
        //List of nodes to traverse
        let list = self.dataModelSharedInstance!.getNodeManager().getNodeList()
        
        //reference to origin node
        var buildingNode = renderedBeaconNode
        //Sets the last referenced node (source node) to the marker node
        dataModelSharedInstance!.getNodeManager().setLastReferencedNode(node: renderedBeaconNode)
        //Traverses nodeList, adding nodes to destination
        for (index, _) in list.enumerated() {
            buildingNode = self.placeNode(subNodeSource: renderedBeaconNode, to: list[index])
        }
        //Places arrows above the nodes within nodeList using the marker node as its origin
        self.placeArrowNodes(sourceNode: renderedBeaconNode)
        //Kicks off an async function to determine if user reached destination (camera node within a certain distance of the last node)
        //self.checkDestinationLoop()
        return buildingNode
    }

    private func placeNode(subNodeSource: SCNNode, to: Index) -> SCNNode{
        let sphere = SCNSphere(radius: 0.05)
        //Makes the AR Node a sphere
        let node = SCNNode(geometry: sphere)
        node.geometry?.firstMaterial?.diffuse.contents = UIColor.blue
        node.name = "test1"
        
        let xOffset = to.xOffset
        let yOffset = to.yOffset
        let zOffset = to.zOffset
        
        //builds an array with x,y,z in it
        let transformationArray = buildArray(x: xOffset, y: yOffset, z: zOffset)
        //Gets the last referencednode, this node is already plotted and its parent is the markernode source. Plotting a node against this lastReferencedNode will take into account the lastReferencedNode position, and add on any translations to the new node being created. Thus all translations to and from any node is from the marker node and the new node being created.
        let lastNode = dataModelSharedInstance!.getNodeManager().getLastReferencedNode()
        
        //Gets the 4x4 matrix float of the last referenced node
        let referenceNodeTransform = matrix_float4x4(lastNode!.transform)
        var translation = matrix_identity_float4x4
        
        translation.columns.3.x = transformationArray[0]
        translation.columns.3.y = transformationArray[1]
        translation.columns.3.z = transformationArray[2]
        
        //Multiplies the lastReferencedNode position by the translation to the new node being created. This sets the new node's position
        node.simdTransform = matrix_multiply(referenceNodeTransform, translation)
        
        //Adds the new node to the marker node source
        subNodeSource.addChildNode(node)
        
        //Checks to see if a line needs to be added. Adds a line above only if the new node is not a starting node. LocationInfo.NodeType
        if (to.type != NodeType.start.rawValue){
            placeLine(sourceNode: subNodeSource, from: lastNode!, to: node)
        }
        
        //Adds the new node created to the nodeList
        self.nodeList!.append(node)
        //Sets the last referenced node to the new node being created which enables the methodology above.
        dataModelSharedInstance!.getNodeManager().setLastReferencedNode(node: node)
        
        return subNodeSource
    }

    private func placeLine(sourceNode: SCNNode, from: SCNNode, to: SCNNode){
        let node = SCNGeometry.cylinderLine(from: from.position, to: to.position, segments: 5)
        node.name = "test1"
        sourceNode.addChildNode(node)
        //Checker to see if the user is building a custom map. If yes, then it refers to the singleton NodeManager.swift and adds a line node.
        if (dataModelSharedInstance!.getLocationDetails().getIsCreatingCustomMap()){
            dataModelSharedInstance!.getNodeManager().addLineNode(node: node)
        }
    }

    private func buildArray(x: Float, y: Float, z: Float) -> Array<Float>{
        var returningArray = Array<Float>()
        returningArray.append(x)
        returningArray.append(y)
        returningArray.append(z)
        return returningArray
    }

    private func placeArrowNodes(sourceNode: SCNNode){
        //the nodeList that needs to be traversed.
        let traverseList = self.nodeList
        let size = traverseList!.count - 1
        for (index, _) in traverseList!.enumerated() {
            if (index != size){
                let node1 = traverseList![index]
                let node2 = traverseList![index + 1]
                
                let referenceNodeTransform = matrix_float4x4(node1.transform)
                var translation = matrix_identity_float4x4
                
                translation.columns.3.x = 0
                translation.columns.3.y = 0
                //Raises the position of the arrow above the node (z value)
                translation.columns.3.z = Float(ArkitNodeDimension.arrowNodeXOffset) * -1
                
                //returns a clone of a SCNNode which was already initialized when NodeManager was initialized.
                let arrow = dataModelSharedInstance!.getNodeManager().getArrowNode()
                arrow.simdTransform = matrix_multiply(referenceNodeTransform, translation)
            
                sourceNode.addChildNode(arrow)
                //The way the arrow's x,y,z is setup in art.scnassets/arrow.scn allows the arrow to point perfectly towards node2.position when calling SCNNode.look.
                arrow.look(at: node2.position)
            }
        }
    }

    private func placeBuildingNode(sourceNode: SCNNode, lastNode: SCNNode, targetNode: Index){
        let sphere = SCNSphere(radius: 0.03)
        let node = SCNNode(geometry: sphere)
        //Determines color of node
        switch targetNode.type{
        case NodeType.start.rawValue:
            node.geometry?.firstMaterial?.diffuse.contents = UIColor.white
        case NodeType.destination.rawValue:
            node.geometry?.firstMaterial?.diffuse.contents = UIColor.white
        case NodeType.intermediate.rawValue:
            node.geometry?.firstMaterial?.diffuse.contents = UIColor.blue
        default:
            break
        }
        
        let referenceNodeTransform = matrix_float4x4(lastNode.transform)
        var translation = matrix_identity_float4x4
        translation.columns.3.x = targetNode.xOffset
        translation.columns.3.y = targetNode.yOffset
        translation.columns.3.z = targetNode.zOffset
        //sets the new node position to the lastReferencedNode.transform multiplied by the translation.
        node.simdTransform = matrix_multiply(referenceNodeTransform, translation)
        
        //Sets the last referened node to the new node.
        dataModelSharedInstance!.getNodeManager().setLastReferencedNode(node: node)
        //Adds a ScnNode to the NodeManager.scnNodeList
        dataModelSharedInstance!.getNodeManager().addScnNode(node: node)
        //Adds the new node to the marker node.
        sourceNode.addChildNode(node)
    }
    
    /*/ resetToCustomMapStartNode()
     If the undo button is pressed while creating a custom map all the way to the beginning, it will reset the scene to the start state.
     */
    private func resetToCustomMapStartNode() {
        //Sets the NodeManager.startingNodeIsSet = false, which indicates that the user needs to place the start node when building a custom map.
        dataModelSharedInstance!.getNodeManager().setStartingNodeIsSet(isSet: false)
        //Instructs the user with instructions on how to create  custom map. Text is located at Constants.TextConstants
    }

    private func getNodeDataAndPlotBuildingNode(type: NodeType) {
		//Gets current position using the camera as a reference
		let cameraTransform = dataModelSharedInstance!.getSceneView().session.currentFrame!.camera.transform
		let cameraPosition = SCNVector3Make(cameraTransform.columns.3.x,
											cameraTransform.columns.3.y,
											cameraTransform.columns.3.z)
		//The source marker node
		let sourceNode = dataModelSharedInstance!.getNodeManager().getReferencedBeaconNode()

		var referencedNode: SCNNode
		//Determines what node to use as a reference to plot new node
        if (type == NodeType.start){
            referencedNode = sourceNode!
        } else {
            referencedNode = dataModelSharedInstance!.getNodeManager().getLastReferencedNode()!
        }
        //position of the reference (last traversed) node
        let referencedPosition = referencedNode.position
        
        //Finds the delta from the last referenced node and the camera node.
        let xDist = cameraPosition.x - referencedPosition.x
        let yDist = cameraPosition.y - referencedPosition.y
        let zDist = cameraPosition.z - referencedPosition.z
        
        var newNode: Index
        //Constructs a new node depending on the type.
        if (type == NodeType.destination || type == NodeType.intermediate) {
            if (type == NodeType.destination){
                newNode = Index(type: NodeType.destination.rawValue, xOffset: xDist, yOffset: yDist, zOffset: zDist)
            } else {
                newNode = Index(type: NodeType.intermediate.rawValue, xOffset: xDist, yOffset: yDist, zOffset: zDist)
            }
            //Adds a new SCNNode to the screen when the user added a new node.
            placeBuildingNode(sourceNode: sourceNode!, lastNode: referencedNode, targetNode: newNode)
            //Get the last referenced SCNNode. the function above this "placeBuildingNode" adds its newly created SCNNode to the NodeManager.scnNodeList. getting this node allows for the next function to work.
            let currentNode = dataModelSharedInstance!.getNodeManager().getLastScnNode()
            //Places a line referenced to the marker node, from the lastReferencedNode, to the new SCNNode just created.
            placeLine(sourceNode: sourceNode!, from: referencedNode, to: currentNode!)
        } else {
            // Constructs a new Node of type NodeType.Start
            newNode = Index(type: NodeType.start.rawValue, xOffset: xDist, yOffset: yDist, zOffset: zDist)
            placeBuildingNode(sourceNode: sourceNode!, lastNode: referencedNode, targetNode: newNode)
        }
        //Adds to NodeManager.nodeList the new SCNNode Created
        dataModelSharedInstance!.getNodeManager().addNode(node: newNode)
    }
    
    //MARK: - Handler Functions
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        guard let planeAnchor = anchor as? ARPlaneAnchor else { return }

        let width = CGFloat(planeAnchor.extent.x)
        let height = CGFloat(planeAnchor.extent.z)
        let plane = SCNPlane(width: width, height: height)

        plane.materials.first?.diffuse.contents = UIColor.blue.withAlphaComponent(0.3)

        let planeNode = SCNNode(geometry: plane)

        let x = CGFloat(planeAnchor.center.x)
        let y = CGFloat(planeAnchor.center.y)
        let z = CGFloat(planeAnchor.center.z)

        planeNode.position = SCNVector3(x, y, z)
        planeNode.eulerAngles.x = -.pi / 2

        node.addChildNode(planeNode)
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
        guard let _ = anchor as?  ARPlaneAnchor,
              let planeNode = node.childNodes.first
        else { return }
        
        planeNode.removeFromParentNode()
    }

	func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
		print("Detected UPdate")
	}
	
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        var node: SCNNode?
        //Checks to see if either the user is creating a custom map or started up navigation and if the marker root node has not been scanned yet.
        if (dataModelSharedInstance!.getLocationDetails().getIsNavigating()
			|| dataModelSharedInstance!.getLocationDetails().getIsCreatingCustomMap()
			&& !dataModelSharedInstance!.getLocationDetails().getIsBeaconRootNodeFound()) {
            
            //Locks the WorldOrigin to the first beacon scanned
            if (!dataModelSharedInstance!.getLocationDetails().isWorldOriginSet) {
                dataModelSharedInstance!.getSceneView().session.setWorldOrigin(relativeTransform: anchor.transform)
//                dataModelSharedInstance!.getSceneView().debugOptions = [ARSCNDebugOptions.showWorldOrigin]
//                Notifies inside the centralized datahub that the world origin is set to the marker node
                dataModelSharedInstance!.getLocationDetails().setIsWorldOriginSet(isSet: true)
            }
            
			//If a marker is scanned and recognized, this function is ran. It checks to see if user is navigating already and if so, will do the proper setup to kickoff navigation
			if dataModelSharedInstance!.getLocationDetails().getIsNavigating() {
				//Returns a node with an AR Object indicating the marker location
				node = returnBeaconHighlightNode(anchor: anchor)
				//Sets up navigation using the marker node as the origin for all other nodes.
				node = self.setUpNavigation(renderedBeaconNode: node!)
				//Sets world node for intermediate nodes
				self.dataModelSharedInstance!.getNodeManager().setReferencedBeaconNode(node: node!)
				DispatchQueue.main.async{
					self.delegate?.toggleScanToNavigate(shouldShow: true)
				}
			} else {
				//Returns the marker node with an AR Object highlighted to show its position
				node = returnBeaconHighlightNode(anchor: anchor)
				//Sets world node for intermediate nodes
				dataModelSharedInstance!.getNodeManager().setReferencedBeaconNode(node: node!)
				DispatchQueue.main.async{
					self.delegate?.toggleAddButton(shouldShow: true)
				}
			}
        } else {
            //If the user is not building a custom map or navigating, returns just the marker node with an AR Object bound to its location
            node = returnBeaconHighlightNode(anchor: anchor)!
			self.dataModelSharedInstance!.getNodeManager().setReferencedBeaconNode(node: node!)
        }
        return node
    }
    
    //MARK: - Helper Handler Functions
	private func returnBeaconHighlightNode(anchor: ARAnchor) -> SCNNode? {
		let node = SCNNode()
		
		//Validates the marker that was scanned.
		if let imageAnchor = anchor as? ARImageAnchor{
			let size = imageAnchor.referenceImage.physicalSize
			let plane = SCNPlane(width: size.width, height: size.height)
			plane.firstMaterial?.diffuse.contents = UIColor.white.withAlphaComponent(0)
			plane.cornerRadius = 0.005
			
			let planeNode = SCNNode(geometry: plane)
			planeNode.eulerAngles.x = -.pi / 2
			//Assumes that the image is upright on a vertical surface.
			node.addChildNode(planeNode)
			
			var shapeNode : SCNNode?
			//Retrieves the referenceName inside the Assets.xcassets/ARResources
			beaconImageName = imageAnchor.referenceImage.name
			
			//If the marker node has not been scanned while navigating or creating a custom map, this function will run
			if (!dataModelSharedInstance!.getLocationDetails().getIsBeaconRootNodeFound()){
				if (dataModelSharedInstance!.getLocationDetails().getIsNavigating() || dataModelSharedInstance!.getLocationDetails().getIsCreatingCustomMap()){
					self.dataModelSharedInstance!.getNodeManager().setReferencedBeaconName(name: beaconImageName)
					dataModelSharedInstance!.getLocationDetails().setIsBeaconRootNodeFound(isFound: true)
				}
			}
			
			//This is the area where you are able to choose what markers you want to use. The way this is setup right now is that every marker returns the same uniform AR Object.
			if imageAnchor.referenceImage.name == "marker-1" || imageAnchor.referenceImage.name == "marker-2" {
				shapeNode = dataModelSharedInstance!.getNodeManager().getbeaconNode()
			}
			guard let shape = shapeNode else {return nil}
			
			//Adds the ARObject to the marker position
			node.addChildNode(shape)
		}
		return node
	}
    
    //MARK: - Timer Functions
    private func checkDestinationLoop() {
        //Gets the last node, destination node
        let lastNode = dataModelSharedInstance!.getNodeManager().getLastReferencedNode()
        let lastNodePosition = lastNode!.position
        
        var cameraTransform = dataModelSharedInstance!.getSceneView().session.currentFrame!.camera.transform
        var cameraPosition = SCNVector3Make(cameraTransform.columns.3.x,
        cameraTransform.columns.3.y, cameraTransform.columns.3.z)
        
        var distance = lastNodePosition.distance(receiver: cameraPosition)

        //Allows the asycnrhonous portion of this function
        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global(qos: .default).async {
            while distance > 1.5 && self.dataModelSharedInstance!.getLocationDetails().getIsNavigating(){
                    cameraTransform = self.dataModelSharedInstance!.getSceneView().session.currentFrame!.camera.transform
                    cameraPosition = SCNVector3Make(cameraTransform.columns.3.x,
                    cameraTransform.columns.3.y, cameraTransform.columns.3.z)
                    
                    distance = lastNodePosition.distance(receiver: cameraPosition)
                    usleep(500000)
            }
            group.leave()
        }
        //Handles the completion of the group.leave() function call when the destination is within range
        group.notify(queue: .main) {
            self.delegate!.destinationReached()
        }
    }
    
}

extension ARSceneViewDelegate {
    func handleEndButton() {
        //Sets within the data center that the destination node when creating a custom map is set.
        dataModelSharedInstance!.getNodeManager().setDestinationNodeIsSet(isSet: true)
        //Plots the destination node at the current camera position.
        getNodeDataAndPlotBuildingNode(type: NodeType.destination)        
    }

    func handleUndoButton() {
        //Gets all necessary nodes that were added from the last add button.
        let scnNode = dataModelSharedInstance!.getNodeManager().getLastScnNode()
        let node = dataModelSharedInstance!.getNodeManager().getLastNode()
        let lineNode = dataModelSharedInstance!.getNodeManager().getLastLineNode()
        
        //Checks to see if the last node added was type NodeType.destination. If so, it will remove the save button and toggle the Add & End button.
        if (node!.type == NodeType.destination.rawValue) {
            delegate!.toggleEndButton(shouldShow: true)
            delegate!.toggleAddButton(shouldShow: true)
            delegate!.toggleSaveButton(shouldShow: false)
        }
        //Checks to see if the node is not nil. Ensures that it will only remove the recent node.
        if (scnNode != nil && node != nil){
            dataModelSharedInstance!.getNodeManager().removeLastScnNode()
            dataModelSharedInstance!.getNodeManager().removeLastNode()
            scnNode!.removeFromParentNode()
        }
        //Checks to see if the lineNode added is not nil.
        if (lineNode != nil) {
            dataModelSharedInstance!.getNodeManager().removeLastLineNode()
            lineNode!.removeFromParentNode()
        }
        //Retrieves the sizes of the scnNodeList and nodeList within NodeManager.
        let sizeOfScnNodeList = dataModelSharedInstance!.getNodeManager().getLengthOfScnNodeList()
        let sizeOfNodeList = dataModelSharedInstance!.getNodeManager().getLengthOfNodeList()
        
        //If both sizes are 0, then it will reset to the start state of creating a custom map.
        if (sizeOfNodeList == 0 && sizeOfScnNodeList == 0){
            resetToCustomMapStartNode()
        }
        
        //Handles the proper pointer towards the last node.
        let lastReferenced = dataModelSharedInstance!.getNodeManager().getLastScnNode()
        if (lastReferenced != nil ){
            dataModelSharedInstance!.getNodeManager().setLastReferencedNode(node: lastReferenced!)
        }
    }
    func handleAddButton() {
        //Checks to see if the starting node is already set. If not, it will do the changes to enable the user to place intermediate/destination nodes.
        if (!dataModelSharedInstance!.getNodeManager().getStartingNodeIsSet()) {
            dataModelSharedInstance!.getNodeManager().setStartingNodeIsSet(isSet: true)

            //Plots the starting node, adds it to the data center, plots on the screen
            getNodeDataAndPlotBuildingNode(type: NodeType.start)            
        } else {
            //Plots the intermediate/destination node, adds it to the data center, plots on the screen
            getNodeDataAndPlotBuildingNode(type: NodeType.intermediate)
        }
    }
}

extension SCNVector3 {
    func distance(receiver: SCNVector3) -> Float {
        let xd = self.x - receiver.x
        let yd = self.y - receiver.y
        let zd = self.z - receiver.z
        let distance = Float(sqrt(xd * xd + yd * yd + zd * zd))
        if distance < 0 {
            return distance * -1
        } else {
            return distance
        }
    }
}

extension SCNNode {
    func distance(receiver: SCNNode) -> Float {
        let node1Pos = self.position
        let node2Pos = receiver.position
        let xd = node2Pos.x - node1Pos.x
        let yd = node2Pos.y - node1Pos.y
        let zd = node2Pos.z - node1Pos.z
        let distance = Float(sqrt(xd * xd + yd * yd + zd * zd))
        if distance < 0 {
            return distance * -1
        } else {
            return distance
        }
    }
}

extension SCNGeometry {
    class func cylinderLine(from: SCNVector3, to: SCNVector3, segments: Int) -> SCNNode{
        let x1 = from.x
        let x2 = to.x
        
        let y1 = from.y
        let y2 = to.y
        
        let z1 = from.z
        let z2 = to.z
        
        let distance = sqrtf((x2 - x1) * (x2 - x1) +
            (y2 - y1) * (y2 - y1) +
            (z2 - z1) * (z2 - z1))
        
        //Creates a SCNCylinder with the height of it being the distance from the two SCNVector3
        let cylinder = SCNCylinder(radius: 0.005,
                                   height: CGFloat(distance))
        
        cylinder.radialSegmentCount = segments
        cylinder.firstMaterial?.diffuse.contents = UIColor.white
        
        let lineNode = SCNNode(geometry: cylinder)
        
        //Sets the position of the lineNode's center to the inbetween of the first SCNVector3 and second SCNVector3. This accounts fo the proper size of the line segment when added to the source node (marker node).
        lineNode.position = SCNVector3(((from.x + to.x)/2),
                                       ((from.y + to.y)/2),
                                       ((from.z + to.z)/2))
        
        //Handles the orientation of the line
        lineNode.eulerAngles = SCNVector3(Float.pi/2,
                                          acos((to.z - from.z)/distance),
                                          atan2(to.y - from.y, to.x - from.x))
        
        return lineNode
    }
}
