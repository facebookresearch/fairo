import Foundation
import ARKit

class NodeManager {
    
    //MARK: - Properities
    
    static var nodeManagerSharedInstance = NodeManager()
    private var dataModelSharedInstance: DataModel?
    private var referencedBeaconName: String?
    private var referenceBeaconNode: SCNNode?
    private var lastReferencedNode: SCNNode?
    private var nodeList: Array<Index> = []
    private var scnNodeList: Array<SCNNode> = []
    private var lineNodeList: Array<SCNNode> = []
    private var startingNodeIsSet = false
    private var destinationNodeIsSet = false
    private var isNodeListGenerated = false
    private var destinationScnNode: SCNNode?
    
    private var beaconNode: SCNNode?
    private var arrowNode: SCNNode?
    
    //MARK: - Init
    
    /*/ init()
     Initializer for the class
     */
    private init (){
        self.dataModelSharedInstance = DataModel.dataModelSharedInstance
        configureArrow()
        configureBeacons()
    }
    
    //MARK: - SaveCustomMap
    
    /*/ retrieveCustomMap(_ destination: String) -> LocationInfo
     Retrieves a custom map when finished creating a custom map.
     Used when user clicks save after finishing construction of custom map.
     
     @param: destination - String - the destination of the custom map.
     */
    func retrieveCustomMap(_ destination: String) -> LocationInfo {
        let nodes = Nodes(index: nodeList)
        let locInfo = LocationInfo(destination: destination, beaconName: "marker-2", nodeCount: nodeList.count, nodes: nodes)
        print(locInfo)
        return locInfo
    }
    
    //MARK: - Node Configurations
    
    /*/ configureBeacons()
     Configures the model of the beacon.
     */
    private func configureBeacons(){
        DispatchQueue.main.async {
            let node = SCNNode()
            let beaconScene = SCNScene(named: "art.scnassets/diamond.scn")
            for childNode in beaconScene!.rootNode.childNodes{
                node.addChildNode(childNode)
            }
            self.beaconNode = node
        }
    }
    /*/ configureBeacons()
     Configures the model of the arrow.
     */
    private func configureArrow(){
        DispatchQueue.main.async {
            let node = SCNNode()
            let arrowScene = SCNScene(named: "art.scnassets/arrow.scn")
            for childNode in arrowScene!.rootNode.childNodes{
                node.addChildNode(childNode)
            }
            self.arrowNode = node
        }
    }
    
    //MARK: - Helper Functions

    /*/ reset()
     Resets the class to resting state
     */
    func reset(){
        self.lastReferencedNode = nil
        self.referencedBeaconName = nil
        self.referenceBeaconNode = nil
        self.nodeList = Array<Index>()
        self.lineNodeList = Array<SCNNode>()
        self.scnNodeList = Array<SCNNode>()
        self.startingNodeIsSet = false
        self.destinationNodeIsSet = false
        self.isNodeListGenerated = false
        self.destinationScnNode = nil
    }
    /*/ addNode(node: Index)
     Adds a node (LocationInfo.Index) to the nodeList
     
     @param: node - Index - a new node
     */
    func addNode(node: Index){
        self.nodeList.append(node)
    }
    /*/ addScnNode(node: SCNNode)
     adds a SCNNode to the scnNodeList
     
     @param: SCNNode - the SCNNode to be added to the scnNodeList
     */
    func addScnNode(node: SCNNode){
        scnNodeList.append(node)
    }
    /*/ addLineNode(node: SCNNode)
     adds a lineNode to the lineNodeList
     
     @param: SCNNode - the line SCNNode to be added to the lineNodeList
     */
    func addLineNode(node: SCNNode){
        lineNodeList.append(node)
    }
    /*/ removeLastScnNode()
     removes the last SCNNode from the scnNodeList
     */
    func removeLastScnNode(){
        scnNodeList.removeLast()
    }
    /*/ removeLastNode()
     removes the last node (LocationInfo.Index) from the nodeList
     */
    func removeLastNode(){
        nodeList.removeLast()
    }
    /*/ removeLastLineNode()
     removes the last line SCNNode from the lineNodeList
     */
    func removeLastLineNode(){
        lineNodeList.removeLast()
    }
    /*/ generateNodeList(completion: @escaping(Bool) -> Void
     generates a nodeList to be constructed to use for navigation. If it alreadt exist
     send a completion(true)
     */
    func generateNodeList(completion: @escaping(Bool) -> Void){
        if (!isNodeListGenerated) {
            var destination = self.dataModelSharedInstance!.getLocationDetails().getDestination()
            destination = destination.lowercased()

            let beaconNodeScannedName = referencedBeaconName

            let infoArray : [String?] = [
                destination,
                beaconNodeScannedName
            ]
            
            //Sends a network request to retrieve a Map from the server
//            NetworkService.networkServiceSharedInstance.requestNavigationInfo(URLConstants.navigationRequest, infoArr: infoArray){ result in
//                    switch result{
//                        case .failure(_):
//                            completion(false)
//                        case .success(let data):
//                            let jsonDecoded = Formatter.FormatterSharedInstance.decodeJSONDestination(JSONdata: data)
//                            if jsonDecoded != nil {
//                                self.nodeList = Formatter.FormatterSharedInstance.buildNodeListWithJsonData(jsonDecoded: jsonDecoded!)
//
//                                self.setIsNodeListGenerated(isSet: true)
//                                completion(true)
//                            } else { completion(false); }
//                    }
//                }
        } else {
            completion(true)
        }
    }
    
    //MARK: - Setter Functions
    
    /*/ setNodeList(list: Array<Index>)
     Sets the nodeList to an Array of Index(LocationInfo.Index)
     
     @param: list - Array<Index> - An Index Array
     */
    func setNodeList(list: Array<Index>){
        self.nodeList = list
    }
    /*/ setReferencedBeaconName(name: String?)
     Sets referencedBeaconName to a String of the referenced beacon
     
     @param: name - String - A String
     */
    func setReferencedBeaconName(name: String?){
        self.referencedBeaconName = name
    }
    /*/ setReferencedBeaconNode(node: SCNNode)
     Sets the referenceBeaconNode to SCNNode
     
     @param: node - SCNNode - A SCNNode
     */
    func setReferencedBeaconNode(node: SCNNode){
        self.referenceBeaconNode = node
    }
    /*/ setStartingNodeIsSet(isSet: Bool)
     sets startingNodeIsSet to a boolean value
     
     @param: isSet - Bool - boolean value
     */
    func setStartingNodeIsSet(isSet: Bool){
        self.startingNodeIsSet = isSet
    }
    /*/ setStartingNodeIsSet(isSet: Bool)
     sets destinationNodeIsSet to a boolean value
     
     @param: isSet - Bool - boolean value
     */
    func setDestinationNodeIsSet(isSet: Bool){
        self.destinationNodeIsSet = isSet
    }
    /*/ setLastReferencedNode (node: SCNNode)
     sets lastReferencedNode to a SCNNode
     
     @param: node - SCNNode - a SCNNode
     */
    func setLastReferencedNode (node: SCNNode){
        lastReferencedNode = node
    }
    /*/ setIsNodeListGenerated(isSet: Bool)
     sets isNodeListGenerated to a boolean value
     
     @param: isSet - Bool - boolean value
     */
    func setIsNodeListGenerated(isSet: Bool){
        self.isNodeListGenerated = isSet
    }
    /*/ setdestinationScnNode(node: SCNNode)
     sets destinationScnNode to a SCNNode
     
     @param: node - SCNNode - a SCNNode
     */
    func setdestinationScnNode(node: SCNNode){
        destinationScnNode = node
    }
    
    //MARK: - Getter Functions
    
    /*/ getNodeList() -> Array<Index>
     returns nodeList
     
     @return: nodeList - Array<Index> - the current nodeList stored in NodeManager
     */
    func getNodeList() -> Array<Index>{
        return self.nodeList
    }
    /*/ getReferencedBeaconName() -> String?
     returns the referencedBeaconName
     
     @return: referencedBeaconName - String - the referenced beacon's name
     */
    func getReferencedBeaconName() -> String? {
        return self.referencedBeaconName
    }
    /*/ getReferencedBeaconNode() -> SCNNode?
     returns the referencedBeaconNode
     
     @return: referenceBeaconNode - SCNNode - the referenced Beacon Node
     */
    func getReferencedBeaconNode() -> SCNNode? {
        return self.referenceBeaconNode
    }
    /*/ getStartingNodeIsSet() -> Bool
     returns startingNodeIsSet
     
     @return: startingNodeIsSet - Bool - boolean value which states if the starting node is set
     */
    func getStartingNodeIsSet() -> Bool {
        return self.startingNodeIsSet
    }
    /*/ getDestinationNodeIsSet() -> Bool
     returns destinationNodeIsSet
     
     @return: destinationNodeIsSet - Bool - boolean value which states if the destination node is set
     */
    func getDestinationNodeIsSet() -> Bool {
        return self.destinationNodeIsSet
    }
    /*/ getLastReferencedNode() -> SCNNode?
     returns the lastReferencedNode
     
     @return: lastReferencedNode - SCNNode - the last referenced SCNNode
     */
    func getLastReferencedNode() -> SCNNode? {
        return lastReferencedNode
    }
    /*/ getArrowNode() -> SCNNode
     returns a clone of the arrowNode
     
     @return: SCNNode - the clone of the ArrowNode
     */
    func getArrowNode() -> SCNNode {
        return self.arrowNode!.clone()
    }
    /*/ getbeaconNode() -> SCNNode
     returns a clone of the beaconNode
     
     @return: SCNNode - the clone of the beaconNode
     */
    func getbeaconNode() -> SCNNode {
        return self.beaconNode!.clone()
    }
    /*/ getLastLineNode() -> SCNNode?
     returns the last line node in lineNodeList
     
     @return: lineNodeList.last - SCNNode - last line Node
     */
    func getLastLineNode() -> SCNNode? {
        return lineNodeList.last
    }
    /*/ getLastNode() -> Index?
     returns nodeList.last
     
     @return: nodeList.last - Index - the last element in the nodeList
     */
    func getLastNode() -> Index? {
        return nodeList.last
    }
    /*/ getLastScnNode() -> SCNNode?
     returns the last scnNode in scnNodeList
     
     @return: scnNodeList.last - SCNNode - last scnNode
     */
    func getLastScnNode() -> SCNNode? {
        return scnNodeList.last
    }
    /*/ getLengthOfNodeList() -> Int
     returns length of nodeList
     
     @return: nodeList.count - Int - length of the list
     */
    func getLengthOfNodeList() -> Int{
        return nodeList.count
    }
    /*/getLengthOfScnNodeList() -> Int
     returns length of scnNodeList
     
     @return: scnNodeList.count - Int - length of the list
     */
    func getLengthOfScnNodeList() -> Int{
        return scnNodeList.count
    }
    /*/ getLengthOfLineNodeList() -> Int
     returns length of lineNodeList
     
     @return: lineNodeList.count - Int - length of the list
     */
    func getLengthOfLineNodeList() -> Int {
        return lineNodeList.count
    }
    /*/ getIsNodeListGenerated() -> Bool
     returns isNodeListGenerated
     
     @return: isNodeListGenerated - Bool - boolean value determining if nodeList is generated
     */
    func getIsNodeListGenerated() -> Bool {
        return isNodeListGenerated
    }
    /*/ getdestinationScnNode() -> SCNNode?
     returns the destinationScnNode
     
     @return: destinationScnNode - SCNNode - the destinationScnNode
     */
    func getdestinationScnNode() -> SCNNode?{
        return destinationScnNode
    }
}
