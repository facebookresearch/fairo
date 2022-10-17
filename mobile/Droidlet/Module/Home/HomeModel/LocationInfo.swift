import Foundation

enum LocatinInfoStringNames: String {
    case destination = "destination"
    case nodes = "nodes"
    case beaconName = "beacon_name"
    case nodeCount = "node_count"
}
class LocationInfo: NSObject, NSCoding, Codable {
    
    let destination: String
    let beaconName: String
    let nodeCount: Int
    let nodes: Nodes

    enum CodingKeys: String, CodingKey {
        case destination = "destination"
        case beaconName = "beacon_name"
        case nodeCount = "node_count"
        case nodes
    }
    /*/ encode(with coder: NSCoder)
     Encodes the data with the given key
     */
    func encode(with coder: NSCoder) {
        coder.encode(self.destination, forKey: LocatinInfoStringNames.destination.rawValue)
        coder.encode(self.nodeCount, forKey: LocatinInfoStringNames.nodeCount.rawValue)
        coder.encode(self.nodes, forKey: LocatinInfoStringNames.nodes.rawValue)
        coder.encode(self.beaconName, forKey: LocatinInfoStringNames.beaconName.rawValue)
    }
    /*/ init?(coder decoder: NSCoder)
     On init, decodes the data with the given key
     */
    required init?(coder decoder: NSCoder) {
        self.destination = decoder.decodeObject(forKey: LocatinInfoStringNames.destination.rawValue) as! String
        self.nodes = decoder.decodeObject(forKey: LocatinInfoStringNames.nodes.rawValue) as! Nodes
        self.beaconName = decoder.decodeObject(forKey: LocatinInfoStringNames.beaconName.rawValue) as! String
        self.nodeCount = decoder.decodeInteger(forKey: LocatinInfoStringNames.nodeCount.rawValue)
    }
    /*/ init(destination: String, beaconName: String, nodeCount: Int, nodes: Nodes)
     Inits the class with specific data
     
     @param: destination - String - destination name
     @param: beaconName - String - beaconScannedName
     @param: nodeCount - Int - amount of nodes
     @param: nodes - Nodes - nodes obj containning all of the nodes
     */
    init(destination: String, beaconName: String, nodeCount: Int, nodes: Nodes) {
        self.destination = destination
        self.beaconName = beaconName
        self.nodeCount = nodeCount
        self.nodes = nodes
    }
    /*/ getJSON() -> NSMutableDictionary
     Constructs a JSON representation of the data within this class and
     returns the json representation
     
     @return: dict - NSMutableDictionary - Json representation of the data
     */
    func getJSON() -> NSMutableDictionary {
        let dict = NSMutableDictionary()
        dict.setValue(destination, forKey: LocatinInfoStringNames.destination.rawValue)
        dict.setValue(beaconName, forKey: LocatinInfoStringNames.beaconName.rawValue)
        dict.setValue(nodeCount, forKey: LocatinInfoStringNames.nodeCount.rawValue)
        dict.setValue(nodes.getJSON(), forKey: LocatinInfoStringNames.nodes.rawValue)
        dict.setValue(UUID().uuidString, forKey: "uid")
        return dict
    }
}
