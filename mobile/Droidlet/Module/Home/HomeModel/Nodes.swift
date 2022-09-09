import Foundation

enum NodesStringNames: String {
//    case nodes = "nodes"
    case indexArray = "index"
}

//MARK: - Nodes

class Nodes: NSObject, NSCoding, Codable {
    let index: [Index]
    
    /*/ encode(with coder: NSCoder)
     Encodes the data with the given key
     */
    func encode(with coder: NSCoder) {
        coder.encode(self.index, forKey: NodesStringNames.indexArray.rawValue)
    }
    /*/ init?(coder decoder: NSCoder)
     On init, decodes the data with the given key
     */
    required init?(coder decoder: NSCoder) {
        self.index = decoder.decodeObject(forKey: NodesStringNames.indexArray.rawValue) as! Array<Index>
    }
    /*/ init(index: [Index])
     Inits the class with specific data
     
     @param - index - [Index] - an array of Index
     */
    init(index: [Index]){
        self.index = index
    }
    /*/ getJSON() -> NSMutableDictionary
     Constructs a JSON representation of the data within this class and
     returns the json representation
     
     @return: values - NSMutableDictionary - Json representation of the data
     */
    func getJSON() -> NSMutableDictionary {
        let values = NSMutableDictionary()

        var indexArray = Array<NSMutableDictionary>()
        for item in index {
            indexArray.append(item.getJSON())
        }
        values.setValue(indexArray, forKey: NodesStringNames.indexArray.rawValue)

        return values
    }
}

enum IndexStringNames: String {
    case type = "type"
    case xOffset = "x_offset"
    case yOffset = "y_offset"
    case zOffset = "z_offset"
}
enum NodeType: String{
    case start = "start"
    case intermediate = "intermediate"
    case destination = "destination"
}

//MARK: - Index

class Index: NSObject, NSCoding, Codable {
    
    let type: NodeType.RawValue
    let xOffset, yOffset, zOffset: Float

    enum CodingKeys: String, CodingKey {
        case type = "type"
        case xOffset = "x_offset"
        case yOffset = "y_offset"
        case zOffset = "z_offset"
    }
    
    /*/ encode(with coder: NSCoder)
     Encodes the data with the given key
     */
    func encode(with coder: NSCoder) {
        coder.encode(self.type, forKey: IndexStringNames.type.rawValue)
        coder.encode(self.xOffset, forKey: IndexStringNames.xOffset.rawValue)
        coder.encode(self.yOffset, forKey: IndexStringNames.yOffset.rawValue)
        coder.encode(self.zOffset, forKey: IndexStringNames.zOffset.rawValue)
    }
    /*/ init?(coder decoder: NSCoder)
     On init, decodes the data with the given key
     */
    required init?(coder decoder: NSCoder) {
        let temp = decoder.decodeObject(forKey: IndexStringNames.type.rawValue) as! String
        var retrievedType: String?
        switch temp{
            case NodeType.start.rawValue:
                retrievedType = NodeType.start.rawValue
            case NodeType.intermediate.rawValue:
                retrievedType = NodeType.intermediate.rawValue
            case NodeType.destination.rawValue:
                retrievedType = NodeType.destination.rawValue
            default:
                break
        }
        self.type = retrievedType!
        self.xOffset = decoder.decodeFloat(forKey: IndexStringNames.xOffset.rawValue)
        self.yOffset = decoder.decodeFloat(forKey: IndexStringNames.yOffset.rawValue)
        self.zOffset = decoder.decodeFloat(forKey: IndexStringNames.zOffset.rawValue)

    }
    /*/ init(destination: String, beaconName: String, nodeCount: Int, nodes: Nodes)
     Inits the class with specific data
     
     @param: type - String - string representation of the type
     @param: xOffset - Float - Float value of the xOffset from world origin
     @param: yOffset - Float - Float value of the yOffset from world origin
     @param: zOffset - Float - Float value of the zOffset from world origin
     */
    init(type: String, xOffset: Float, yOffset: Float, zOffset: Float) {
        self.type = type
        self.xOffset = xOffset
        self.yOffset = yOffset
        self.zOffset = zOffset
    }
    /*/ getJSON() -> NSMutableDictionary
     Constructs a JSON representation of the data within this class and
     returns the json representation
     
     @return: dict - NSMutableDictionary - Json representation of the data
     */
    func getJSON() -> NSMutableDictionary {
        let dict = NSMutableDictionary()
        dict.setValue(type, forKey: IndexStringNames.type.rawValue)
        dict.setValue(xOffset, forKey: IndexStringNames.xOffset.rawValue)
        dict.setValue(yOffset, forKey: IndexStringNames.yOffset.rawValue)
        dict.setValue(zOffset, forKey: IndexStringNames.zOffset.rawValue)
        return dict
    }
}

