import Foundation

struct ImageChatModel: Codable {
    var map: MapInfoModel?
    var url: String?
    var id: String?
}

struct MapInfoModel: Codable {
    var uid: String?
    var nodes: Nodes?
    var nodeCount: Int?
    var beaconName: String?
    var destination: String?

    private enum CodingKeys : String, CodingKey {
        case uid, nodes, nodeCount = "node_count", beaconName = "beacon_name", destination
    }
}

struct AgentImageModel: Codable {
    var images: [ImageChatModel]?
    var message: String?
}
