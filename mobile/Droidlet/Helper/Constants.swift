import Foundation
import UIKit

struct Constants {
    static let FireBaseDetectObjectKey = "detect_object"
    static let FireBaseObjectName = "name"
    static let FireBaseObjectTransform = "transform"
}

struct ArkitNodeDimension {
    static let arrowNodeXOffset = CGFloat(0.1)
}

struct APIMainEnvironmentConfig {
    static let api = "http://34.145.124.241:8000"
    static let baseURL = "http://34.145.124.241:5000"
    
    enum Endpoint: String {
        case update_device_token = "/firebase/update_device_token"
        case answer = "/firebase/answer"
        case map = "/firebase/map"
    }
}

enum AgentState: Int {
    case none = 0
    case send
    case thingking
    
    func getText() -> (Bool, String) {
        switch self {
        case .none:
            return (false, "")
        case .send:
            return (true, "Sending command...")
        case .thingking:
            return (true, "Appthinking...")
        }
    }
}

enum DashboardMode: String {
    case camera = "Camera"
    case ar = "AR"
    case video = "Video"
}
