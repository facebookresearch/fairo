import Foundation

enum HomeEndpoint {
    case markImage(value: NotificationAnswer, message_id: String, option: String?)
    case getImage(url: String)
    case uploadMap(param: NSMutableDictionary)
}

extension HomeEndpoint: APIRequest {
    var urlRequest: URLRequest {
        switch self {
        case .markImage(let value, let message_id, let option):
            let urlComponents = NSURLComponents(string: APIMainEnvironmentConfig.baseURL + APIMainEnvironmentConfig.Endpoint.answer.rawValue)!

            guard let url = urlComponents.url
                else {preconditionFailure("Invalid URL format")}
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            let accessToken = SessionManage.shared.accessToken!

            let headers = [
                        "Authorization": "Bearer \(accessToken)",
                        "Content-Type": "application/json"
                    ]
            request.allHTTPHeaderFields = headers
            
            var body = ["message_id": message_id,
                        "answer": value.rawValue] as [String : Any]
            
            body["option"] = option ?? ""
            
            do {
                let dataBody = try JSONSerialization.data(withJSONObject: body, options: [])
                request.httpBody = dataBody
            } catch let error {
                Logger.logDebug(error.localizedDescription)
            }
            
            return request
        case .getImage(let url):
            let urlComponents = NSURLComponents(string: url)!

            guard let url = urlComponents.url
                else {preconditionFailure("Invalid URL format")}
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            return request
        case .uploadMap(let param):
            let urlComponents = NSURLComponents(string: APIMainEnvironmentConfig.baseURL + APIMainEnvironmentConfig.Endpoint.map.rawValue)!
            guard let url = urlComponents.url
                else {preconditionFailure("Invalid URL format")}
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            let accessToken = SessionManage.shared.accessToken!
            let headers = [
                        "Authorization": "Bearer \(accessToken)",
                        "Content-Type": "application/json"
                    ]
            request.allHTTPHeaderFields = headers
            
            let uid = UUID().uuidString
            param.setValue(uid, forKey: "uid")

            do {
                let dataBody = try JSONSerialization.data(withJSONObject: ["maps": ["pi": param]], options: [])
                request.httpBody = dataBody
            } catch let error {
                Logger.logDebug(error.localizedDescription)
            }
            
            return request
        }
    }
}
