import Foundation
import UIKit

typealias Parameters = [String: Any]

enum ARCoreEndpoint {
    case uploadImage(data: Data, fileName: String)
    case getImage(fileName: String)
}

extension ARCoreEndpoint: APIRequest {
    var urlRequest: URLRequest {
        switch self {
        case .uploadImage(let data, let fileName):
            let urlComponents = NSURLComponents(string: APIMainEnvironmentConfig.baseURL + "/image/upload")!
            guard let url = urlComponents.url
            else {preconditionFailure("Invalid URL format")}
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            let accessToken = SessionManage.shared.accessToken ?? ""
            let boundary = UUID().uuidString
            
            let headers = [
                "Authorization": "Bearer \(accessToken)",
                "Content-Type": "multipart/form-data; boundary=\(boundary)",
                "Accept": "application/json"
            ]
            
            request.allHTTPHeaderFields = headers
            
            let mimeType = data.mimeType!
            request.httpBody = createHttpBody(binaryData: data, mimeType: mimeType, boundary: boundary)
            
            return request
        case .getImage(let fileName):
            let urlComponents = NSURLComponents(string: APIMainEnvironmentConfig.baseURL + "/image/get")!
            guard let url = urlComponents.url
            else {preconditionFailure("Invalid URL format")}
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            let accessToken = SessionManage.shared.accessToken ?? ""
            let headers = [
                //"content-type": "image/png",
                "Authorization": "Bearer \(accessToken)"
            ]
            request.allHTTPHeaderFields = headers
            return request
        }
    }
}

extension ARCoreEndpoint {
    private func createHttpBody(binaryData: Data, mimeType: String, boundary: String) -> Data {
        var postContent = "--\(boundary)\r\n"
        let fileName = "\(UUID().uuidString).jpeg"
        postContent += "Content-Disposition: form-data; name=\"\(fileName)\"; filename=\"\(fileName)\"\r\n"
        postContent += "Content-Type: \(mimeType)\r\n\r\n"
        
        var data = Data()
        guard let postData = postContent.data(using: .utf8) else { return data }
        data.append(postData)
        data.append(binaryData)
        
        var parameters: Parameters? {
            return [
                "number": 1
            ]
        }
        
        if let parameters = parameters {
            var content = ""
            parameters.forEach {
                content += "\r\n--\(boundary)\r\n"
                content += "Content-Disposition: form-data; name=\"\($0.key)\"\r\n\r\n"
                content += "\($0.value)"
            }
            if let postData = content.data(using: .utf8) { data.append(postData) }
        }
        
        guard let endData = "\r\n--\(boundary)--\r\n".data(using: .utf8) else { return data }
        data.append(endData)
        return data
    }
    
}

extension Data {
    var mimeType: String? {
        var values = [UInt8](repeating: 0, count: 1)
        copyBytes(to: &values, count: 1)
        
        switch values[0] {
        case 0xFF:
            return "image/jpeg"
        case 0x89:
            return "image/png"
        case 0x47:
            return "image/gif"
        case 0x49, 0x4D:
            return "image/tiff"
        default:
            return nil
        }
    }
}
