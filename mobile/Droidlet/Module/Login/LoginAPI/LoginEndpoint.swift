import Foundation

enum LoginEndpoint {
    case login(token: String)
}

extension LoginEndpoint: APIRequest {
    var urlRequest: URLRequest {
        switch self {
        case .login(let token):
            let urlComponents = NSURLComponents(string: APIMainEnvironmentConfig.baseURL + "/firebase_user")!
            guard let url = urlComponents.url
                else {preconditionFailure("Invalid URL format")}
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            let headers = [
                        "content-type": "application/json"
                    ]
            request.allHTTPHeaderFields = headers
            
            let json = ["id_token" : token]
            let jsonData = try? JSONSerialization.data(withJSONObject: json)
            request.httpBody = jsonData

            return request
        }
    }
}
