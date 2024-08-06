import Foundation
import Combine

class ARCoreService {
    static let shared = ARCoreService()
    
    func uploadImage(data: Data, fileName: String) {
        guard let url = ARCoreEndpoint.uploadImage(data: data, fileName: fileName).urlRequest.url else {
            return
        }
        
        let urlRequest = ARCoreEndpoint.uploadImage(data: data, fileName: fileName).urlRequest
        
        let dataTask = URLSession.shared.dataTask(with: urlRequest) { data, response, error in
            
            if let error = error {
                Logger.logDebug(error.localizedDescription)
                return
            }
            
            guard let response = response as? HTTPURLResponse else { return }
            
            if response.statusCode == 200 {
                guard let data = data else { return }
            }
        }
        
        dataTask.resume()
    }
}
