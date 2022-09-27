import Foundation
import Combine
import UIKit
import SwiftUI

class HomeService {
    static let shared = HomeService()

    func markImage(value: NotificationAnswer, message_id: String, option: String?) {
        guard let _ = HomeEndpoint.markImage(value: value, message_id: message_id, option: option).urlRequest.url else {
            return
        }
        
        let urlRequest = HomeEndpoint.markImage(value: value, message_id: message_id, option: option).urlRequest
        let dataTask = URLSession.shared.dataTask(with: urlRequest) { (data, response, error) in
            if let error = error {
                Logger.logDebug(error.localizedDescription)
                return
            }
            
            guard let response = response as? HTTPURLResponse else { return }
            
            if response.statusCode == 200 {
                guard let data = data else { return }
                Logger.logDebug(String(data: data, encoding: .utf8) ?? "")
            }
        }
        
        dataTask.resume()
    }
    
    func getImage(url: String, completion: @escaping (UIImage?)->()) {
        guard let _ = HomeEndpoint.getImage(url: url).urlRequest.url else { return }
        let dataTask = URLSession.shared.dataTask(with: HomeEndpoint.getImage(url: url).urlRequest) { (data, response, error) in
            if let error = error {
                Logger.logDebug(error.localizedDescription)
                return
            }
            
            guard let response = response as? HTTPURLResponse else { return }
            
            if response.statusCode == 200 {
                guard let data = data else { return }
                completion(UIImage(data: data))
            }
        }
        
        dataTask.resume()
    }
    
    func uploadMap(_ param: NSMutableDictionary) {
        guard let _ = HomeEndpoint.uploadMap(param: param).urlRequest.url else { return }
        let dataTask = URLSession.shared.dataTask(with: HomeEndpoint.uploadMap(param: param).urlRequest) { (data, response, error) in
            if let error = error {
                Logger.logDebug(error.localizedDescription)
                return
            }
            
            guard let response = response as? HTTPURLResponse else { return }
            
            if response.statusCode == 200 {
//                guard let data = data else { return }
            }
        }
        
        dataTask.resume()
    }
    
    func parseMapData(_ data: Data, completion: @escaping (Data)->()) {
        do {
            let json = try JSONSerialization.jsonObject(with: data, options: [])
            
            if let _json = json as? [Any] {
                parseArray(_json, completion: completion)
            } else if let _json = json as? [String : Any] {
                parseDictionary(_json, completion: completion)
            }
        } catch {
            Logger.logDebug(error.localizedDescription)
        }
    }
    
    func parseArray(_ array: [Any], completion: @escaping (Data)->()) {
        guard let first = array.first as? [String : Any] else {
            return
        }
        parseDictionary(first, completion: completion)
    }
    
    func parseDictionary(_ dictionary: [String : Any], completion: @escaping (Data)->()) {
        guard let locationInfo = (dictionary["maps"] as? [String : Any])?["pi"] else { return }
        
        do {
            let locationInfoData = try JSONSerialization.data(withJSONObject: locationInfo, options: [])
            completion(locationInfoData)
        } catch {
            Logger.logDebug(error.localizedDescription)
        }
    }
}
