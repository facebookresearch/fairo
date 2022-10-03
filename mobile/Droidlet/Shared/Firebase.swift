import ARKit
import RealityKit
import Firebase
import FirebaseStorage
import FirebaseAuth
import Foundation
import UIKit

class Droidbase {
    
    static let FireBaseIO = "https://yss21031-default-rtdb.firebaseio.com/" + userCvPath + ".json"
    private static let userCvPath = "human-object-recognition"
    private let cvDbRef = Database.database().reference(withPath: userCvPath)
    private let cvDbRef3D = Database.database().reference(withPath: "human-object-recognition3D")
    private let storageRef = Storage.storage().reference()
    
    static let shared = Droidbase()
    
    // Mark: - Test
    private let cvMap = Database.database().reference(withPath: "map")
    func saveMap(_ json: NSDictionary, name: String) {
        cvMap.child(name).setValue(json)
    }
    
    func getMap(_ child: String, completion: @escaping (Any?) -> ()) {
        cvMap.child("-N-vruawa3BCOQzG4pbN").getData { error, snapshot in
            if let error = error {
                Logger.logDebug(error.localizedDescription)
                return
            }
            
            completion(snapshot.value)
        }
    }
    
    //
    func saveToCloud(_ arView: ARView?, _ name:String, _ image: UIImage) {
        guard let data = image.jpegData(compressionQuality: 0.6) else {
            fatalError("Can't save snaphot")
        }
        let metadata = StorageMetadata()
        metadata.contentType = "image/jpg"
        let timestamp = String(Date().timeIntervalSince1970)
        //let mesh = serializedMap(sceneView: arView)
        
//        let dataRef = storageRef.child("images/\(timestamp).jpg")
//        dataRef.putData(data, metadata: metadata)
        cvDbRef.childByAutoId().setValue(["name": name, "snapshot": timestamp, "mesh": "mesh"])
    }
    
    func save3DToCloud(_ arView: ARView?, _ name:String, url: URL?) {
        guard let url = url else {
            return
        }
        
        do {
            let data = try Data(contentsOf: url)
            let metadata = StorageMetadata()
            metadata.contentType = "image/obj"
            let timestamp = String(Date().timeIntervalSince1970)
            
            let dataRef = storageRef.child("images3d/\(timestamp).obj")
            dataRef.putData(data, metadata: metadata)
            cvDbRef3D.childByAutoId().setValue(["name": name, "snapshot": timestamp, "mesh": "mesh"])
        } catch {
            
        }
    }
    
    func serializedMap(sceneView: ARView) -> String {
        let worldMap = sceneView.session.getCurrentWorldMap
        do {
            let data = try NSKeyedArchiver.archivedData(withRootObject: worldMap, requiringSecureCoding: true)
            return data.base64EncodedString()
        } catch {
            fatalError("Can't save map: \(error.localizedDescription)")
        }
    }
    
    func getObjectAPI(completion: ((UIImage) -> ())?) {
        let semaphore = DispatchSemaphore (value: 0)
        
        var request = URLRequest(url: URL(string: Droidbase.FireBaseIO)!,timeoutInterval: Double.infinity)
        request.httpMethod = "GET"
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            guard let data = data else {
                print(String(describing: error))
                semaphore.signal()
                return
            }
            
            do {
                guard let dicts = try JSONSerialization.jsonObject(with: data, options: []) as? [String : Any],
                      let dict = dicts.reversed().first?.value as? [String: Any], let snapshot = dict["snapshot"] as? String else {
                          return
                      }
                self.downloadImageAPI(snapshot, completion: completion)
            } catch {
                Logger.logDebug(String(describing: error.localizedDescription))
            }
            semaphore.signal()
        }
        
        task.resume()
        semaphore.wait()
    }
    
    func downloadImageAPI(_ name: String, completion: ((UIImage) -> ())?) {
        getImageURL(name: name) { url in
            let semaphore = DispatchSemaphore (value: 0)
            
            var request = URLRequest(url: url,timeoutInterval: Double.infinity)
            request.httpMethod = "GET"
            
            let task = URLSession.shared.dataTask(with: request) { data, response, error in
                guard let data = data else {
                    print(String(describing: error))
                    semaphore.signal()
                    return
                }
                
                if let image = UIImage(data: data) {
                    completion?(image)
                }
                semaphore.signal()
            }
            
            task.resume()
            semaphore.wait()
        }
    }
    
    func getImageURL(name: String, completion: ((URL) -> ())?) {
        let starsRef = storageRef.child("images/\(name).jpg")
        
        // Fetch the download URL
        starsRef.downloadURL { url, error in
            guard let url = url else { return }
            completion?(url)
        }
    }
}

class FirebaseAuthManager {
    func login(credential: AuthCredential, completionBlock: @escaping (_ success: Bool, _ token: String?) -> Void) {
        Auth.auth().signIn(with: credential, completion: { (firebaseUser, error) in
            firebaseUser?.user.getIDTokenResult(forcingRefresh: true, completion: { idToken, error in
                completionBlock(error == nil, idToken?.token)
            })
        })
    }
}
