import ARKit
import RealityKit
import Firebase
import FirebaseStorage
import Foundation
import UIKit

class Droidbase {

    private static let userCvPath = "human-object-recognition"
    private let cvDbRef = Database.database().reference(withPath: userCvPath)
    private let storageRef = Storage.storage().reference()

    static let shared = Droidbase()

    func saveToCloud(_ arView: ARView, _ name:String, _ image: UIImage) {
        guard let data = image.pngData() else {
            fatalError("Can't save snaphot")
        }
        let metadata = StorageMetadata()
        metadata.contentType = "image/png"
        let timestamp = String(Date().timeIntervalSince1970)
        let mesh = serializedMap(sceneView: arView)

        let dataRef = storageRef.child(timestamp)
        dataRef.putData(data, metadata: metadata)
        cvDbRef.childByAutoId().setValue(["name": name, "snapshot": timestamp, "mesh": mesh])
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
}
