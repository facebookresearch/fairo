import Foundation
import FirebaseDatabase
import SocketIO
import ARKit
import RealityKit
import Starscream
import AVFoundation
import UIKit
import SwiftUI
import Alamofire

class HomeViewModel: ObservableObject {
    var settings: UserSettings = .shared

    private var pickerImage: UIImage?
    @Published var showPicker = false
    @Published var source: Picker.Source = .library
    
    @Published var listChat: [ChatModel] = []
    @Published var inputTextChat: String = ""
	@Published var navigationHintState: String = ""
    @Published var mapSavingState: SavingMapState = .selectImage
    @Published var agentState: AgentState = .none
    @Published var dashboardMode: DashboardMode = .ar
	@Published var arState: NavigationViewState = .idle

    var image: UIImage?
    var name: String = ""
    var uid: String?
    @Published var showEditPhotoVC = false

    var fileUrl: URL?

    @Published var depthImg: UIImage? = nil
    @Published var showEditVideoVC = false
	@Published var selectedLocation: LocationInfo? = nil
    @Published var showManageMap: Bool = false
    @Published var showImagePreview: Bool = false
    @Published var showNavigationMap: Bool = false
    @Published var isODORunning: Bool = false
    @Published var showingUploadAlert: Bool = false
    @Published var featurePoints: String = ""
    @Published var trackingStatus: String = ""
    var isSocketConnected: Bool {
        get {
            if let socketManager = manager {
                return (socketManager.defaultSocket.status == .connected)
            } else {
                return false
            }
        }
    }

    var didSelectNavigating = false
    
    // MARK: - Attribute
    private let service = CameraService()
    var session: AVCaptureSession
        
    var manager: SocketManager!
    var socket: SocketIOClient!
    var iosView: DroidletARView!


	enum NavigationViewState: Equatable {
		case idle
		case findingMarker
		case start
		case waypoint
		case destination
		case navigation
	}
    
    enum SavingMapState: String {
        case selectImage = "Take a photo of the destination"
        case saveAndUpload = "Save and upload"
    }
	
    init() {
        session = service.session
        service.delegate = self
		ARSceneViewDelegate.ARSCNViewDelegateInstance.delegate = self
    }
    
    func startARSection() {
        guard let configuration = iosView.sceneView.session.configuration as? ARWorldTrackingConfiguration else {
            return
        }

        iosView.sceneView.session.run(configuration)
    }
    
    func showPhotoPicker() {
        if source == .camera {
            if !Picker.checkPermissions() {
                print("There is no camera on this device")
                return
            }
        }
        showPicker = true
    }
    
    func setSelectedLocationByUID() {
        guard let theUID = self.uid,
        self.listChat.count > 0 else {
            return
        }
        let filteredArray = listChat.filter { $0.attachment.filter { $0.map?.uid == theUID}.count != 0}
        if let imageObject = filteredArray.first,
           let attachment = imageObject.attachment.first,
           let map = attachment.map {
            if let destination = map.destination,
               let beacon = map.beaconName,
               let count = map.nodeCount,
               let nodes = map.nodes {
                let location: LocationInfo = LocationInfo.init(destination: destination, beaconName: beacon, nodeCount: count, nodes: nodes)
                self.selectedLocation = location
            }
        }
    }

    func pauseARSection() {
        iosView.sceneView.session.pause()
    }
    
    func logout() {
        SessionManage.shared.logout()
        settings.expride = false
    }
    
    // MARK: - Camera
    func startCamera() {
        service.checkForPermissions()
        service.configure()
    }
    
    func stopCamera() {
        service.stop(completion: {
            
        })
    }
    
    func capturePhoto() {
        service.capturePhoto()
    }
    
    // MARK: - Firebase
//    func upload(image: UIImage, name: String, url: URL?) {
//        switch dashboardMode {
//        case .camera:
////            guard let data = image.jpegData(compressionQuality: 0.8) else { return }
////            let timestamp = String(Date().timeIntervalSince1970)
////            ARCoreService.shared.uploadImage(data: data, fileName: timestamp)
//            Droidbase.shared.saveToCloud(nil, name, image)
//        case .ar:
//            Droidbase.shared.save3DToCloud(nil, name, url: url)
//        default:
//            break
//        }
//    }
    
    func getObjectAPI() {
        Droidbase.shared.getObjectAPI(completion: { image in
            //self.image = image
            if let cgImage = image.cgImage {
                let arImage = ARReferenceImage(cgImage, orientation: .up, physicalWidth: 0.072)
                self.iosView.markerImage(arImage)
            }
        })
    }
    
    func downloadImageAPI(_ name: String) {
        Droidbase.shared.downloadImageAPI(name, completion: nil)
    }
    
    // MARK: - Socket
    func connectSocket() {
        manager = SocketManager(socketURL: URL(string: APIMainEnvironmentConfig.api)!, config: [.log(false), .compress])
        socket = manager.defaultSocket

        socket.on(clientEvent: .connect) {data, ack in
            print("socket connected")
        }
        
        socket.on("showAssistantReply") { data, ack in
            if let items = data as? [[String: String]], !items.isEmpty {
                self.agentState = .none
                let agent_reply = items.first!["agent_reply"] ?? ""
                
                if agent_reply.hasPrefix("Agent: ") {
                    self.decodeAssistantReply(agent_reply)
                }
            }
        }
        
        socket.on("updateAgentType") { data, ask in
            if let items = data as? [[String: String]], !items.isEmpty, items.first!["agent_type"] == "craftassist" {
                self.agentState = .thingking
            }
        }
        
        socket.on("depth") { value, ask in
//            Logger.logDebug("recipe depthImg")
            guard let value = value as? [NSDictionary] else { return }
            guard let base64String = value.first?["depthImg"] as? String else { return }
            self.depthImg = self.decodeImage(base64String)
        }
        
        socket.on("rgb") { value, ask in
//            Logger.logDebug("recipe rgb data")
            guard let value = value as? [String] else { return }
            self.depthImg = self.decodeImage(value.first ?? "")
        }
        
        socket.connect()
    }
    
    func emit(_ message: String) {
        agentState = .send
        socket.emit("sendCommandToAgent", message) {
            Logger.logDebug("Emit success")
        }
    }
    
    private func decodeImage(_ base64String: String) -> UIImage? {
        guard let data = Data(base64Encoded: base64String) else { return nil }
        let image = UIImage(data: data)
        return image
    }
    
    private func decodeAssistantReply(_ agent_reply: String) {
        let range = NSRange(location: 0, length: "Agent: ".count)
        let reply = (agent_reply as? NSString)?.replacingCharacters(in: range, with: "")
                
        guard let data = reply?.data(using: .utf8) else { return }
        
        do {
            let json = try JSONSerialization.jsonObject(with: data, options: .allowFragments)
            if let json = json as? [String : Any] {
                decodeAssistantReplyImage(json)
            }
        } catch {
            Logger.logDebug(error.localizedDescription)
            decodeAssistantReplyText(agent_reply)
        }
    }
    
    private func decodeAssistantReplyText(_ agent_reply: String) {
        let model = ChatModel(text: agent_reply, isUserInput: false)
        self.listChat.append(model)
    }
    
    private func decodeAssistantReplyImage(_ agent_reply: [String : Any]) {
        do {
            let data = try JSONSerialization.data(withJSONObject: agent_reply, options: .prettyPrinted)
            let agentImageModel = try JSONDecoder().decode(AgentImageModel.self, from: data)
            if let images = agentImageModel.images {
                let model = ChatModel(text: agentImageModel.message ?? "", isUserInput: false, attachment: images)
                self.listChat.append(model)
            }
        } catch {
            Logger.logDebug(error.localizedDescription)
        }
    }
}

extension HomeViewModel: ARScnViewDelegate {
	func toggleScanToNavigate(shouldShow: Bool) {
		self.arState = .navigation
		self.navigationHintState = "Destination: \(self.selectedLocation?.destination ?? "??")"
	}
	
	func destinationReached() {
	}
	
	func toggleUndoButton(shouldShow: Bool) {
	}
	
	func toggleEndButton(shouldShow: Bool) {
	}
	
	func toggleSaveButton(shouldShow: Bool) {
	}
	
	func toggleAddButton(shouldShow: Bool) {
		if shouldShow, DataModel.dataModelSharedInstance.getLocationDetails().getIsCreatingCustomMap() {
			if !(DataModel.dataModelSharedInstance.getNodeManager().getStartingNodeIsSet()) {
				self.arState = .start
				self.navigationHintState = "Add the start location"
			}
		}
	}
}

// MARK: - Odometry fucntions
extension HomeViewModel {
    func toggleOdometry() {
        self.iosView.toggleRecording()
        self.iosView.isRecording = self.isODORunning
    }
    
    func uploadODORecord() {
        self.iosView.uploadRecordDataFile()
    }
}

extension HomeViewModel {
    func markImage(value: NotificationAnswer, message_id: String, option: String?) {
        HomeService.shared.markImage(value: value, message_id: message_id, option: option)
    }
    
    func downloadImage(url: String) {
        HomeService.shared.getImage(url: url) { image in
            DispatchQueue.main.async {
                self.settings.notificationObject.setImage(image)
            }
        }
    }
    
    func sendAnchor() {
        settings.request(.ar)
    }
    
    func locationImageAction() {
        showImagePreview = true
    }
    
    // MARK: - Navigation Map
    func markerImage(_ image: CVPixelBuffer, orientation: CGImagePropertyOrientation) {
        let arImage = ARReferenceImage(image, orientation: orientation, physicalWidth: 0.072)
        self.iosView.markerImage(arImage)

    }
    
    func resetToNormalState() {
        iosView.sceneView.session.pause()
        setUpAndRunConfiguration()
    }
    
    private func setUpAndRunConfiguration() {
		if AVCaptureDevice.authorizationStatus(for: .video) ==  .authorized {
			if ARWorldTrackingConfiguration.isSupported {
				let configuration = ARWorldTrackingConfiguration()
				configuration.worldAlignment = .gravityAndHeading
				
				//Reference marker images
				if let ARImages = ARReferenceImage.referenceImages(inGroupNamed: "ARResources", bundle: Bundle.main) {
					configuration.detectionImages = ARImages
				} else {
					print("Images could not be loaded")
				}
				let ARSCNViewDelegateSharedInstance = ARSceneViewDelegate.ARSCNViewDelegateInstance
				self.iosView.sceneView.delegate = ARSCNViewDelegateSharedInstance

				//Cleans the ARScene
				iosView.sceneView.scene.rootNode.enumerateChildNodes{ (node, stop) in
					node.removeFromParentNode()
				}
				iosView.sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
			}
		}
    }

    func handleAddButton() {
        if (DataModel.dataModelSharedInstance.getLocationDetails().getIsCreatingCustomMap()) {
            ARSceneViewDelegate.ARSCNViewDelegateInstance.handleAddButton()
			if !(DataModel.dataModelSharedInstance.getNodeManager().getStartingNodeIsSet()) {
				self.arState = .start
				self.navigationHintState = "Add the start location"
			} else {
				self.arState = .waypoint
				self.navigationHintState = "Add waypoint or destination to finish"
			}
        }
    }
    
	func handleEndButton() {
		ARSceneViewDelegate.ARSCNViewDelegateInstance.handleEndButton()
		self.arState = .destination
		self.navigationHintState = "Save the navigation route"
	}
	
    func saveMapName(with image: UIImage) {
        self.pickerImage = image
        self.mapSavingState = .saveAndUpload
    }
    
    func saveDataStore() {
        let alert = settings.alert(message: nil, inputText: nil) { name in
            let dataModelSharedInstance = DataModel.dataModelSharedInstance
            let nm = dataModelSharedInstance.getNodeManager()
            let customMap = nm.retrieveCustomMap(name)
            
            // Upload the customMap
            self.uploadMapInfo(location: customMap)

            // Saves the customMap, type=LocationInfo, to the dataStore.
            let dataStore = dataModelSharedInstance.getDataStoreManager().dataStore
            var customMapList = dataStore.getLocationInfoList()
            customMapList.append(customMap)
            dataStore.setLocationInfoList(list: customMapList)
            dataModelSharedInstance.getDataStoreManager().saveDataStore()

            //Resets to normal state
            dataModelSharedInstance.resetNavigationToRestingState()
			self.arState = .idle
			self.navigationHintState = ""
            self.pickerImage = nil
            self.mapSavingState = .selectImage
			DataModel.dataModelSharedInstance.resetNavigationToRestingState()
            self.resetToNormalState()
        }
        
        if let delegate = UIApplication.shared.delegate as? AppDelegate,
           let parentViewController = delegate.window?.rootViewController {
            parentViewController.present(alert, animated: true)
        }
        
    }
    
	func cancelNavigation() {
		//Resets to normal state
		let dataModelSharedInstance = DataModel.dataModelSharedInstance
		self.arState = .idle
		self.navigationHintState = ""
		dataModelSharedInstance.resetNavigationToRestingState()
		dataModelSharedInstance.getLocationDetails().setIsCreatingCustomMap(isCreatingCustomMap: false)
		dataModelSharedInstance.getLocationDetails().setIsUploadingMap(isSet: false)
		self.resetToNormalState()
	}
	
    func uploadMap() {
        //Push to Firebase
        let dataModelSharedInstance = DataModel.dataModelSharedInstance
        guard let locInfo = dataModelSharedInstance.getDataStoreManager().dataStore.getLocationInfoList().last else { return }
        let json = locInfo.getJSON()
        
        HomeService.shared.uploadMap(json)
    }
    
    func uploadMapInfo(location: LocationInfo) {
        let uploadURL = "http://34.145.124.241:5000/image/upload"
        let headers: HTTPHeaders = [
            "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTY1NTQ2NDAwMiwianRpIjoiOTBlY2M5OTgtM2NkNC00YzYwLWE4NzgtMjcxM2JkZGY2YzkyIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6IntcImlkXCI6IFwielpObXFuWjdic2M0dE9ITWJWaGRcIiwgXCJ1c2VybmFtZVwiOiBcInZtb1wiLCBcInBhc3N3b3JkXCI6IFwiJDJiJDEyJGZ3U3NzbkJ6U0VnWDk2RkF5cDdIdGVsZUU5cEtmNzRQVlZpNmNvQ3FVaG5OR1pTMFdFNVhTXCIsIFwidXNlcl91aWRcIjogXCJcIn0iLCJuYmYiOjE2NTU0NjQwMDIsImV4cCI6MTY1ODA1NjAwMn0.N6DoY8YqrMFKhD_r0gHtP3Jq42Ao9W5mxpOopLtsCJg",
            "Content-type": "multipart/form-data",
            "Accept": "*/*",
            "Accept-Encoding" : "gzip, deflate, br"
        ]
        
        let locationJSON = location.getJSON()
        let mapJSON  = ["map": locationJSON]
        
        AF.upload(
            multipartFormData: { multipartFormData in
                if let selectedImage = self.pickerImage,
                   let imageData = selectedImage.jpegData(compressionQuality: 1),
                   let jsonData = try? JSONSerialization.data(withJSONObject: mapJSON, options: .prettyPrinted) {
                    multipartFormData.append(imageData, withName: "file", fileName: "file.jpg", mimeType: "image/jpeg")
                    multipartFormData.append(jsonData, withName: "map")
                }
            },
            to: uploadURL, method: .post , headers: headers)
        .response { res in
            if let data = res.data {
                let response = try! JSONDecoder().decode(APIGeneralResponse.self, from: data)
                print("======= UPLOAD RESULT =====\n\(response)")
            }
        }
    }
}

func decodeJSONDestination(JSONdata: Data) -> LocationInfo? {
    let data = try? JSONDecoder().decode(LocationInfo.self, from: JSONdata)
    return data
}
