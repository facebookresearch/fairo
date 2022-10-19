import UIKit
import Foundation
import ARKit
import Combine
import ARCoreCloudAnchors
import SwiftUI
import os.log
import Accelerate
import Alamofire

enum ARState: Int {
    case defaultState
    case creatingRoom
    case roomCreated
    case hosting
    case hostingFinished
    case enterRoomCode
    case resolving
    case resolvingFinished
}

protocol IosARViewDelegate: AnyObject {
    func tapGesture(_ frame: ARFrame, raycastResult: ARRaycastResult)
    func trackingInfoDidUpdate(points: String, status: String)
}

class DroidletARView: UIView, UIGestureRecognizerDelegate, ARSessionDelegate {
	var sceneView: ARSCNView
	var coachingView: ARCoachingOverlayView
    var showPlanes = false
    var customPlaneTexturePath: String? = nil
    private var trackedPlanes = [UUID: (SCNNode, SCNNode)]()
    
    var cancellableCollection = Set<AnyCancellable>() //Used to store all cancellables in (needed for working with Futures)
    var anchorCollection = [String: ARAnchor]() //Used to bookkeep all anchors created by Flutter calls
    
    var arcoreSession: GARSession? = nil
    private var arcoreMode: Bool = true
    private var configuration: ARWorldTrackingConfiguration!
    private var tappedPlaneAnchorAlignment = ARPlaneAnchor.Alignment.horizontal // default alignment
    
    private var panStartLocation: CGPoint?
    private var panCurrentLocation: CGPoint?
    private var panCurrentVelocity: CGPoint?
    private var panCurrentTranslation: CGPoint?
    private var rotationStartLocation: CGPoint?
    private var rotation: CGFloat?
    private var rotationVelocity: CGFloat?
    private var panningNode: SCNNode?
    private var panningNodeCurrentWorldLocation: SCNVector3?
    
    weak var delegate: IosARViewDelegate?
    private var state = ARState(rawValue: 0)
    private var arAnchor: ARAnchor?
    private var garAnchor: GARAnchor?
    let customQueue: DispatchQueue = DispatchQueue(label: "droilet.ar.ys")
    private var accumulatedPointCloud = AccumulatedPointCloud()

    // constants for collecting data
    var isRecording: Bool = false
    private let numTextFiles = 2
    private let ARKIT_CAMERA_POSE = 0
    private let ARKIT_POINT_CLOUD = 1
    private let mulSecondToNanoSecond: Double = 1000000000
    private var previousTimestamp: Double = 0
    
    // text file input & output
    var fileHandlers = [FileHandle]()
    var fileURLs = [URL]()
    var fileNames: [String] = ["camera_pose.txt", "point_cloud.txt"]
    
    var settings: UserSettings = .shared
    
	init(config: Dictionary<String,Any>? = nil) {
		self.sceneView = ARSCNView(frame: CGRect.zero)
		self.coachingView = ARCoachingOverlayView(frame: CGRect.zero)
        super.init(frame: CGRect.zero)
		if let arConfig = config {
			initializeARView(arguments: arConfig)
		} else {
			initializeARView(arguments: ["planeDetectionConfig": 1,
										 "showPlanes": true,
										 "showFeaturePoints": true,
										 "showWorldOrigin": false,
										 "handleTaps": true])
		}
    }
    
    required init?(coder: NSCoder) {
        self.sceneView = ARSCNView(frame: CGRect.zero)
        self.coachingView = ARCoachingOverlayView(frame: CGRect.zero)
        super.init(coder: coder)
    }
    
    func markerImage(_ image: ARReferenceImage) {
        //Reference marker images
        
		let _image = image
        _image.name = "marker-2"
        let ARImages = Set([_image])
        configuration.detectionImages = ARImages
        sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }
    
    func initializeARView(arguments: Dictionary<String,Any>){


        // Set plane detection configuration
        self.configuration = ARWorldTrackingConfiguration()
        if let planeDetectionConfig = arguments["planeDetectionConfig"] as? Int {
            switch planeDetectionConfig {
            case 1:
                configuration.planeDetection = .horizontal
                
            case 2:
                if #available(iOS 11.3, *) {
                    configuration.planeDetection = .vertical
                }
            case 3:
                if #available(iOS 11.3, *) {
                    configuration.planeDetection = [.horizontal, .vertical]
                }
            default:
                configuration.planeDetection = []
            }
        }
        
        // Set plane rendering options
        if let configShowPlanes = arguments["showPlanes"] as? Bool {
            showPlanes = configShowPlanes
            if (showPlanes){
                // Visualize currently tracked planes
                for plane in trackedPlanes.values {
                    plane.0.addChildNode(plane.1)
                }
            } else {
                // Remove currently visualized planes
                for plane in trackedPlanes.values {
                    plane.1.removeFromParentNode()
                }
            }
        }
        if let configCustomPlaneTexturePath = arguments["customPlaneTexturePath"] as? String {
            customPlaneTexturePath = configCustomPlaneTexturePath
        }
        
        // Set debug options
        var debugOptions = ARSCNDebugOptions().rawValue
        if let showFeaturePoints = arguments["showFeaturePoints"] as? Bool {
            if (showFeaturePoints) {
                debugOptions |= ARSCNDebugOptions.showFeaturePoints.rawValue
            }
        }
        if let showWorldOrigin = arguments["showWorldOrigin"] as? Bool {
            if (showWorldOrigin) {
                debugOptions |= ARSCNDebugOptions.showWorldOrigin.rawValue
            }
        }
        self.sceneView.debugOptions = ARSCNDebugOptions(rawValue: debugOptions)
        
        if let configHandleTaps = arguments["handleTaps"] as? Bool {
            if (configHandleTaps){
                let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
                tapGestureRecognizer.delegate = self
                self.sceneView.gestureRecognizers?.append(tapGestureRecognizer)
            }
        }
        
        // Add coaching view
        if let configShowAnimatedGuide = arguments["showAnimatedGuide"] as? Bool {
            if configShowAnimatedGuide {
                if self.sceneView.superview != nil && self.coachingView.superview == nil {
                    self.sceneView.addSubview(self.coachingView)
                    self.coachingView.autoresizingMask = [
                        .flexibleWidth, .flexibleHeight
                    ]
                    self.coachingView.session = self.sceneView.session
                    self.coachingView.activatesAutomatically = true
                    if configuration.planeDetection == .horizontal {
                        self.coachingView.goal = .horizontalPlane
                    }else{
                        self.coachingView.goal = .verticalPlane
                    }
                }
            }
        }
        
        self.addSubview(sceneView)
        sceneView.translatesAutoresizingMaskIntoConstraints = false
        sceneView.topAnchor.constraint(equalTo: self.topAnchor).isActive = true
        sceneView.bottomAnchor.constraint(equalTo: self.bottomAnchor).isActive = true
        sceneView.leadingAnchor.constraint(equalTo: self.leadingAnchor).isActive = true
        sceneView.trailingAnchor.constraint(equalTo: self.trailingAnchor).isActive = true
        
        let ARSCNViewDelegateSharedInstance = ARSceneViewDelegate.ARSCNViewDelegateInstance
        self.sceneView.delegate = ARSCNViewDelegateSharedInstance
        sceneView.session.delegate = self

        if let ARImages = ARReferenceImage.referenceImages(inGroupNamed: "ARResources", bundle: Bundle.main) {
            configuration.detectionImages = ARImages
        } else {
            print("Images could not be loaded")
        }
        
        sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        
        // Test
        let locationDetailsSharedInstance = LocationDetails.LocationDetailsSharedInstance
        let locationManagerSharedInstance = LocationManager.locationManagerInstance
        let nodeManagerSharedInstance = NodeManager.nodeManagerSharedInstance
        let dataStoreManagerSharedInstance = DataStoreManager.dataStoreManagerSharedInstance
        
        let dataModelSharedInstance = DataModel.dataModelSharedInstance
        dataModelSharedInstance.setSceneView(view: sceneView)
        dataModelSharedInstance.setLocationDetails(locationDetails: locationDetailsSharedInstance)
        dataModelSharedInstance.setLocationManager(locationManager: locationManagerSharedInstance)
        dataModelSharedInstance.setARSCNViewDelegate(ARSCNViewDelegate: ARSCNViewDelegateSharedInstance)
        dataModelSharedInstance.setNodeManager(nodeManager: nodeManagerSharedInstance)
        dataModelSharedInstance.setDataStoreManager(dataStoreManager: dataStoreManagerSharedInstance)
        
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        
        // obtain current transformation 4x4 matrix
        let timestamp = frame.timestamp * self.mulSecondToNanoSecond
        previousTimestamp = timestamp
        let ARKitTrackingState = frame.camera.trackingState

        let imageFrame = frame.capturedImage
        let imageResolution = frame.camera.imageResolution
        
        let T_gc = frame.camera.transform
        
        let r_11 = T_gc.columns.0.x
        let r_12 = T_gc.columns.1.x
        let r_13 = T_gc.columns.2.x
        
        let r_21 = T_gc.columns.0.y
        let r_22 = T_gc.columns.1.y
        let r_23 = T_gc.columns.2.y
        
        let r_31 = T_gc.columns.0.z
        let r_32 = T_gc.columns.1.z
        let r_33 = T_gc.columns.2.z
        
        let t_x = T_gc.columns.3.x
        let t_y = T_gc.columns.3.y
        let t_z = T_gc.columns.3.z
        
        // dispatch queue to display UI
        if let del = self.delegate {
            let points = String(format:"%05d", self.accumulatedPointCloud.count)
            let status = "\(ARKitTrackingState)"
            del.trackingInfoDidUpdate(points: points, status: status)
        }
        
        // custom queue to save ARKit processing data
        self.customQueue.async {
            if ((self.fileHandlers.count == self.numTextFiles) && self.isRecording) {
                
                // 1) record ARKit 6-DoF camera pose
                let ARKitPoseData = String(format: "%.0f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f \n",
                                           timestamp,
                                           r_11, r_12, r_13, t_x,
                                           r_21, r_22, r_23, t_y,
                                           r_31, r_32, r_33, t_z)
                if let ARKitPoseDataToWrite = ARKitPoseData.data(using: .utf8) {
                    self.fileHandlers[self.ARKIT_CAMERA_POSE].write(ARKitPoseDataToWrite)
                } else {
                    os_log("Failed to write data record", log: OSLog.default, type: .fault)
                }
                
                // 2) record ARKit 3D point cloud only for visualization
                if let rawFeaturePointsArray = frame.rawFeaturePoints {
                    
                    // constants for feature points
                    let points = rawFeaturePointsArray.points
                    let identifiers = rawFeaturePointsArray.identifiers
                    let pointsCount = points.count
                    
                    let kDownscaleFactor: CGFloat = 4.0
                    let scale = Double(1 / kDownscaleFactor)
                    
                    var projectedPoints = [CGPoint]()
                    var validPoints = [vector_float3]()
                    var validIdentifiers = [UInt64]()
                    
                    
                    // project all feature points into image 2D coordinate
                    for i in 0...(pointsCount - 1) {
                        let projectedPoint = frame.camera.projectPoint(points[i], orientation: .landscapeRight, viewportSize: imageResolution)
                        if ((projectedPoint.x >= 0 && projectedPoint.x <= imageResolution.width - 1) &&
                            (projectedPoint.y >= 0 && projectedPoint.y <= imageResolution.height - 1)) {
                            projectedPoints.append(projectedPoint)
                            validPoints.append(points[i])
                            validIdentifiers.append(identifiers[i])
                        }
                    }
                    
                    
                    // compute scaled YCbCr image buffer
                    let scaledBuffer = self.IBASampleScaledCapturedPixelBuffer(imageFrame: imageFrame, scale: scale)
                    let scaledLumaBuffer = scaledBuffer.0
                    let scaledCbcrBuffer = scaledBuffer.1
                    
                    
                    // perform YCbCr image sampling
                    if !(projectedPoints.count > 0) {
                        return
                    }
                    for i in 0...(projectedPoints.count - 1) {
                        let projectedPoint = projectedPoints[i]
                        let lumaPoint = CGPoint(x: Double(projectedPoint.x) * scale, y: Double(projectedPoint.y) * scale)
                        let cbcrPoint = CGPoint(x: Double(projectedPoint.x) * scale, y: Double(projectedPoint.y) * scale)
                        
                        let lumaPixelAddress = scaledLumaBuffer.data + scaledLumaBuffer.rowBytes * Int(lumaPoint.y) + Int(lumaPoint.x)
                        let cbcrPixelAddress = scaledCbcrBuffer.data + scaledCbcrBuffer.rowBytes * Int(cbcrPoint.y) + Int(cbcrPoint.x) * 2;
                        
                        let luma = lumaPixelAddress.load(as: UInt8.self)
                        let cb = cbcrPixelAddress.load(as: UInt8.self)
                        let cr = (cbcrPixelAddress + 1).load(as: UInt8.self)
                        
                        let color = simd_make_uint3(UInt32(luma), UInt32(cb), UInt32(cr))
                        self.accumulatedPointCloud.appendPointCloud(validPoints[i], validIdentifiers[i], color)
                    }
                }
            }
        }
    }
    
    func transformNode(name: String, transform: Array<NSNumber>) {
        let node = sceneView.scene.rootNode.childNode(withName: name, recursively: true)
        node?.transform = deserializeMatrix4(transform)
    }
    
    @objc func handleTap(_ recognizer: UITapGestureRecognizer) {
        guard let sceneView = recognizer.view as? ARSCNView else {
            return
        }
        let touchLocation = recognizer.location(in: sceneView)
        let raycastQuery = sceneView.raycastQuery(from: touchLocation, allowing: .estimatedPlane, alignment: .any)
        
        guard let raycastQuery = raycastQuery else { return }
        let planeAndPointHitResults = sceneView.session.raycast(raycastQuery)
        
        if planeAndPointHitResults.count > 0, let hitAnchor = planeAndPointHitResults.first, let currentFrame = sceneView.session.currentFrame {
                        
            let posX = hitAnchor.worldTransform.columns.3.x
            let posY = hitAnchor.worldTransform.columns.3.y
            let posZ = hitAnchor.worldTransform.columns.3.z
            let previousPoint = SCNVector3(posX, posY, posZ)
            
            let sphereNode = SCNNode(geometry: SCNSphere(radius: 0.01))
            sphereNode.position = previousPoint
            sphereNode.name = "test"
            sphereNode.simdPivot.columns.3.x = 0
            sphereNode.geometry?.firstMaterial?.diffuse.contents = UIColor.orange
            
            sceneView.scene.rootNode.addChildNode(sphereNode)
            
            settings.anchors.append(ARAnchorModel(currentFrame: currentFrame, hitAnchor: hitAnchor))
        }
        
    }
    
    // Recursive helper function to traverse a node's parents until a node with a name starting with the specified characters is found
    func nearestParentWithNameStart(node: SCNNode?, characters: String) -> SCNNode? {
        if let nodeNamePrefix = node?.name?.prefix(characters.count) {
            if (nodeNamePrefix == characters) { return node }
        }
        if let parent = node?.parent { return nearestParentWithNameStart(node: parent, characters: characters) }
        return nil
    }
    
    func addPlaneAnchor(transform: Array<NSNumber>, name: String){
        let arAnchor = ARAnchor(transform: simd_float4x4(deserializeMatrix4(transform)))
        anchorCollection[name] = arAnchor
        sceneView.session.add(anchor: arAnchor)
        while (sceneView.node(for: arAnchor) == nil) {
            usleep(1) // wait 1 millionth of a second
        }
    }
    
    func deleteAnchor(anchorName: String) {
        if let anchor = anchorCollection[anchorName]{
            // Delete all child nodes
            if var attachedNodes = sceneView.node(for: anchor)?.childNodes {
                attachedNodes.removeAll()
            }
            // Remove anchor
            sceneView.session.remove(anchor: anchor)
            // Update bookkeeping
            anchorCollection.removeValue(forKey: anchorName)
        }
    }

    func enter(_ state: ARState) {
        switch state {
        case .defaultState:
            if arAnchor != nil {
                sceneView.session.remove(anchor: arAnchor!)
                arAnchor = nil
            }
            if garAnchor != nil {
                arcoreSession?.remove(garAnchor!)
                garAnchor = nil
            }
        default:
            break
        }
        
        self.state = state
        
    }
    
}

// MARK: - ARCoachingOverlayViewDelegate
extension DroidletARView: ARCoachingOverlayViewDelegate {
    
    func coachingOverlayViewWillActivate(_ coachingOverlayView: ARCoachingOverlayView){
        // use this delegate method to hide anything in the UI that could cover the coaching overlay view
    }
    
    func coachingOverlayViewDidRequestSessionReset(_ coachingOverlayView: ARCoachingOverlayView) {
        // Reset the session.
        self.sceneView.session.run(configuration, options: [.resetTracking])
    }
}

extension DroidletARView {
    func hostCloudAnchor(withTransform transform: matrix_float4x4) {
        arAnchor = ARAnchor(transform: transform)
        
        guard let arAnchor = arAnchor else { return }
        
        sceneView.session.add(anchor: arAnchor)
        // To share an anchor, we call host anchor here on the ARCore session.
        // session:disHostAnchor: session:didFailToHostAnchor: will get called appropriately.
        garAnchor = try? arcoreSession?.hostCloudAnchor(arAnchor)
        enter(.hosting)
    }
}

extension DroidletARView: GARSessionDelegate {
    func session(_ session: GARSession, didHost anchor: GARAnchor) {
        if state != .hosting || !(anchor == garAnchor) {
            return
        }
        garAnchor = anchor
        enter(.hostingFinished)
    }
    
    func session(_ session: GARSession, didFailToHost anchor: GARAnchor) {
        if state != .hosting || !(anchor == garAnchor) {
            return
        }
        garAnchor = anchor
        enter(.hostingFinished)
    }
    
    func session(_ session: GARSession, didResolve anchor: GARAnchor) {
        if state != .resolving || !(anchor == garAnchor) {
            return
        }
        garAnchor = anchor
        arAnchor = ARAnchor(transform: anchor.transform)
        sceneView.session.add(anchor: arAnchor!)
        enter(.resolvingFinished)
    }
    
    func session(_ session: GARSession, didFailToResolve anchor: GARAnchor) {
        if state != .resolving || !(anchor == garAnchor) {
            return
        }
        garAnchor = anchor
        enter(.resolvingFinished)
    }
    
}

// MARK: - Odometry
extension DroidletARView {
    private func IBASampleScaledCapturedPixelBuffer(imageFrame: CVPixelBuffer, scale: Double) -> (vImage_Buffer, vImage_Buffer) {
        
        // calculate scaled size for buffers
        let baseWidth = Double(CVPixelBufferGetWidth(imageFrame))
        let baseHeight = Double(CVPixelBufferGetHeight(imageFrame))
        
        let scaledWidth = vImagePixelCount(ceil(baseWidth * scale))
        let scaledHeight = vImagePixelCount(ceil(baseHeight * scale))
        
        
        // lock the source pixel buffer
        CVPixelBufferLockBaseAddress(imageFrame, CVPixelBufferLockFlags.readOnly)
        
        // allocate buffer for scaled Luma & retrieve address of source Luma and scale it
        var scaledLumaBuffer = vImage_Buffer()
        var sourceLumaBuffer = self.IBAPixelBufferGetPlanarBuffer(pixelBuffer: imageFrame, planeIndex: 0)
        vImageBuffer_Init(&scaledLumaBuffer, scaledHeight, scaledWidth, 8, vImage_Flags(kvImagePrintDiagnosticsToConsole))
        vImageScale_Planar8(&sourceLumaBuffer, &scaledLumaBuffer, nil, vImage_Flags(kvImagePrintDiagnosticsToConsole))
        
        // allocate buffer for scaled CbCr & retrieve address of source CbCr and scale it
        var scaledCbcrBuffer = vImage_Buffer()
        var sourceCbcrBuffer = self.IBAPixelBufferGetPlanarBuffer(pixelBuffer: imageFrame, planeIndex: 1)
        vImageBuffer_Init(&scaledCbcrBuffer, scaledHeight, scaledWidth, 8, vImage_Flags(kvImagePrintDiagnosticsToConsole))
        vImageScale_CbCr8(&sourceCbcrBuffer, &scaledCbcrBuffer, nil, vImage_Flags(kvImagePrintDiagnosticsToConsole))
        
        // unlock source buffer now
        CVPixelBufferUnlockBaseAddress(imageFrame, CVPixelBufferLockFlags.readOnly)
        
        
        // return the scaled Luma and CbCr buffer
        return (scaledLumaBuffer, scaledCbcrBuffer)
    }
    
    private func IBAPixelBufferGetPlanarBuffer(pixelBuffer: CVPixelBuffer, planeIndex: size_t) -> vImage_Buffer {
        
        // assumes that pixel buffer base address is already locked
        let baseAddress = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, planeIndex)
        let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, planeIndex)
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
        
        return vImage_Buffer(data: baseAddress, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: bytesPerRow)
    }
    
    private func createFiles() -> Bool {
        
        // initialize file handlers
        self.fileHandlers.removeAll()
        self.fileURLs.removeAll()
        
        // create ARKit result text files
        let startHeader = ""
        for i in 0...(self.numTextFiles - 1) {
            var url = URL(fileURLWithPath: NSTemporaryDirectory())
            url.appendPathComponent(fileNames[i])
            self.fileURLs.append(url)
            
            // delete previous text files
            if (FileManager.default.fileExists(atPath: url.path)) {
                do {
                    try FileManager.default.removeItem(at: url)
                } catch {
                    os_log("cannot remove previous file", log:.default, type:.error)
                    return false
                }
            }
            
            // create new text files
            if (!FileManager.default.createFile(atPath: url.path, contents: startHeader.data(using: String.Encoding.utf8), attributes: nil)) {
                print("cannot create file \(self.fileNames[i])")
                return false
            }
            
            // assign new file handlers
            let fileHandle: FileHandle? = FileHandle(forWritingAtPath: url.path)
            if let handle = fileHandle {
                self.fileHandlers.append(handle)
            } else {
                return false
            }
        }
        
        // write current recording time information
        let timeHeader = "# Created at \(timeToString()) \n"
        for i in 0...(self.numTextFiles - 1) {
            if let timeHeaderToWrite = timeHeader.data(using: .utf8) {
                self.fileHandlers[i].write(timeHeaderToWrite)
            } else {
                os_log("Failed to write data record", log: OSLog.default, type: .fault)
                return false
            }
        }
        
        // return true if everything is alright
        return true
    }
    
    func toggleRecording() {
        if (self.isRecording == false) {
            self.accumulatedPointCloud.resetPointCloud()
            // start ARKit data recording
            customQueue.async {
                if (self.createFiles()) {
                    print("File created successfully")
                } else {
                    print("Failed to create the file")
                    return
                }
            }
        } else {
            customQueue.async {                
                // save ARKit 3D point cloud only for visualization
                for i in 0...(self.accumulatedPointCloud.count - 1) {
                    let ARKitPointData = String(format: "%.6f %.6f %.6f %d %d %d \n",
                                                self.accumulatedPointCloud.points[i].x,
                                                self.accumulatedPointCloud.points[i].y,
                                                self.accumulatedPointCloud.points[i].z,
                                                self.accumulatedPointCloud.colors[i].x,
                                                self.accumulatedPointCloud.colors[i].y,
                                                self.accumulatedPointCloud.colors[i].z)
                    if let ARKitPointDataToWrite = ARKitPointData.data(using: .utf8) {
                        self.fileHandlers[self.ARKIT_POINT_CLOUD].write(ARKitPointDataToWrite)
                    } else {
                        os_log("Failed to write data record", log: OSLog.default, type: .fault)
                    }
                }
                
                // close the file handlers
                if (self.fileHandlers.count == self.numTextFiles) {
                    for handler in self.fileHandlers {
                        handler.closeFile()
                    }
                }
            }
            
            // initialize UI on the screen
            if let del = self.delegate {
                del.trackingInfoDidUpdate(points: "...", status: "...")
            }
            
            // resume screen lock
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }
    
    func uploadRecordDataFile() {
//        let boundary: String = UUID().uuidString
        let uploadURL = "http://34.145.124.241:5000/firebase/odometry"
        let headers: HTTPHeaders = [
            "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJmcmVzaCI6ZmFsc2UsImlhdCI6MTY1NDA5NTQ2NiwianRpIjoiYzZlY2UyMGQtNDQyNC00NmY0LTkwNzgtZDNjNGUyOGU2ZjRlIiwidHlwZSI6ImFjY2VzcyIsInN1YiI6IntcImlkXCI6IFwielpObXFuWjdic2M0dE9ITWJWaGRcIiwgXCJ1c2VybmFtZVwiOiBcInZtb1wiLCBcInBhc3N3b3JkXCI6IFwiJDJiJDEyJGZ3U3NzbkJ6U0VnWDk2RkF5cDdIdGVsZUU5cEtmNzRQVlZpNmNvQ3FVaG5OR1pTMFdFNVhTXCIsIFwidXNlcl91aWRcIjogXCJcIn0iLCJuYmYiOjE2NTQwOTU0NjYsImV4cCI6MTY1NjY4NzQ2Nn0.EK4_OFMCZnwlHNZTaJD_shiv7fWFeSH87x56c24kPso",
            "Content-type": "multipart/form-data",
            "Accept": "*/*",
            "Accept-Encoding" : "gzip, deflate, br"
        ]
        
        AF.upload(
            multipartFormData: { multipartFormData in
                for fileURL in self.fileURLs {
                    if let fileData = try? Data(contentsOf: fileURL) {
                        if fileURL.absoluteString.contains("point_cloud") {
                            multipartFormData.append(fileData, withName: "point_cloud" , fileName: "point_cloud.txt", mimeType: "text/plain")
                        } else if fileURL.absoluteString.contains("camera_pose") {
                            multipartFormData.append(fileData, withName: "camera_pose" , fileName: "camera_pose.txt", mimeType: "text/plain")
                        }
                    }
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
