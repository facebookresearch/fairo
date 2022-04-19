import ARKit
import RealityKit
import SwiftUI
import VideoToolbox
import Vision

// The UI view for the augmented reality
// Supports one user story with these steps:
// 0. Display the camera view
// 1. User taps on something
// 2. Ask ML model what the camera is looking at
// 3. Ask user to confirm what the ML says
// 4. Sticks the result on the object as AR text

struct ARViewContainer: UIViewRepresentable {

    let resnetModel: Resnet50 = {
        do {
            let configuration = MLModelConfiguration()
            return try Resnet50(configuration: configuration)
        } catch let error {
            fatalError(error.localizedDescription)
        }
    }()

    let arView = ARView(frame: .zero)
    var rayCastResultValue: ARRaycastResult!
    var visionRequests = [VNRequest]()

    func makeUIView(context: Context) -> ARView {
        let tapGesture = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.tapGestureMethod(_:))
        )
        arView.addGestureRecognizer(tapGesture)
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }

    final class Coordinator: NSObject {
        var parent: ARViewContainer

        // So that we evaluate and save the image right side up
        var deviceOrientation: CGImagePropertyOrientation {
            switch UIDevice.current.orientation {
                case .portrait:           return .right
                case .portraitUpsideDown: return .left
                case .landscapeLeft:      return .up
                case .landscapeRight:     return .down
                default:                  return .right
            }
        }

        init( _ parent: ARViewContainer) {
            self.parent = parent
            guard ARWorldTrackingConfiguration.isSupported else {
                fatalError("ARKit is not available on this device.")
            }
        }

        // Step 1: user taps on an object
        @objc func tapGestureMethod(_ sender: UITapGestureRecognizer) {
            guard let sceneView = sender.view as? ARView else { return }

            let touchLocation = parent.arView.center
            let result = parent.arView.raycast(from: touchLocation,
                                               allowing: .estimatedPlane,
                                               alignment: .any)
            guard let raycastHitTestResult: ARRaycastResult = result.first,
                  let currentFrame = sceneView.session.currentFrame else {
                return
            }

            parent.rayCastResultValue = raycastHitTestResult
            visionRequest(currentFrame.capturedImage)
        }

        // Step 2: ask the model what this is
        private func visionRequest(_ pixelBuffer: CVPixelBuffer) {
            let visionModel = try! VNCoreMLModel(for: parent.resnetModel.model)
            let request = VNCoreMLRequest(model: visionModel) { request, error in
                guard error == nil else {
                    print(error!.localizedDescription)
                    return
                }
                guard let observations = request.results ,
                      let observation = observations.first as? VNClassificationObservation else {
                    print("Could not classify")
                    return
                }
                DispatchQueue.main.async {
                    let named = observation.identifier.components(separatedBy: ", ").first!
                    let confidence = "\(named): \((Int)(observation.confidence * 100))% confidence"
                    self.askName(suggestion: named, confidence: confidence, pixelBuffer)
                }
            }
            request.imageCropAndScaleOption = .centerCrop
            parent.visionRequests = [request]
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                            orientation: deviceOrientation, // or .upMirrored?
                                                            options: [:])
            DispatchQueue.global().async {
                try! imageRequestHandler.perform(self.parent.visionRequests)
            }
        }

        // Step 3: ask the user to confirm
        func askName(suggestion: String, confidence: String, _ pixelBuffer: CVPixelBuffer) {
            let alert = UIAlertController(title: "What do you call this?",
                                          message: confidence,
                                          preferredStyle: .alert)
            alert.addTextField { (textField) in
                textField.text = suggestion
            }
            alert.addAction(UIAlertAction(title: "OK", style: .default) { [weak alert] (_) in
                if let name = alert?.textFields?.first?.text {
                    let image = CIImage(cvPixelBuffer: pixelBuffer).oriented(self.deviceOrientation)
                    self.createText(name)
                    Droidbase.shared.saveToCloud(self.parent.arView, name, UIImage(ciImage: image))
                }
            })
            alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))

            if let delegate = UIApplication.shared.delegate as? AppDelegate,
               let parentViewController = delegate.window?.rootViewController {
                parentViewController.present(alert, animated: true)
            }
        }

        // Step 4: put the name on the object
        func createText(_ generatedText: String) {
            let mesh = MeshResource.generateText(generatedText,
                                                 extrusionDepth: 0.01,
                                                 font: UIFont(name: "HelveticaNeue", size: 0.05)!,
                                                 containerFrame: CGRect.zero,
                                                 alignment: .center,
                                                 lineBreakMode: .byCharWrapping)

            let material = SimpleMaterial(color: .green, roughness: 1, isMetallic: true)
            let modelEntity = ModelEntity(mesh: mesh, materials: [material])
            let anchorEntity = AnchorEntity(world: SIMD3<Float>(parent.rayCastResultValue.worldTransform.columns.3.x,
                                                                parent.rayCastResultValue.worldTransform.columns.3.y,
                                                                parent.rayCastResultValue.worldTransform.columns.3.z))
            anchorEntity.addChild(modelEntity)
            parent.arView.scene.addAnchor(anchorEntity)
        }
    }
}
