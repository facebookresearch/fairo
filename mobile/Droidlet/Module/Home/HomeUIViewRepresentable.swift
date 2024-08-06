import SwiftUI
import RealityKit
import ARKit
import Vision

struct ARViewRepresentable: UIViewRepresentable {
    @ObservedObject var viewModel: HomeViewModel
    let arView: ARView
    var currentBuffer: CVPixelBuffer?
    	
	class Coordinator: NSObject, ARSessionDelegate {
		var parent: ARViewRepresentable
		var viewportSize: CGSize! {
			return parent.arView.frame.size
		}
		
		var hitLocationInView: CGPoint?
        private let visionQueue = DispatchQueue(label: "com.ys.serialVisionQueue")
		
		init(_ parent: ARViewRepresentable) {
			self.parent = parent
		}
		
		// MARK: - ARSessionDelegate
		
		// Pass camera frames received from ARKit to Vision (when not already processing one)
		/// - Tag: ConsumeARFrames
		func session(_ session: ARSession, didUpdate frame: ARFrame) {
			// Do not enqueue other buffers for processing while another Vision task is still running.
			// The camera stream has only a finite amount of buffers available; holding too many buffers for analysis would starve the camera.
			guard self.parent.currentBuffer == nil, case .normal = frame.camera.trackingState, parent.viewModel.detectRemoteControl == true else {
				return
			}
			
			// Retain the image buffer for Vision processing.
			self.parent.currentBuffer = frame.capturedImage
			classifyCurrentImage()
		}
		
		// Run the Vision+ML classifier on the current image buffer.
		/// - Tag: ClassifyCurrentImage
		private func classifyCurrentImage() {
			let requestHandler = VNImageRequestHandler(cvPixelBuffer: self.parent.currentBuffer!, orientation: .leftMirrored)
			visionQueue.async {
				do {
					// Release the pixel buffer when done, allowing the next buffer to be processed.
					defer { self.parent.currentBuffer = nil }
					try requestHandler.perform([self.classificationRequest])
				} catch {
					print("Error: Vision request failed with error \"\(error)\"")
				}
			}
		}
		
		// MARK: - Vision classification
		
		// Vision classification request and model
		/// - Tag: ClassificationRequest
		private lazy var classificationRequest: VNCoreMLRequest = {
			do {
				// Instantiate the model from its generated Swift class.
				//let model = try VNCoreMLModel(for: Inceptionv3().model)
				let model = try VNCoreMLModel(for: YOLOv3Tiny().model)
				let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
					self?.processClassifications(for: request, error: error)
				})
				
				// Crop input images to square area at center, matching the way the ML model was trained.
				request.imageCropAndScaleOption = .centerCrop
				
				// Use CPU for Vision processing to ensure that there are adequate GPU resources for rendering.
				request.usesCPUOnly = true
				
				return request
			} catch {
				fatalError("Failed to load Vision ML model: \(error)")
			}
		}()
		
		// Handle completion of the Vision request and choose results to display.
		/// - Tag: ProcessClassifications
		func processClassifications(for request: VNRequest, error: Error?) {
			guard let results = request.results else {
				print("Unable to classify image.\n\(error!.localizedDescription)")
				return
			}
			
			DispatchQueue.main.async {
				for observation in results where observation is VNRecognizedObjectObservation {
					guard let objectObservation = observation as? VNRecognizedObjectObservation,
						  let topLabelObservation = objectObservation.labels.first else { continue }
					guard let currentFrame = self.parent.arView.session.currentFrame else { return }
					
					// Get the affine transform to convert between normalized image coordinates and view coordinates
					let fromCameraImageToViewTransform = currentFrame.displayTransform(for: .portrait, viewportSize: self.viewportSize)
					// The observation's bounding box in normalized image coordinates
					let boundingBox = objectObservation.boundingBox
					// Transform the latter into normalized view coordinates
					let viewNormalizedBoundingBox = boundingBox.applying(fromCameraImageToViewTransform)
					// The affine transform for view coordinates
					let t = CGAffineTransform(scaleX: self.viewportSize.width, y: self.viewportSize.height)
					// Scale up to view coordinates
					let viewBoundingBox = viewNormalizedBoundingBox.applying(t)
					let midPoint = CGPoint(x: viewBoundingBox.midX,
										   y: viewBoundingBox.midY)
					let estimatedPlane: ARRaycastQuery.Target = .estimatedPlane
					let alignment: ARRaycastQuery.TargetAlignment = .any
					let _results = self.parent.arView.raycast(from: midPoint,
													   allowing: estimatedPlane,
													   alignment: alignment)
					guard let result = _results.first else { return }
					let textEntity = AnchorEntity(world: result.worldTransform)
					textEntity.setText(topLabelObservation.identifier)
					self.parent.arView.scene.anchors.append(textEntity)
					self.parent.viewModel.detectRemoteControl = false
					break
				}
				
			}
		}
		
		@objc func tapGesture(_ gesture: UITapGestureRecognizer) {
			hitLocationInView = gesture.location(in: parent.arView)
			
			// Hightlight object
			 let estimatedPlane: ARRaycastQuery.Target = .estimatedPlane
			 let alignment: ARRaycastQuery.TargetAlignment = .any
			 
			 let result = parent.arView.raycast(from: hitLocationInView!,
												allowing: estimatedPlane,
												alignment: alignment)
			 
			 guard let raycast: ARRaycastResult = result.first else { return }
			 
			 // get its material
//			 let sphere = ModelEntity(mesh: .generateBox(size: SIMD3(raycast.worldTransform.columns.3.x, raycast.worldTransform.columns.3.y, raycast.worldTransform.columns.3.z), cornerRadius: 0.2))
            let sphere = ModelEntity(mesh: .generateSphere(radius: 0.05))
			 let material = SimpleMaterial(color: .green.withAlphaComponent(0.3), roughness: .float(0), isMetallic: true)
			 sphere.model?.materials = [material]
			 
            parent.viewModel.anchor = AnchorEntity(world: raycast.worldTransform)
            parent.viewModel.anchor?.addChild(sphere)
            parent.arView.scene.anchors.append(parent.viewModel.anchor!)
			
			parent.viewModel.showAlert = true
		}
		
	}
	
	func makeUIView(context: Context) -> some UIView {
		arView.session.delegate = context.coordinator
		let gesture = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.tapGesture(_:)))
		arView.addGestureRecognizer(gesture)
		
		let configuration = ARWorldTrackingConfiguration()
		configuration.planeDetection = []
		arView.session.run(configuration, options: [.removeExistingAnchors, .resetTracking])
		return arView
	}
	
	func updateUIView(_ uiView: UIViewType, context: Context) {
		guard let hitLocationInView = context.coordinator.hitLocationInView, !viewModel.objectName.isEmpty, !viewModel.showAlert else { return }
		let estimatedPlane: ARRaycastQuery.Target = .estimatedPlane
		let alignment: ARRaycastQuery.TargetAlignment = .any
		
		let raycast = arView.raycast(from: hitLocationInView,
										   allowing: estimatedPlane,
										   alignment: alignment)
		
		if let result = raycast.first {
			let textEntity = AnchorEntity(world: result.worldTransform)
			textEntity.setText(viewModel.objectName)
			arView.scene.anchors.append(textEntity)
			
			viewModel.postToFirebase(viewModel.objectName, transform: result.worldTransform)
			
		}
        
        if let _anchor = viewModel.anchor {
            arView.scene.anchors.remove(_anchor)
            viewModel.anchor = nil
        }
		
		viewModel.objectName = ""
		context.coordinator.hitLocationInView = nil
	}
	
	func makeCoordinator() -> Coordinator {
		Coordinator(self)
	}
	
}
