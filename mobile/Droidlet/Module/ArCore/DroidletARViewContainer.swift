import SwiftUI
import ARKit
import Combine

struct DroidletARViewContainer: UIViewRepresentable {
    @ObservedObject var viewModel: HomeViewModel
    var settings: UserSettings = .shared
	var isHomeAR = true
    private var cancellable: AnyCancellable?

    var rayCastResultValue: ARRaycastResult?
    var visionRequests = [VNRequest]()
    
    let resnetModel: Resnet50 = {
        do {
            let configuration = MLModelConfiguration()
            return try Resnet50(configuration: configuration)
        } catch let error {
            fatalError(error.localizedDescription)
        }
    }()
    
	init(viewModel: HomeViewModel, isHome: Bool) {
        self.viewModel = viewModel
		self.isHomeAR = isHome
    }

    func makeUIView(context: Context) -> DroidletARView {
		let config : Dictionary<String,Any>? = isHomeAR ? nil : ["planeDetectionConfig": 0,
																 "showPlanes": false,
																 "showFeaturePoints": false,
																 "showWorldOrigin": false,
																 "handleTaps": false]
		viewModel.iosView = DroidletARView(config: config)
        viewModel.iosView.delegate = context.coordinator
        return viewModel.iosView
    }
    
    func updateUIView(_ uiView: DroidletARView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }
    
    final class Coordinator: NSObject, IosARViewDelegate {
        func trackingInfoDidUpdate(points: String, status: String) {
            self.parent.viewModel.featurePoints = points
            self.parent.viewModel.trackingStatus = status
        }
        
        var parent: DroidletARViewContainer
        
        init( _ parent: DroidletARViewContainer) {
            self.parent = parent
        }
        
        func tapGesture(_ frame: ARFrame, raycastResult: ARRaycastResult) {
            self.parent.viewModel.markerImage(frame.capturedImage, orientation: self.parent.settings.deviceOrientation)
        }
                
        // Step 4: put the name on the object
        func createText(_ generatedText: String) {

        }
    }
}
