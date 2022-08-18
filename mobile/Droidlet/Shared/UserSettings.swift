import Foundation
import UIKit
import CoreML
import Vision

enum NotificationAnswer: Int {
    case yes = 1
    case no = 0
}

struct NotificationObject {
    var title: String
    var body: String
    var image_url: String?
    var image: UIImage?
    var message_id: String = ""
    
    mutating func setImage(_ image: UIImage?) {
        self.image = image
    }
    
    static var `default` = NotificationObject(title: "", body: "")
}

class UserSettings: ObservableObject {
    private init() { }
    
    static let shared = UserSettings()
    
    var visionRequests = [VNRequest]()
    let resnetModel: Resnet50 = {
        do {
            let configuration = MLModelConfiguration()
            return try Resnet50(configuration: configuration)
        } catch let error {
            fatalError(error.localizedDescription)
        }
    }()
    
    var deviceOrientation: CGImagePropertyOrientation {
        switch UIDevice.current.orientation {
            case .portrait:           return .right
            case .portraitUpsideDown: return .left
            case .landscapeLeft:      return .up
            case .landscapeRight:     return .down
            default:                  return .right
        }
    }

    var notificationObject: NotificationObject = .default
    var anchors: [ARAnchorModel] = []
    @Published var deviceToken: String = ""
    @Published var expride: Bool = false
    
    @Published var reload: Bool = false

    func alert(message: String?, inputText: String?, okAction: @escaping (String)->()) -> UIAlertController {
        let alert = UIAlertController(title: "What do you call this?",
                                      message: message,
                                      preferredStyle: .alert)
        alert.addTextField { (textField) in
            textField.text = inputText
        }
        alert.addAction(UIAlertAction(title: "OK", style: .default) { [weak alert] (_) in
            if let name = alert?.textFields?.first?.text {
                okAction(name)
            }
        })
        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel))
        
        return alert
    }
    
    // Step 2: ask the model what this is
    func visionRequest(_ dashboardMode: DashboardMode, image: UIImage? = nil, _ pixelBuffer: CVPixelBuffer, completion: (()->())? = nil, testHandle: ((String, String)->())? = nil) {
        let visionModel = try! VNCoreMLModel(for: resnetModel.model)
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
                
                if dashboardMode == .camera {
                    testHandle?(confidence, named)
                } else {
                    self.askName(dashboardMode, image: image, suggestion: named, confidence: confidence, completion: completion)
                }
            }
        }
        request.imageCropAndScaleOption = .centerCrop
        visionRequests = [request]
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                                        orientation: deviceOrientation, // or .upMirrored?
                                                        options: [:])
        DispatchQueue.global().async {
            try! imageRequestHandler.perform(self.visionRequests)
        }
    }
    
    // MARK: - API
    func sendTokenToBE(_ device_token: String) {
        let urlComponents = NSURLComponents(string: APIMainEnvironmentConfig.baseURL + APIMainEnvironmentConfig.Endpoint.update_device_token.rawValue)!
        guard let url = urlComponents.url
        else {preconditionFailure("Invalid URL format")}
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        guard let accessToken = SessionManage.shared.accessToken else { return }
        let headers = [
            "Authorization": "Bearer \(accessToken)",
            "Content-Type": "application/json"
        ]
        request.allHTTPHeaderFields = headers
        
        if device_token.isEmpty { return }
        let body = ["device_token": device_token]
        do {
            let data = try JSONSerialization.data(withJSONObject: body, options: [])
            request.httpBody = data
        } catch let error {
            Logger.logDebug(error.localizedDescription)
        }
        
        let dataTask = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                Logger.logDebug(error.localizedDescription)
                return
            }
            
            guard let response = response as? HTTPURLResponse else { return }
            
            if response.statusCode == 200 {
                guard let data = data else { return }
                Logger.logDebug(String(data: data, encoding: .utf8) ?? "")
            }
            
            if response.statusCode == 401 { // expride
                Logger.logDebug("Token has expired")
                SessionManage.shared.logout()
                // Back to Login
                DispatchQueue.main.async {
                    self.expride = true
                }
            }
        }
        
        dataTask.resume()
    }

    func markImage(value: NotificationAnswer, message_id: String, option: String?) {
        HomeService.shared.markImage(value: value, message_id: message_id, option: option)
    }
}

extension UserSettings {
    func request(_ dashboardMode: DashboardMode, image: UIImage? = nil, completion: (()->())? = nil) {
        switch dashboardMode {
        case .camera:
            guard let pixelBuffer = image?.convertToBuffer() else { return }
            visionRequest(dashboardMode, image: image, pixelBuffer, completion: completion)
        case .ar:
            if anchors.isEmpty { return }
            let first = anchors.first!
            
            let pixelBuffer = first.currentFrame.capturedImage
            visionRequest(dashboardMode, pixelBuffer)
        default:
            break
        }
    }
    
    func askName(_ dashboardMode: DashboardMode, image: UIImage? = nil, suggestion: String, confidence: String, completion: (()->())? = nil) {
        let alert = alert(message: confidence, inputText: suggestion) { name in
            switch dashboardMode {
            case .camera:
                if let image = image {
                    Droidbase.shared.saveToCloud(nil, name, image)
                }
                completion?()
            case .ar:
                self.anchors.forEach { anchor in
                    let image = CIImage(cvPixelBuffer: anchor.currentFrame.capturedImage).oriented(self.deviceOrientation)
                    Droidbase.shared.saveToCloud(nil, name, UIImage(ciImage: image))
                }
                
                self.anchors.removeAll()
                self.reload = true
            default:
                break
            }
        }

        if let delegate = UIApplication.shared.delegate as? AppDelegate,
           let parentViewController = delegate.window?.rootViewController {
            parentViewController.present(alert, animated: true)
        }
    }
}
