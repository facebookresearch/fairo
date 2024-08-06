import UIKit
import SwiftUI

protocol InsertPreviewControllerDelegate: AnyObject {
    func doneEditing(name: String)
    func canceledEditing()
}

class InsertPreviewController: UIViewController {
    var settings: UserSettings = .shared

    @IBOutlet weak var insert3DView: UIView!
    var path: URL?
    
    weak var delegate: InsertPreviewControllerDelegate?

    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard let path = path else { return }
        
        let viewer = Insert3DViewer()
        var model = Insert3DModel()
        model.mesh = path
        model.material = ""
        insert3DView.Insert3D(viewerSetup: viewer, modelSetup: model)

    }
    
    // MARK: - Action
    
    @IBAction func cancel(_ sender: Any) {
        self.dismiss(animated: true, completion: nil)
        self.delegate?.canceledEditing()
    }
    
    @IBAction func edit(_ sender: Any) {        
        let alert = settings.alert(message: nil, inputText: nil) { name in
            self.delegate?.doneEditing(name: name)
            self.dismiss(animated: true)
        }
        present(alert, animated: true)
    }
    
    @IBAction func save(_ sender: Any) {
        self.delegate?.doneEditing(name: "image3D")
        self.dismiss(animated: true, completion: nil)
    }
}
