
import UIKit
import SwiftUI

public final class PhotoEditorViewController: UIViewController {
    var settings: UserSettings = .shared

    /** holding the 2 imageViews original image and drawing & stickers */
    @IBOutlet weak var canvasView: UIView!
    //To hold the image
    @IBOutlet var imageView: UIImageView!
    @IBOutlet weak var imageViewHeightConstraint: NSLayoutConstraint!
    //To hold the drawings and stickers
    @IBOutlet weak var canvasImageView: UIImageView!

    @IBOutlet weak var topToolbar: UIView!
    @IBOutlet weak var bottomToolbar: UIView!

    @IBOutlet weak var topGradient: UIView!
    @IBOutlet weak var bottomGradient: UIView!
    
    @IBOutlet weak var doneButton: UIButton!
    @IBOutlet weak var deleteView: UIView!
    
    //Controls
    @IBOutlet weak var drawButton: UIButton!
    
    public var image: UIImage?
    public var name: String = ""
    /**
     Array of Colors that will show while drawing or typing
     */
    public var colors  : [UIColor] = []
    
    public var photoEditorDelegate: PhotoEditorDelegate?
    
    // list of controls to be hidden
    public var hiddenControls : [control] = []
    
    var swiped = false
    var drawColor: UIColor = UIColor.red
    var isDrawing: Bool = false
    var lastPoint: CGPoint!
    var lastPanPoint: CGPoint?
    
    //Register Custom font before we load XIB
    public override func loadView() {
        super.loadView()
    }
    
    override public func viewDidLoad() {
        super.viewDidLoad()
        self.setImageView(image: image!)
        
        deleteView.layer.cornerRadius = deleteView.bounds.height / 2
        deleteView.layer.borderWidth = 2.0
        deleteView.layer.borderColor = UIColor.white.cgColor
        deleteView.clipsToBounds = true
        
        hideControls()
    }
    
    func setImageView(image: UIImage) {
        imageView.image = image
        let size = image.suitableSize(widthLimit: UIScreen.main.bounds.width)
        imageViewHeightConstraint.constant = (size?.height)!
    }
    
    func hideToolbar(hide: Bool) {
        topToolbar.isHidden = hide
        topGradient.isHidden = hide
        bottomToolbar.isHidden = hide
        bottomGradient.isHidden = hide
    }
}
