
import Foundation
import UIKit

// MARK: - Control
public enum control {
    case draw
    case text
}

extension PhotoEditorViewController {

     //MARK: Top Toolbar
    @IBAction func cancelButtonTapped(_ sender: Any) {
        photoEditorDelegate?.canceledEditing()
        self.dismiss(animated: true, completion: nil)
    }

    @IBAction func drawButtonTapped(_ sender: Any) {
        isDrawing = true
        canvasImageView.isUserInteractionEnabled = false
        doneButton.isHidden = false
        hideToolbar(hide: true)
    }
    
    @IBAction func doneButtonTapped(_ sender: Any) {
        view.endEditing(true)
        doneButton.isHidden = true
        canvasImageView.isUserInteractionEnabled = true
        hideToolbar(hide: false)
        isDrawing = false
    }
    
    @IBAction func continueButtonPressed(_ sender: Any) {
        let img = self.canvasView.toImage()
        presentAlert(image: img)
    }

    //MAKR: helper methods
    
    @objc func image(_ image: UIImage, withPotentialError error: NSErrorPointer, contextInfo: UnsafeRawPointer) {
        let alert = UIAlertController(title: "Image Saved", message: "Image successfully saved to Photos library", preferredStyle: UIAlertController.Style.alert)
        alert.addAction(UIAlertAction(title: "OK", style: UIAlertAction.Style.default, handler: nil))
        self.present(alert, animated: true, completion: nil)
    }
    
    func hideControls() {
        for control in hiddenControls {
            switch control {
            case .draw:
                drawButton.isHidden = true
            case .text:
                break
            }
        }
    }
    
    func presentAlert(image: UIImage) {
        guard let pixelBuffer = image.convertToBuffer() else { return }
        settings.visionRequest(.camera, image: image, pixelBuffer, completion: nil) { confidence, name in
            DispatchQueue.main.async {
                let alert = self.settings.alert(message: confidence, inputText: name) { name in
                    self.dismiss(animated: true, completion: nil)
                    self.photoEditorDelegate?.doneEditing(image: image, name: name)
                }
                self.present(alert, animated: true)
            }

        }
    }
}
