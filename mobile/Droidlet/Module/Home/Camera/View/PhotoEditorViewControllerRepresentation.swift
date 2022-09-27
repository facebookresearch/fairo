import SwiftUI

struct PhotoEditorViewControllerRepresentation: UIViewControllerRepresentable {
    
    @ObservedObject var viewModel: HomeViewModel
    var image: UIImage?
    
    typealias UIViewControllerType = PhotoEditorViewController
    
    func makeUIViewController(context: Context) -> PhotoEditorViewController {
        let photoEditor = PhotoEditorViewController(nibName:"PhotoEditorViewController",bundle: nil)
        photoEditor.image = self.image
        photoEditor.name = viewModel.name
        photoEditor.photoEditorDelegate = context.coordinator
        return photoEditor
    }
    
    func updateUIViewController(_ uiViewController: PhotoEditorViewController, context: Context) {
        
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }
    
    class Coordinator: NSObject, PhotoEditorDelegate {
        var parent: PhotoEditorViewControllerRepresentation
        
        init(_ parent: PhotoEditorViewControllerRepresentation) {
            self.parent = parent
        }
        
        func doneEditing(image: UIImage, name: String) {
            Droidbase.shared.saveToCloud(nil, name, image)
            parent.viewModel.image = nil
            parent.viewModel.dashboardMode = .ar
        }
        
        func canceledEditing() {
            parent.viewModel.image = nil
        }
    }
}
