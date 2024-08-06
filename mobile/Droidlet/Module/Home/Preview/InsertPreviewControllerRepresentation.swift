import SwiftUI

struct InsertPreviewControllerRepresentation: UIViewControllerRepresentable {
    
    @ObservedObject var viewModel: HomeViewModel

    typealias UIViewControllerType = InsertPreviewController

    func makeUIViewController(context: Context) -> InsertPreviewController {
        let viewController = InsertPreviewController(nibName:"InsertPreviewController",bundle: nil)
        viewController.path = viewModel.fileUrl
        viewController.delegate = context.coordinator
        return viewController
    }
    
    func updateUIViewController(_ uiViewController: InsertPreviewController, context: Context) {
        
    }
    
    func makeCoordinator() -> Coordinator {
        return Coordinator(self)
    }
    
    class Coordinator: NSObject, InsertPreviewControllerDelegate {
        var parent: InsertPreviewControllerRepresentation
        
        init(_ parent: InsertPreviewControllerRepresentation) {
            self.parent = parent
        }
        
        func doneEditing(name: String) {
//            parent.viewModel.upload(image: UIImage(), name: name, url: parent.viewModel.fileUrl)
        }
        
        func canceledEditing() {
            
        }
    }
}


