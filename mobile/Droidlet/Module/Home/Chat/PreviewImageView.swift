import SwiftUI

struct PreviewImageView: View {
    @ObservedObject var viewModel: HomeViewModel
    @Environment(\.presentationMode) var presentationMode
    @Binding var shouldNavigate: Bool
    var dataModelInstance: DataModel = DataModel.dataModelSharedInstance
    
    var body: some View {
        ZStack {
            ZStack {
                if let image = viewModel.image {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
            }
            VStack {
                HStack {
                    Button("Close") {
                        presentationMode.wrappedValue.dismiss()
                        shouldNavigate = false
                    }
                    .padding(.top, 10)
                    .padding(.leading, 15)
                    .buttonStyle(AccentFilledButton())

                    Spacer()
                    
                    Button("Navigate") {
                        presentationMode.wrappedValue.dismiss()
                        viewModel.setSelectedLocationByUID()
                        viewModel.didSelectNavigating = true
                        shouldNavigate = true
                    }
                    .padding(.top, 10)
                    .padding(.trailing, 15)
                    .buttonStyle(AccentFilledButton())
                }
                Spacer()
            }
        }
        
    }
}
