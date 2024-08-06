import SwiftUI
import SDWebImageSwiftUI

struct ChatImageView: View {
    @ObservedObject var viewModel: HomeViewModel
    var chatModel: ChatModel
    
    var body: some View {
        ScrollView (.horizontal) {
            HStack {
                ForEach (chatModel.attachment, id: \.id) { imageChatModel in
                    if let urlString = imageChatModel.url, let mapInfo = imageChatModel.map, let uid = mapInfo.uid {
                        ImgView(imageManager: ImageManager(url: URL(string: urlString)), viewModel: viewModel, uid: uid)
                    }
                }
            }
        }
    }
}

struct ImgView : View {
    @StateObject var imageManager: ImageManager
    @ObservedObject var viewModel: HomeViewModel
    var uid: String
    
    var body: some View {
        // Your custom complicated view graph
        Group {
            if imageManager.image != nil {
                Image(uiImage: imageManager.image!)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 50, height: 50, alignment: .center)
                    .gesture(
                        TapGesture()
                            .onEnded({ _ in
                                viewModel.image = imageManager.image
                                viewModel.uid = uid
                                viewModel.locationImageAction()
                            })
                    )
            } else {
                Rectangle().fill(Color.gray)
            }
        }
        // Trigger image loading when appear
        .onAppear { self.imageManager.load() }
        // Cancel image loading when disappear
        .onDisappear { //self.imageManager.cancel()
            
        }
    }
}
