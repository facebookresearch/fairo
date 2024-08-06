import Foundation

struct ChatModel {
    let text: String
    let isUserInput: Bool
    var attachment: [ImageChatModel] = []
}
