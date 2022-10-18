import SwiftUI

struct ChatDetailView: View {
    @ObservedObject var viewModel: HomeViewModel

    var body: some View {
        ScrollView {
            VStack(spacing: 5) {
                ForEach(0..<viewModel.listChat.count, id: \.self) { index in
                    if viewModel.listChat[index].isUserInput {
                        HStack {
                            Spacer()
                            Text(viewModel.listChat[index].text)
                                .padding(EdgeInsets(top: 0, leading: 5, bottom: 0, trailing: 5))
                                .foregroundColor(.white)
                                .background(Color.blue)
                                .cornerRadius(10)
                        }
                    } else {
                        if !viewModel.listChat[index].attachment.isEmpty {
                            HStack {
                                Text(viewModel.listChat[index].text)
                                    .padding(EdgeInsets(top: 0, leading: 5, bottom: 0, trailing: 5))
                                    .foregroundColor(.white)
                                    .background(Color.green)
                                    .cornerRadius(10)
                                Spacer()
                            }
                            
                            ChatImageView(viewModel: viewModel, chatModel: viewModel.listChat[index])
                        } else {
                            HStack {
                                Text(viewModel.listChat[index].text)
                                    .padding(EdgeInsets(top: 0, leading: 5, bottom: 0, trailing: 5))
                                    .foregroundColor(.white)
                                    .background(Color.green)
                                    .cornerRadius(10)
                                Spacer()
                            }
                        }
                    }
                }
            }
            
            Spacer(minLength: 5)
            if viewModel.agentState.getText().0 {
                AppThinkingView(text: viewModel.agentState.getText().1)
            }
        }
        .frame(height: 200)
    }
}

struct ChatDetailView_Previews: PreviewProvider {
    static var previews: some View {
        ChatDetailView(viewModel: HomeViewModel())
    }
}

struct AppThinkingView: View {
    var text: String
    var body: some View {
        HStack {
            Spacer()
            Text(text)
                .padding(EdgeInsets(top: 0, leading: 5, bottom: 0, trailing: 5))
                .foregroundColor(.white)
                .background(Color.black)
                .cornerRadius(10)
        }
    }
}
