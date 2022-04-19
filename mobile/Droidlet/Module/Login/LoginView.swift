import SwiftUI

struct LoginView: View {
    @ObservedObject var viewModel: LoginViewModel
    @State private var isLoading: Bool = false
    
    var body: some View {
        VStack(spacing: 20) {            
            Text("Welcome")
                .font(Font.largeTitle)
                .foregroundColor(Color.black)
                .padding(.top, 50)
            Spacer()
            Button {
                isLoading = true
                viewModel.signinWithGoogle()
			} label: {
				Text("Sign in with google")
					.padding(5)
					.foregroundColor(Color.white)
					.cornerRadius(10)
					.padding()
					.background(Color(red: 0, green: 0, blue: 0.5))
					.foregroundColor(.white)
					.clipShape(Capsule())
			}
			Spacer()
            if isLoading {
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle(tint: .gray))
                    .scaleEffect(3)
            }

            Spacer()
        }
        .onReceive(viewModel.$loggedIn) { _ in
            isLoading = false
        }
        .onAppear {
            UserDefaults.standard.setValue(true, forKey: "firstLaunch")
            viewModel.settings.expride = false
        }
    }
}

struct LoginView_Previews: PreviewProvider {
    static var previews: some View {
        LoginView(viewModel: LoginViewModel())
    }
}
