import Foundation
import FirebaseAuth
import Firebase
import Combine
import GoogleSignIn

class LoginViewModel: ObservableObject {    
    let loginService = LoginService()
	let signInConfig = GIDConfiguration(clientID: "746460567328-7h3pc88t253q07h66snds9rebdjj0bqm.apps.googleusercontent.com")
    var settings: UserSettings = .shared
	@Published var loggedIn: Bool = false
	
    var cancellables = Set<AnyCancellable>()

	private func authenticateUser(for user: GIDGoogleUser?, with error: Error?) {
		if let error = error {
			print(error.localizedDescription)
			return
		}
		
		guard let authentication = user?.authentication, let idToken = authentication.idToken else { return }
		
		let credential = GoogleAuthProvider.credential(withIDToken: idToken, accessToken: authentication.accessToken)
		FirebaseAuthManager().login(credential: credential) {[weak self] (success, token) in
			if let token = token {
				self?.login(token)
			}
		}
	}
	
	func signinWithGoogle() {
		if GIDSignIn.sharedInstance.hasPreviousSignIn() {
			GIDSignIn.sharedInstance.restorePreviousSignIn { [unowned self] user, error in
				authenticateUser(for: user, with: error)
			}
		} else {
			guard let clientID = FirebaseApp.app()?.options.clientID else { return }
			let configuration = GIDConfiguration(clientID: clientID)
			guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene else { return }
			guard let rootViewController = windowScene.windows.first?.rootViewController else { return }
			GIDSignIn.sharedInstance.signIn(with: configuration, presenting: rootViewController) { [unowned self] user, error in
				authenticateUser(for: user, with: error)
			}
		}
	}
    
    func login(_ token: String) {
        loginService.login(token)
            .sink { completion in
                switch completion {
                case .failure(let error):
                    Logger.logDebug(error.localizedDescription)
                    SessionManage.shared.logout()
                    DispatchQueue.main.async {
                        self.settings.expride = true
						self.loggedIn = false
                    }
                case .finished:
					self.loggedIn = true
                }
            } receiveValue: { model in
                SessionManage.shared.storeAccessToken(model.access_token)
            }
            .store(in: &cancellables)
    }
}
