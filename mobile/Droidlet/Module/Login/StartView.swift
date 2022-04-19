import SwiftUI

struct StartView: View {
    @StateObject private var viewModel: LoginViewModel = LoginViewModel()
    @StateObject var settings: UserSettings = .shared

    var body: some View {
        if UserDefaults.standard.bool(forKey: "firstLaunch") == false {
			SessionManage.shared.logout()
		}
		
		if SessionManage.shared.accessToken != nil,
		   !settings.expride,
		   viewModel.loggedIn {
			return AnyView(ContentView())
		} else {
			return AnyView(LoginView(viewModel: viewModel))
		}
	}
}

struct StartView_Previews: PreviewProvider {
    static var previews: some View {
        StartView()
    }
}
