import SwiftUI

struct ControlsView: View {
    @ObservedObject var viewModel: HomeViewModel

    var body: some View {
        VStack {
            HStack(spacing: 10) {
				Button ("Logout"){
					viewModel.logout()
				}
				.buttonStyle(AccentFilledButton())
				.padding(.top, 20)
				.padding(.leading, 10)

                Spacer()
                
                Button {
                    viewModel.dashboardMode = .camera
                } label: {
                    Text(DashboardMode.camera.rawValue)
                        .modifier(Title(enable: viewModel.dashboardMode == .camera))
                }
                .padding(.trailing)
                
                Button {
                    viewModel.dashboardMode = .ar
                } label: {
                    Text(DashboardMode.ar.rawValue)
                        .modifier(Title(enable: viewModel.dashboardMode == .ar))
                }
                .padding(.trailing)
                
                Button {
                    viewModel.dashboardMode = .video
                } label: {
                    Text(DashboardMode.video.rawValue)
                        .modifier(Title(enable: viewModel.dashboardMode == .video))
                }
                .padding(.trailing)
            }

            Spacer()
        }
    }
}

struct ControlsView_Previews: PreviewProvider {
    static var previews: some View {
        ControlsView(viewModel: HomeViewModel())
    }
}
