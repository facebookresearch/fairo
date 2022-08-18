import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject var viewModel: HomeViewModel = HomeViewModel()
    @StateObject var settings: UserSettings = .shared
    @State var shouldNavigate: Bool = false
    @State var showARNavigation: Bool = false
    var body: some View {
        NavigationView {
            ZStack {
                switch viewModel.dashboardMode {
                case .camera:
                    ZStack {
                        CameraPreview(session: viewModel.session)
                            .edgesIgnoringSafeArea(.all)
                            .onAppear {
                                viewModel.startCamera()
                            }
                            .onDisappear {
                                // stop camera
                                viewModel.stopCamera()
                            }
                        VStack {
                            HStack(spacing: 5) {
                                Button ("Send"){
                                    viewModel.capturePhoto()
                                }
                                .buttonStyle(AccentFilledButton())
                                .padding(.top, 50)
                                .padding(.leading, 10)
                                
                                Spacer()
                            }
                            Spacer()
                        }.padding(.top, 30)
                    }
                    .fullScreenCover(isPresented: $viewModel.showEditPhotoVC) {
                        PhotoEditorViewControllerRepresentation(viewModel: viewModel, image: viewModel.image)
                    }
                    
                case .ar:
                    ZStack {
                        DroidletARViewContainer(viewModel: viewModel, isHome: true)
                            .edgesIgnoringSafeArea(.all)
                        VStack {
                            HStack(spacing: 5) {

                                Button ("Send"){
                                    viewModel.sendAnchor()
                                }
                                .buttonStyle(AccentFilledButton())
                                .padding(.leading, 10)
                                
                                Spacer()

                                VStack {
                                    NavigationLink(destination: NavigationARView(viewModel: viewModel).navigationBarHidden(true), isActive: $showARNavigation) {
                                        Button ("Navigation"){
                                            showARNavigation = true
                                        }
                                        .buttonStyle(AccentFilledButton())
                                        .padding(.top, 50)
                                        .padding(.trailing, 10)
                                    }
                                    
                                    NavigationLink(destination: NavigationARView(viewModel: viewModel).navigationBarHidden(true), isActive: $shouldNavigate) {
                                        EmptyView()
                                    }
                                    
                                    Button (viewModel.isODORunning ? "Stop Odometry" : "Start Odometry") {
                                        viewModel.isODORunning.toggle()
                                        viewModel.toggleOdometry()
                                        if !viewModel.isODORunning {
                                            viewModel.showingUploadAlert.toggle()
                                        }
                                    }
                                    .buttonStyle(AccentFilledButton())
                                    .padding(.top, 10)
                                    .padding(.trailing, 10)
                                    .alert(isPresented:$viewModel.showingUploadAlert) {
                                        Alert(
                                            title: Text("Odometry stopped"),
                                            message: Text("Upload the record?"),
                                            primaryButton: .default(Text("Upload")) {
                                                print("Uploading...")
                                                viewModel.uploadODORecord()
                                            },
                                            secondaryButton: .cancel()
                                        )
                                    }
                                }
                            }
                            Spacer()
                            VStack(alignment: .leading) {
                                HStack {
                                    Text("Feature points: ")
                                        .foregroundColor(.green)
                                    Text(viewModel.featurePoints)
                                        .foregroundColor(.green)
                                        .font(.headline)
                                }
                                .padding(.leading, 20)
                                
                                HStack {
                                    Text("Tracking status: ")
                                        .foregroundColor(.green)
                                    
                                    Text(viewModel.trackingStatus)
                                        .foregroundColor(.green)
                                        .font(.headline)
                                }
                                .padding(.leading, 20)
                                
                            }
                            .frame(width: UIScreen.main.bounds.width, alignment: .leading)
                            .padding(.bottom, 110)
                            .opacity(viewModel.isODORunning ? 1 : 0)
                        }
                        .padding(.top, 30)
                    }
                    .popover(isPresented: $viewModel.showImagePreview) {
                        PreviewImageView(viewModel: viewModel, shouldNavigate: $shouldNavigate)
                    }
                    .onReceive(NotificationCenter.default.publisher(for: Notification.Name("Notification"))) { _ in
                        let alert = settings.alert(message: "", inputText: "") { name in
                            let message_id = self.settings.notificationObject.message_id
                            HomeService.shared.markImage(value: .no, message_id: message_id, option: name)
                        }
                        
                        if let delegate = UIApplication.shared.delegate as? AppDelegate,
                           let parentViewController = delegate.window?.rootViewController {
                            parentViewController.present(alert, animated: true)
                        }
                    }
                    .onReceive(settings.$deviceToken) { token in
                        settings.sendTokenToBE(token)
                    }
                    .onReceive(settings.$reload, perform: { value in
                        if value {
                            viewModel.iosView.sceneView.scene.rootNode.childNodes.forEach { node in
                                if node.name ==  "test" {
                                    node.removeFromParentNode()
                                }
                            }
                        }
                    })
                case .video:
                    ZStack {
                        Image(uiImage: viewModel.depthImg ?? UIImage())
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                        
                        if viewModel.depthImg != nil {
                            VStack {
                                HStack(spacing: 5) {
                                    Button("Edit") {
                                        viewModel.showEditVideoVC = true
                                    }
                                    .buttonStyle(AccentFilledButton())
                                    .padding(.top, 50)
                                    .padding(.leading, 10)
                                    Spacer()
                                }
                                Spacer()
                            }
                        }
                    }
                    .fullScreenCover(isPresented: $viewModel.showEditVideoVC) {
                        PhotoEditorViewControllerRepresentation(viewModel: viewModel, image: viewModel.depthImg)
                    }
                }
                
                ControlsView(viewModel: viewModel)
                ChatView(viewModel: viewModel)
            }
            .onAppear() {
                DispatchQueue.main.asyncAfter(deadline: .now() + 1, execute: {
                    viewModel.startARSection()
                })
                if !viewModel.isSocketConnected {
                    viewModel.connectSocket()
                    settings.sendTokenToBE(settings.deviceToken)
                }
            }
            .onDisappear() {
                viewModel.pauseARSection()
            }
            .navigationBarHidden(true)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
